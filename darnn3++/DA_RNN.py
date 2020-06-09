from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import concatenate
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math
import sklearn.metrics as metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = '../nasdaq100/nasdaq100_padding.csv'
dataset = pd.read_csv(filename)
#print(dataset.values.shape)
#(40560,82)

class EncoderAtt(nn.Module):
    def __init__(self, input_size, hidden_size, T):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(EncoderAtt, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data):
        # input_data: batch_size * T - 1 * input_size
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.hidden_size).zero_())
        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data) # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim = 2) # batch_size * input_size * (2*hidden_size + T - 1)
            # Eqn. 9: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1)) # (batch_size * input_size) * 1
            attn_weights = F.softmax(x.view(-1, self.input_size)) # batch_size * input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size * input_size
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class DecoderAtt(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        super(DecoderAtt, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                        nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T - 1 * encoder_hidden_size
        # y_history: batch_size * (T-1)
        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2), input_encoded), dim=2)
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                 )).view(-1, self.T - 1))  # batch_size * T - 1, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # batch_size * encoder_hidden_size
            if t < self.T - 1:
                # Eqn. 15
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim=1))  # batch_size * 1
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size
        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim=1))
        # self.logger.info("hidden %s context %s y_pred: %s", hidden[0][0][:10], context[0][:10], y_pred[:10])
        return y_pred

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())

def preprocess_data(dat, col_names):
    #对数据进行归一化和标准化   https://blog.csdn.net/u011734144/article/details/84066784
    # scale 包含了均值和方差。proc_dat就是利用均值和方差得到的标准化后的数据
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]
    return feats, targs, scale

# Train the model
class da_rnn:
    def __init__(self, file_data, encoder_hidden_size=64, decoder_hidden_size=64, T=10,
                 learning_rate=0.01, batch_size=128, parallel=True, debug=False):
        self.T = T
        dat = pd.read_csv(file_data, nrows=100 if debug else None)
        self.X, self.y, self.scaler = preprocess_data(dat, ("NDX",))
        self.y = (self.y).reshape((self.y).shape[0],)
        self.batch_size = batch_size
        self.encoder = EncoderAtt(input_size=self.X.shape[1], hidden_size=encoder_hidden_size, T=T).to(device)
        self.decoder = DecoderAtt(encoder_hidden_size=encoder_hidden_size, decoder_hidden_size=decoder_hidden_size, T=T).to(device)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                           lr=learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr=learning_rate)
        self.train_size = int(self.X.shape[0] * 0.8)

    def train(self, n_epochs=10):
        iter_per_epoch = int(np.ceil(self.train_size * 1. / self.batch_size))
        print("Iterations per epoch: %3.3f ~ %d.", self.train_size * 1. / self.batch_size, iter_per_epoch)
        self.iter_losses = np.zeros(n_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(n_epochs)

        self.loss_func = nn.MSELoss()
        min_loss = 100000
        n_iter = 0

        learning_rate = 1.

        for i in range(n_epochs):
            perm_idx = np.random.permutation(self.train_size - self.T)
            j = 0
            while j < self.train_size:
                batch_idx = perm_idx[j:(j + self.batch_size)]
                X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
                y_history = np.zeros((len(batch_idx), self.T - 1))
                y_target = self.y[batch_idx + self.T]

                for k in range(len(batch_idx)):
                    X[k, :, :] = self.X[batch_idx[k] : (batch_idx[k] + self.T - 1), :]
                    y_history[k, :] = self.y[batch_idx[k]: (batch_idx[k] + self.T - 1)]

                loss = self.train_iteration(X, y_history, y_target)
                self.iter_losses[int(i * iter_per_epoch + j / self.batch_size)] = loss
                #if (j / self.batch_size) % 50 == 0:
                #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / self.batch_size, loss)
                j += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

            self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
            print("Epoch %d, loss: %3.10f." % (i, self.epoch_losses[i]))

            # with torch.no_grad():
            #     if i % 10 == 0:
            #         y_val_pred = self.predict(on_train=False,on_val=True)
            #         true = concatenate(((self.y).reshape(self.y.shape[0], 1), self.X), axis=1)
            #         true = self.scaler.inverse_transform(true)
            #         y_true = true[self.train_size:self.train_size+int(dataset.shape[0]*0.1), 0]
            #
            #         val_loss = metrics.mean_squared_error(y_true,y_val_pred)
            #
            #         if val_loss < min_loss:
            #             min_loss = val_loss
            #             print("save model " , i, "loss:" , min_loss)
            #             #torch.save(model.state_dict(), 'model.pth')
            #             torch.save(self.encoder.state_dict(), 'models/encoder.pth')
            #             torch.save(self.decoder.state_dict(), 'models/decoder.pth')
            if self.epoch_losses[i] < min_loss:
                min_loss = self.epoch_losses[i]
                print("save model " , i, "loss:" , self.epoch_losses[i])
                torch.save(self.encoder.state_dict(), 'models/encoder.pth')
                torch.save(self.decoder.state_dict(), 'models/decoder.pth')

        y_train_pred = self.predict(on_train=True)
        y_test_pred = self.predict(on_train=False)
        y_pred = np.concatenate((y_train_pred, y_test_pred))
        plt.figure()
        plt.plot(range(1, 1 + len(self.y)), self.y, label="True")
        plt.plot(range(self.T, len(y_train_pred) + self.T), y_train_pred, label='Predicted - Train')
        plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1), y_test_pred, label='Predicted - Test')
        plt.legend(loc='upper left')
        plt.savefig('results/final_result.png')


    def train_iteration(self, X, y_history, y_target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(device)))
        y_pred = self.decoder(input_encoded, Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(device)))
        y_pred = y_pred.view(-1)
        y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor).to(device))
        loss = self.loss_func(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def predict(self, path_en = '',path_de = '',on_train = False,on_val=False):
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)
        if on_val:
            y_pred = np.zeros(int(dataset.shape[0]*0.1))
        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_size - self.T,  batch_idx[j]+ self.train_size - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(device))
            # if(path_en != ''):
            #     self.encoder.load_state_dict(torch.load(path_en))
            #     self.decoder.load_state_dict(torch.load(path_de))
            #     print('加载模型成功')
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(device)))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size
        return y_pred



io_dir = '../nasdaq100/nasdaq100_padding.csv'
model = da_rnn(file_data='{}'.format(io_dir), parallel=False, learning_rate=.001)

model.train(n_epochs=400)
y_pred = model.predict('models/encoder.pth','models/decoder.pth')




plt.figure()
plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
plt.show()

plt.figure()
plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
plt.show()

plt.figure()
plt.plot(y_pred, label = 'Predicted')
plt.plot(model.y[model.train_size:], label = "True")
plt.legend(loc = 'upper left')
plt.show()


MAE = sum(abs(model.y[model.train_size:]-y_pred))/(dataset.values.shape[0]-model.train_size)
print("-----MAE:",MAE)
RMSE = math.pow((sum(np.power((model.y[model.train_size:]-y_pred),2))/(dataset.values.shape[0]-model.train_size)),1/2)
print("-----RMSE:",RMSE)
MAPE = sum(abs((model.y[model.train_size:]-y_pred)/model.y[model.train_size:]))/(dataset.values.shape[0]-model.train_size)
print("-----MAPE:",MAPE)


mae1 = metrics.mean_absolute_error(model.y[model.train_size:],y_pred)
mse1 = metrics.mean_squared_error(model.y[model.train_size:],y_pred)
mape1 = np.mean(np.abs((y_pred - model.y[model.train_size:]) / model.y[model.train_size:])) * 100
print(mae1,np.sqrt(mse1),mape1)