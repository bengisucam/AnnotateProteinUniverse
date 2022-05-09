import torch
import torch.nn as nn
import torch
import numpy as np

input_dim = 21
hidden_dim = 100


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(hidden_dim, 200)
        self.fc2 = nn.Linear(200, 300)

    def forward(self, x):
        #print("enc x : ", x.shape)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        #print("enc lstm out:  ", lstm_out.shape)
        fc1_out = self.fc1(lstm_out)
        fc2_out = self.fc2(fc1_out)
        #print("enc fc1 out:  ", fc1_out.shape)
        return fc2_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(300, 200, 1, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(200, 100)

    def forward(self, x):
        #print("dec x : ", x.shape)
        lstm_out, _ = self.lstm(x)
        #print("dec lstm out:  ", lstm_out.shape)
        fc1_out = self.fc1(self.dropout(lstm_out))
        #print("dec fc1 out:  ", fc1_out.shape)
        prediction = torch.softmax(fc1_out, dim=2)
        return prediction


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        z = self.Encoder(x)
        prediction = self.Decoder(z)
        return prediction
