import torch
import torch.nn as nn
import torch
import numpy as np

input_dim = 21
hidden_dim = 100


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out


class LSTMModel(nn.Module):
    def __init__(self,LSTM):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM

    def forward(self, x):
        lstm_out = self.lstm(x)
        prediction = torch.softmax(lstm_out, dim=2)
        return prediction
