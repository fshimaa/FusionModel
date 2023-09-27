import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class LSTM (nn.Module):
    def __init__(self , num_classes=101):
        super (LSTM , self).__init__ ()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, in_features):
        hidden = None
        out, hidden = self.lstm(in_features)
        x = self.fc1(out[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x





