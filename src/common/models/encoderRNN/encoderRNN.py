import torch
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, dropout=0.0, bidirectional=False):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.directions, hidden_size)

    def forward(self, input):
        output, hidden = self.gru(input)
        output = self.dropout(output)

        if self.bidirectional:
            hidden = torch.tanh(
                self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))).unsqueeze(0)

        return output, hidden