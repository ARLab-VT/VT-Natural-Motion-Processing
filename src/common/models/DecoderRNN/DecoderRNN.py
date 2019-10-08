import torch
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, dropout=0.0):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.gru = nn.GRU(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        output = self.out(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)