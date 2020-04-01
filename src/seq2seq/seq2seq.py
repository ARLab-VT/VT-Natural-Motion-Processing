import torch
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, bidirectional=False):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
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


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        output = self.out(output)
        return output + input, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, features_dim, enc_hidden_size, dec_hidden_size, attention, bidirectional_encoder=False):
        super().__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.output_dim = output_dim
        self.features_dim = features_dim
        self.attention = attention

        self.directions = 2 if bidirectional_encoder else 1

        self.rnn = nn.GRU(self.directions * enc_hidden_size +
                          features_dim, dec_hidden_size)

        self.out = nn.Linear(self.directions * enc_hidden_size +
                             dec_hidden_size + features_dim, output_dim)

    def forward(self, input, hidden, annotations):
        """
        Computes an attention score
        :param input: (1, batch_size, directions * enc_hidden_dim)
        :param hidden: (1, batch_size, dec_hidden_dim)
        :param annotations: (seq_len, batch_size, hidden_dim)
        :return: output (1, batch_size, output_dim) and hidden (1, batch_size, dec_hidden_dim)
        """

        attention = self.attention(hidden, annotations)

        attention = attention.unsqueeze(1)

        annotations = annotations.permute(1, 0, 2)

        context_vector = torch.bmm(attention, annotations)

        context_vector = context_vector.permute(1, 0, 2)

        rnn_input = torch.cat((input, context_vector), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        # assert torch.all(torch.isclose(output, hidden))

        input = input.squeeze(0)
        output = output.squeeze(0)
        context_vector = context_vector.squeeze(0)

        output = self.out(torch.cat((output, context_vector, input), dim=1))

        # output = [batch_size, output_dim]

        return output.unsqueeze(0), hidden    


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_size, method, bidirectional_encoder=False):
        super().__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.method = method
        self.directions = 2 if bidirectional_encoder else 1

        if method in ['general', 'biased-general', 'activated-general']:
            bias = not(method == 'general')
            self.Wa = nn.Linear(hidden_size,
                                self.directions * hidden_size,
                                bias=bias)
        elif method == 'add':
            self.Wa = nn.Linear((self.directions * hidden_size),
                                hidden_size,
                                bias=False)
            self.Wb = nn.Linear(hidden_size,
                                hidden_size,
                                bias=False)
            self.va = nn.Parameter(torch.rand(hidden_size))
        elif method == 'concat':
            self.Wa = nn.Linear((self.directions * hidden_size) + hidden_size,
                                hidden_size, bias=False)
            self.va = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, annotations):
        """
        Computes an attention score
        :param hidden: (1, batch_size, hidden_size)
        :param annotations: (seq_len, batch_size, directions * hidden_size)
        :return: a softmax score (batch_size)
        """

        assert list(hidden.shape) == [1, self.batch_size, self.hidden_size]

        assert self.batch_size == annotations.shape[1]
        assert self.directions * self.hidden_size == annotations.shape[2]
        self.seq_len = annotations.shape[0]

        hidden = hidden.squeeze(0)

        assert list(hidden.shape) == [self.batch_size, self.hidden_size]

        annotations = annotations.permute(1, 0, 2)

        assert list(annotations.shape) == [self.batch_size,
                                           self.seq_len,
                                           self.directions * self.hidden_size]

        score = self._score(hidden, annotations)

        assert list(score.shape) == [self.batch_size, self.seq_len]

        return F.softmax(score, dim=1)

    def _score(self, hidden, annotations):
        """
        Computes an attention score
        :param hidden: (batch_size, hidden_size)
        :param annotations: (batch_size, seq_len, directions * hidden_size)
        :return: a score (batch_size, seq_len)
        """

        if 'general' in self.method:
            x = self.Wa(hidden)

            x = x.unsqueeze(-1)

            score = annotations.bmm(x)

            if self.method == 'activated-general':
                score = torch.tanh(score)

            assert list(score.shape) == [self.batch_size,
                                         self.seq_len,
                                         1]

            score = score.squeeze(-1)

            return score

        elif self.method == 'dot':
            hidden = hidden.unsqueeze(-1)     
            
            hidden = hidden.repeat(1, self.directions, 1)
            
            score = annotations.bmm(hidden)
            
            assert list(score.shape) == [self.batch_size,
                                         self.seq_len,
                                         1]
           
            score = score.squeeze(-1)

            return score

        elif self.method == 'add':
            x1 = self.Wa(annotations)

            x2 = self.Wb(hidden)

            x2 = x2.unsqueeze(1)

            x2 = x2.repeat(1, self.seq_len, 1)

            energy = x1 + x2

            energy = energy.permute(0, 2, 1)

            assert list(energy.shape) == [self.batch_size,
                                          self.hidden_size,
                                          self.seq_len]

            va = self.va.repeat(self.batch_size, 1).unsqueeze(1)

            score = torch.bmm(va, energy)

            assert list(score.shape) == [self.batch_size,
                                         1,
                                         self.seq_len]

            score = score.squeeze(1)

            return score

        elif self.method == 'concat':
            hidden = hidden.unsqueeze(1)

            hidden = hidden.repeat(1, self.seq_len, 1)

            energy = torch.tanh(self.Wa(torch.cat((hidden, annotations), 2)))

            energy = energy.permute(0, 2, 1)

            assert list(energy.shape) == [self.batch_size,
                                          self.hidden_size,
                                          self.seq_len]

            va = self.va.repeat(self.batch_size, 1).unsqueeze(1)

            assert list(va.shape) == [self.batch_size,
                                      1,
                                      self.hidden_size]

            score = torch.bmm(va, energy)

            assert list(score.shape) == [self.batch_size,
                                         1,
                                         self.seq_len]

            score = score.squeeze(1)

            return score
