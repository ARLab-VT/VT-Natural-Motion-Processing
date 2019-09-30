import torch
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F



"""
Modified from (https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb) and (https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53), which are great tutorials on different seq2seq models and attention. The Attention module uses Luong et al. This is also known as "general" attention.

"""


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_size, method, bidirectional_encoder=False):
        super().__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.method = method
        self.directions = 2 if bidirectional_encoder else 1

        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(
                hidden_size, self.directions * hidden_size, bias=False)
        elif method == 'concat':
            self.Wa = nn.Linear((2 * hidden_size) + hidden_size,
                                hidden_size, bias=False)
            self.va = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        """
        Computes an attention score
        :param hidden: (1, batch_size, hidden_dim)
        :param encoder_outputs: (seq_len, batch_size, directions * hidden_dim)
        :return: a softmax score (batch_size)
        """

        assert list(hidden.shape) == [1, self.batch_size, self.hidden_size]

        assert self.batch_size == encoder_outputs.shape[1]
        assert self.directions * self.hidden_size == encoder_outputs.shape[2]
        self.seq_len = encoder_outputs.shape[0]

        hidden = hidden.squeeze(0)

        assert list(hidden.shape) == [self.batch_size, self.hidden_size]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        assert list(encoder_outputs.shape) == [
            self.batch_size, self.seq_len, self.directions * self.hidden_size]

        score = self._score(hidden, encoder_outputs)

        assert list(score.shape) == [self.batch_size, self.seq_len]

        return F.softmax(score, dim=1)

    def _score(self, hidden, encoder_outputs):
        """
        Computes an attention score
        :param hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, seq_len, hidden_dim)
        :return: a score (batch_size, seq_len)
        """

        if self.method == 'dot':
            hidden = hidden.unsqueeze(-1)
            hidden = hidden.repeat(1, self.directions, 1)
            score = encoder_outputs.bmm(hidden).squeeze(-1)
            return score

        elif self.method == 'general':
            x = self.Wa(hidden)
            x = x.unsqueeze(-1)
            score = encoder_outputs.bmm(x).squeeze(-1)
            return score

        elif self.method == 'concat':
            hidden = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
            energy = torch.tanh(
                self.Wa(torch.cat((hidden, encoder_outputs), 2)))
            energy = energy.permute(0, 2, 1)
            va = self.va.repeat(self.batch_size, 1).unsqueeze(1)
            score = torch.bmm(va, energy).squeeze(1)
            return score
