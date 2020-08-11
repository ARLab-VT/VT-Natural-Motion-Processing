# Copyright (c) 2020-present, Assistive Robotics Lab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Seq2Seq Encoders and Decoders.

Reference:
  [1] https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import torch
from torch import nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    """An encoder for seq2seq architectures."""

    def __init__(self, input_size, hidden_size,
                 dropout=0.0, bidirectional=False):
        """Initialize encoder for use with decoder in seq2seq architecture.

        Args:
            input_size (int): Number of features in the input
            hidden_size (int): Number of hidden units in the GRU layer
            dropout (float, optional): Dropout applied after GRU layer.
                Defaults to 0.0.
            bidirectional (bool, optional): Whether encoder is bidirectional.
                Defaults to False.
        """
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        self.directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.directions, hidden_size)

    def forward(self, input):
        """Forward pass through encoder.

        Args:
            input (torch.tensor): (seq_len,
                                   batch_size,
                                   input_size)

        Returns:
            tuple: Returns output and hidden state of decoder.
                output (torch.Tensor): (seq_len,
                                        batch_size,
                                        directions*hidden_size)
                hidden (torch.Tensor): (1, batch_size, hidden_size)
        """
        output, hidden = self.gru(input)
        output = self.dropout(output)

        if self.bidirectional:
            hidden = torch.tanh(
                self.fc(torch.cat((hidden[-2, :, :],
                                   hidden[-1, :, :]), dim=1))).unsqueeze(0)

        return output, hidden


class DecoderRNN(nn.Module):
    """A decoder for use in seq2seq architectures."""

    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        """Initialize DecoderRNN.

        Args:
            input_size (int): number of features in input
            hidden_size (int): number of hidden units in GRU layer
            output_size (int): number of features in output
            dropout (float, optional): Dropout applied after GRU layer.
                Defaults to 0.0.
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """Forward pass through decoder.

        Args:
            input (torch.Tensor): input batch to pass through RNN
                (1, batch_size, input_size)
            hidden (torch.Tensor): hidden state of the decoder
                (1, batch_size, hidden_size)

        Returns:
            tuple: Returns output and hidden state of decoder.
                output (torch.Tensor): (1, batch_size, output_size)
                hidden (torch.Tensor): (1, batch_size, hidden_size)
        """
        output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        output = self.out(output)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    """A decoder with an attention layer for use in seq2seq architectures."""

    def __init__(self, output_size, feature_size, enc_hidden_size,
                 dec_hidden_size, attention, bidirectional_encoder=False):
        """Initialize AttnDecoderRNN.

        Args:
            output_size (int): size of output
            features_size (int): number of features in input batch
            enc_hidden_size (int): hidden size of encoder
            dec_hidden_size (int): hidden size of decoder
            attention (str): attention method for use in Attention layer
            bidirectional_encoder (bool, optional): Whether encoder used with
                decoder is bidirectional. Defaults to False.
        """
        super().__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.attention = attention

        self.directions = 2 if bidirectional_encoder else 1

        self.rnn = nn.GRU(self.directions * enc_hidden_size + feature_size,
                          dec_hidden_size)

        self.out = nn.Linear(self.directions * enc_hidden_size +
                             dec_hidden_size + feature_size,
                             output_size)

    def forward(self, input, hidden, annotations):
        """Forward pass through decoder.

        Args:
            input (torch.Tensor): (1, batch_size, feature_size)
            hidden (torch.Tensor): (1, batch_size, dec_hidden_size)
            annotations (torch.Tensor): (seq_len,
                                         batch_size,
                                         directions * enc_hidden_size)

        Returns:
            tuple: Returns output and hidden state of decoder.
                output (torch.Tensor): (1, batch_size, output_size)
                hidden (torch.Tensor): (1, batch_size, dec_hidden_size)
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

        # output = [batch_size, output_size]

        return output.unsqueeze(0), hidden


class Attention(nn.Module):
    """An Attention layer for the AttnDecoder with multiple methods."""

    def __init__(self, hidden_size, batch_size,
                 method, bidirectional_encoder=False):
        """Initialize Attention layer.

        Args:
            hidden_size (int): Size of hidden state in decoder.
            batch_size (int): Size of batch, used for shape checks.
            method (str): Attention technique/method to use. Supports
                general, biased-general, activated-general, dot, add, and
                concat.
            bidirectional_encoder (bool, optional): Whether encoder used with
                decoder is bidirectional. Defaults to False.
        """
        super().__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.method = method
        self.directions = 2 if bidirectional_encoder else 1

        if method in ["general", "biased-general", "activated-general"]:
            bias = not(method == "general")
            self.Wa = nn.Linear(hidden_size,
                                self.directions * hidden_size,
                                bias=bias)
        elif method == "add":
            self.Wa = nn.Linear((self.directions * hidden_size),
                                hidden_size,
                                bias=False)
            self.Wb = nn.Linear(hidden_size,
                                hidden_size,
                                bias=False)
            self.va = nn.Parameter(torch.rand(hidden_size))
        elif method == "concat":
            self.Wa = nn.Linear((self.directions * hidden_size) + hidden_size,
                                hidden_size, bias=False)
            self.va = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, annotations):
        """Forward pass through attention layer.

        Args:
            hidden (torch.Tensor): (1, batch_size, hidden_size)
            annotations (torch.Tensor): (seq_len,
                                           batch_size,
                                           directions * hidden_size)

        Returns:
            torch.Tensor: (batch_size, seq_len)
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
        """Compute an attention score with hidden state and annotations.

        Args:
            hidden (torch.Tensor): (batch_size, hidden_size)
            annotations (torch.Tensor): (batch_size,
                                         seq_len,
                                         directions * hidden_size)

        Returns:
            torch.Tensor: (batch_size, seq_len)
        """
        if "general" in self.method:
            x = self.Wa(hidden)

            x = x.unsqueeze(-1)

            score = annotations.bmm(x)

            if self.method == "activated-general":
                score = torch.tanh(score)

            assert list(score.shape) == [self.batch_size,
                                         self.seq_len,
                                         1]

            score = score.squeeze(-1)

            return score

        elif self.method == "dot":
            hidden = hidden.unsqueeze(-1)

            hidden = hidden.repeat(1, self.directions, 1)

            score = annotations.bmm(hidden)

            assert list(score.shape) == [self.batch_size,
                                         self.seq_len,
                                         1]

            score = score.squeeze(-1)

            return score

        elif self.method == "add":
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

        elif self.method == "concat":
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
