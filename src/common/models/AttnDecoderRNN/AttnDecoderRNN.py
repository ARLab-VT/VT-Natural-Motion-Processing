import torch
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F


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

    def forward(self, input, hidden, encoder_outputs):
        """
        Computes an attention score
        :param input: (1, batch_size, enc_hidden_dim)
        :param hidden: (1, batch_size, dec_hidden_dim)
        :param encoder_outputs: (seq_len, batch_size, hidden_dim)
        :return: output (1, batch_size, output_dim) and hidden (1, batch_size, dec_hidden_dim)
        """

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, seq len, enc hid dim]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim]

        rnn_input = torch.cat((input, weighted), dim=2)

        # rnn_input = [1, batch size, enc hid dim + hid dim]

        output, hidden = self.rnn(rnn_input, hidden)

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        input = input.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, input), dim=1))

        # output = [bsz, output dim]

        return output.unsqueeze(0), hidden
