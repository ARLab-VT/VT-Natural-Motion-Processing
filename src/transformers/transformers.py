# Copyright (c) 2020-present, Assistive Robotics Lab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Transformer classes with quaternion normalization.

Reference:
  [1] https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """PositionalEncoding injects position-dependent signals in input/targets.

    Transformers have no concept of sequence like RNNs, so this acts to inject
    information about the order of a sequence.

    Useful reference for more info:
    https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model//2]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class InferenceTransformerEncoder(nn.Module):
    """Transformer Encoder for use in inferring unit quaternions."""

    def __init__(self, num_input_features, num_heads, dim_feedforward, dropout,
                 num_layers, num_output_features, quaternions=False):
        """Initialize the Transformer Encoder.

        Args:
            num_input_features (int): number of features in the input
                data
            num_heads (int): number of heads in each layer for multi-head
                attention
            dim_feedforward (int): dimensionality of the feedforward layers in
                each layer
            dropout (float): dropout amount in the layers
            num_layers (int): number of layers in Encoder
            num_output_features (int): number of features in the output
                data
            quaternions (bool, optional): whether quaternions are used in the
                output; will normalize if True. Defaults to False.
        """
        super(InferenceTransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(num_input_features, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(num_input_features,
                                                        num_heads,
                                                        dim_feedforward,
                                                        dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers,
                                             norm=None)
        self.linear1 = nn.Linear(num_input_features, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, num_output_features)
        self.quaternions = quaternions

    def forward(self, src):
        """Forward pass through Transformer Encoder model for training/testing.

        Args:
            src (torch.Tensor): used by encoder to generate output

        Returns:
            torch.Tensor: output from the Transformer
        """
        pos_enc = self.pos_encoder(src)
        enc_output = self.encoder(pos_enc)
        output = self.linear2(self.linear1(enc_output))

        if self.quaternions:
            original_shape = output.shape

            output = output.view(-1, 4)
            output = F.normalize(output, p=2, dim=1).view(original_shape)

        return output


class InferenceTransformer(nn.Module):
    """Transformer for use in inferring unit quaternions."""

    def __init__(self, num_features, num_heads, dim_feedforward, dropout,
                 num_layers, quaternions=False):
        """Initialize the Transformer model.

        Args:
            num_features (int): number of features in the input and target
                data
            num_heads (int): number of heads in each layer for multi-head
                attention
            dim_feedforward (int): dimensionality of the feedforward layers in
                each layer
            dropout (float): dropout amount in the layers
            num_layers (int): number of layers in Encoder and Decoder
            quaternions (bool, optional): whether quaternions are used in the
                output; will normalize if True. Defaults to False.
        """
        super(InferenceTransformer, self).__init__()

        self.pos_encoder = PositionalEncoding(num_features, dropout)
        self.pos_decoder = PositionalEncoding(num_features, dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(num_features,
                                                        num_heads,
                                                        dim_feedforward,
                                                        dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers,
                                             norm=None)
        self.decoder_layer = nn.TransformerDecoderLayer(num_features,
                                                        num_heads,
                                                        dim_feedforward,
                                                        dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer,
                                             num_layers,
                                             norm=None)
        self.tgt_mask = None
        self.quaternions = quaternions

    def generate_square_subsequent_mask(self, sz):
        """Mask the upcoming values in the tensor to avoid cheating.

        Args:
            sz (int): sequence length of tensor

        Returns:
            torch.Tensor: mask of subsequent entries during forward pass
        """
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def inference(self, memory, target):
        """Forward pass through Transformer model at validation/test time.

        Args:
            memory (torch.Tensor): memory passed from the Encoder
            target (torch.Tensor): predictions built up over time during
                inference

        Returns:
            torch.Tensor: output from the model (passed into model in next
                iteration)
        """
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(target):
            self.tgt_mask = self.generate_square_subsequent_mask(len(target))
            self.tgt_mask = self.tgt_mask.to(target.device)

        pos_dec = self.pos_decoder(target)
        output = self.decoder(pos_dec, memory, tgt_mask=self.tgt_mask)

        if self.quaternions:
            original_shape = output.shape
            output = F.normalize(output.view(-1, 4),
                                 p=2,
                                 dim=1).view(original_shape)

        return output

    def forward(self, src, target):
        """Forward pass through Transformer model for training.

        Use inference function at validation/test time to get accurate
        measure of performance.

        Args:
            src (torch.Tensor): used by encoder to generate memory
            target (torch.Tensor): target for decoder to try to match;
                Transformers use teacher forcing so targets are used as input

        Returns:
            torch.Tensor: output from the Transformer
        """
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(target):
            self.tgt_mask = self.generate_square_subsequent_mask(len(target))
            self.tgt_mask = self.tgt_mask.to(target.device)

        pos_enc = self.pos_decoder(src)
        memory = self.encoder(pos_enc)

        pos_dec = self.pos_decoder(target)
        output = self.decoder(pos_dec, memory, tgt_mask=self.tgt_mask)

        if self.quaternions:
            original_shape = output.shape
            output = F.normalize(output.view(-1, 4),
                                 p=2,
                                 dim=1).view(original_shape)

        return output
