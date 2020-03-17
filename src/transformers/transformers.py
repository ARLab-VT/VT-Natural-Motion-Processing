import torch
import torch.nn as nn
import torch.nn.functional as F
from common.data_utils import *
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model//2]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MotionTransformerEncoderLayer(nn.Module):
    """
    Custom TransformerEncoderLayer for training of motion data. This layer does not contain
    the linear layer found the original Transformer model.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super(MotionTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src

class MotionTransformer(nn.Module):    
    def __init__(self, num_input_features, num_heads, dim_feedforward, dropout, num_layers, num_output_features, representation='quaternions'): 
        super(MotionTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(num_input_features, dropout)
        self.encoder_layer = MotionTransformerEncoderLayer(num_input_features, 
                                                        num_heads,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, norm=None)
        self.decoder = nn.Linear(num_input_features, num_output_features)
        self.src_mask = None
        self.representation = representation
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        pos_enc = self.pos_encoder(src)
        enc_output = self.encoder(pos_enc, self.src_mask)
        velocities = self.decoder(enc_output)

        last_pose = src[-1, :, :]
        output = torch.zeros_like(velocities).to(src.device)

        for i in range(velocities.shape[0]):
            new_pose = last_pose + velocities[i,:,:]

            if self.representation == 'quaternions':
                original_shape = new_pose.shape

                new_pose = new_pose.view(-1,4)
                new_pose = F.normalize(new_pose, p=2, dim=1).view(original_shape)
            
            output[i, :, :] = new_pose
            
            last_pose = new_pose    

        return output


class ConversionTransformer(nn.Module):    
    def __init__(self, num_input_features, num_heads, dim_feedforward, dropout, num_layers, num_output_features): 
        super(ConversionTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(num_input_features, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(num_input_features, 
                                                        num_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, norm=None)
        self.decoder = nn.Linear(num_input_features, num_output_features)
        self.src_mask = None
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        pos_enc = self.pos_encoder(src)
        output = self.encoder(pos_enc, self.src_mask)
        output = self.decoder(output)
        return output
