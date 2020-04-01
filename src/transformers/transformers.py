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

class InferenceTransformerEncoder(nn.Module):    
    def __init__(self, num_input_features, num_heads, dim_feedforward, dropout, num_layers, num_output_features, quaternions=False): 
        super(InferenceTransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(num_input_features, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(num_input_features, 
                                                        num_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, norm=None)
        self.linear1 = nn.Linear(num_input_features, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, num_output_features)
        self.quaternions = quaternions

    def forward(self, src):
        pos_enc = self.pos_encoder(src)
        enc_output = self.encoder(pos_enc)
        output = self.linear2(self.linear1(enc_output))

        if self.quaternions:
            original_shape = output.shape

            output = output.view(-1,4)
            output = F.normalize(output, p=2, dim=1).view(original_shape)

        return output


class InferenceTransformer(nn.Module):    
    def __init__(self, num_input_features, num_heads, dim_feedforward, dropout, num_layers, num_output_features, quaternions=False): 
        super(InferenceTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(num_input_features, dropout)
        self.pos_decoder = PositionalEncoding(num_output_features, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(num_input_features, 
                                                        num_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, norm=None)
        self.decoder_layer = nn.TransformerDecoderLayer(num_input_features,
                                                  num_heads,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers, norm=None)
        self.tgt_mask = None
        self.quaternions = quaternions

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, target):
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(target):
            self.tgt_mask = self.generate_square_subsequent_mask(len(target)).to(target.device)

        pos_enc = self.pos_decoder(src)
        memory = self.encoder(pos_enc)

        pos_dec = self.pos_decoder(target)
        output = self.decoder(target, memory, tgt_mask=self.tgt_mask)
        
        if self.quaternions:
            original_shape = output.shape
            output = F.normalize(output.view(-1,4), p=2, dim=1).view(original_shape)

        return output
