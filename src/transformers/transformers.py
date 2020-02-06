import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        print(pe.shape)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model//2]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MotionTransformer(nn.Module):    
    def __init__(self, num_input_features, num_heads, dim_feedforward, dropout, num_layers, num_output_features): 
        super(MotionTransformer, self).__init__()
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
        enc_output = self.encoder(pos_enc, self.src_mask)
        velocities = self.decoder(enc_output)
        
        last_pose = src[-1, :, :]
        output = torch.zeros_like(velocities).to(src.device)
        
        for i in range(velocities.shape[0]):
            output[i, :, :] = last_pose + velocities[i, :, :]
            last_pose = output[i, :, :]
        
        return output

class Discriminator(nn.Module):
    def __init__(self, num_input_features, num_heads, dim_feedforward, dropout, num_layers, seq_length):
        super(Discriminator, self).__init__()
        self.pos_encoder = PositionalEncoding(num_input_features, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(num_input_features,
                                                        num_heads,
                                                        dim_feedforward,
                                                        dropout,
                                                        'relu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, norm=None)
        self.decoder = nn.Linear(num_input_features, 1)
        self.src_mask = None
        self.output_shape = (seq_length, 1)


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
        src = self.pos_encoder(src)
        output = self.encoder(src, self.src_mask)
        return self.decoder(output) 
