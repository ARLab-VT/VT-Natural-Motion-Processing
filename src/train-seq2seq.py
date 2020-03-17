#!/usr/bin/env python
# coding: utf-8

from seq2seq.training_utils import *
from seq2seq.seq2seq import *
from common.data_utils import *
from common.logging import *
from pathlib import Path
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset, RandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import random
import os
import time
import argparse
import h5py

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--task',
                        help='task for neural network to train on; either prediction or conversion')
    parser.add_argument('--data-path',
                        help='path to h5 files containing data (must contain training.h5 and validation.h5)')
    parser.add_argument('--model-file-path',
                        help='path to model file for saving it after training')
    parser.add_argument('--batch-size',
                        help='batch size for training', default=32)
    parser.add_argument('--learning-rate',
                        help='learning rate for encoder and decoder', default=0.001)
    parser.add_argument('--seq-length',
                        help='sequence length for encoder/decoder', default=20)    
    parser.add_argument('--stride',
                        help='stride used when running prediction tasks', default=3)
    parser.add_argument('--num-epochs',
                        help='number of epochs for training', default=1)
    parser.add_argument('--hidden-size',
                        help='hidden size in both the encoder and decoder')
    parser.add_argument('--bidirectional',
                        help='will use bidirectional encoder', default=False, action='store_true')
    parser.add_argument('--attention',
                        help='will use decoder with attention with this method', default='general')

    args = parser.parse_args()

    if args.data_path is None:
        parser.print_help()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    for arg in vars(args):
        logger.info("{} - {}".format(arg, getattr(args, arg)))
    
    logger.info("Starting seq2seq model training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seq_length = int(args.seq_length)
    stride = int(args.stride)
    lr = float(args.learning_rate)
 
    train_file_path = args.data_path + '/training.h5'
    
    X, y = read_variables(train_file_path, args.task, seq_length, stride)

    logger.info("{}, {}".format(X.shape, y.shape))

    encoder_feature_size = X.shape[1]
    decoder_feature_size = y.shape[1]

    val_file_path = args.data_path + '/validation.h5'    
    
    X_val, y_val = read_variables(val_file_path, args.task, seq_length, stride)   

    scaler = None
    y = y.view(-1, seq_length, y.shape[1])
    #y = np.flip(y, axis=1).copy() 

    y_val = y_val.view(-1, seq_length, y_val.shape[1])
    #y_val = np.flip(y_val, axis=1).copy()    

    X = X.view(-1, seq_length, X.shape[1])

    X_val = X_val.view(-1, seq_length, X_val.shape[1])

    logger.info("Training shapes (X, y): {}, {}".format(X.shape, y.shape))
    logger.info("Validation shapes (X, y): {}, {}".format(X_val.shape, y_val.shape))    

    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = int(args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    logger.info("Number of training samples: {}".format(len(train_dataset)))
    logger.info("Number of validation samples: {}".format(len(val_dataset)))

    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = int(args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    encoder = get_encoder(encoder_feature_size,
                          device,
                          hidden_size=int(args.hidden_size),
                          bidirectional=args.bidirectional)

    encoder_optim = optim.AdamW(encoder.parameters(), lr=lr) 

    use_attention = False
    if args.attention in ['add', 'concat', 'general', 'activated-general', 'biased-general']:
        decoder = get_attn_decoder(decoder_feature_size,
                                   args.attention,
                                   device,
                                   hidden_size=int(args.hidden_size),
                                   bidirectional_encoder=args.bidirectional)
        use_attention = True
    else:
        decoder = get_decoder(decoder_feature_size,
                              device,
                              hidden_size=int(args.hidden_size))

    decoder_optim = optim.AdamW(decoder.parameters(), lr=lr) 

    
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)    
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    models = (encoder, decoder)
    optims = (encoder_optim, decoder_optim)
    dataloaders = (train_dataloader, val_dataloader)
    epochs = int(args.num_epochs)
    criterion = nn.L1Loss()

    logger.info("Encoder for training: {}".format(str(encoder)))
    logger.info("Decoder for training: {}".format(str(decoder)))
    logger.info("Number of parameters: {}".format(str(encoder_params + decoder_params)))
    logger.info("Optimizers for training: {}".format(str(encoder_optim)))
    logger.info("Criterion for training: {}".format(str(criterion)))

    fit(models, optims, epochs, dataloaders, criterion,
        scaler, device, args.model_file_path, use_attention=use_attention)
    
