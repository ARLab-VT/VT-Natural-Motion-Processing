#!/usr/bin/env python
# coding: utf-8

from common.training_utils import *
from common.models import *
from common.data_utils import *
from pathlib import Path
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset, RandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import math
import random
import os
import time
import argparse
import h5py
from common.logging import logger

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()

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

def read_X_and_y(h5_file_path, seq_length):
    X, y = None, None
    h5_file = h5py.File(h5_file_path, 'r')
    for filename in h5_file.keys():
        X_temp = h5_file[filename]['X']
        y_temp = h5_file[filename]['Y']

        X_temp = discard_remainder(X_temp, seq_length)
        y_temp = discard_remainder(y_temp, seq_length)

        if X is None and y is None:
            X = X_temp
            y = y_temp
        else:
            X = np.append(X, X_temp, axis=0)
            y = np.append(y, y_temp, axis=0)
    h5_file.close()
    return X, y   

if __name__ == "__main__":

    args = parse_args()
    
    for arg in vars(args):
        logger.info("{} - {}".format(arg, getattr(args, arg)))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seq_length = int(args.seq_length)
    lr = float(args.learning_rate)
 
    train_file_path = args.data_path + '/training.h5'
    X, y = read_X_and_y(train_file_path, seq_length)

    encoder_feature_size = X.shape[1]
    decoder_feature_size = y.shape[1]

    val_file_path = args.data_path + '/validation.h5'    
    X_val, y_val = read_X_and_y(val_file_path, seq_length)

    #scaler = RobustScaler().fit(np.append(y, y_val, axis=0))
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(np.append(y, y_val, axis=0))
    y = scaler.transform(y)
    y = reshape_to_sequences(y, seq_length=seq_length)
    y = np.flip(y, axis=1).copy()

    y_val = scaler.transform(y_val)
    y_val = reshape_to_sequences(y_val, seq_length=seq_length)
    y_val = np.flip(y_val, axis=1).copy()

    #X_scaler = RobustScaler().fit(np.append(X, X_val, axis=0))
    #X = X_scaler.transform(X)
    X -= np.mean(X, axis=0)
    X = reshape_to_sequences(X, seq_length=seq_length)

    #X_val = X_scaler.transform(X_val)
    X_val -= np.mean(X_val, axis=0)
    X_val = reshape_to_sequences(X_val, seq_length=seq_length)

    X, y = torch.tensor(X), torch.tensor(y)
    X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
    
    
    logger.info("Training shapes (X, y): {}, {}".format(X.shape, y.shape))
    logger.info("Validation shapes (X, y): {}, {}".format(X_val.shape, y_val.shape))

    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = int(args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    logger.info("Number of training samples: {}".format(len(train_dataset)))
    logger.info("Number of validation samples: {}".format(len(val_dataset)))

    encoder, encoder_optim = get_encoder(encoder_feature_size,
                                         device,
                                         hidden_size=int(args.hidden_size),
                                         lr=lr,
                                         bidirectional=args.bidirectional)

    use_attention = False
    if args.attention in ['add', 'concat', 'general', 'activated-general', 'biased-general']:
        decoder, decoder_optim = get_attn_decoder(decoder_feature_size,
                                                  args.attention,
                                                  device,
                                                  hidden_size=int(args.hidden_size),
                                                  lr=lr,
                                                  bidirectional_encoder=args.bidirectional)
        use_attention = True
    else:
        decoder, decoder_optim = get_decoder(decoder_feature_size,
                                             device,
                                             hidden_size=int(args.hidden_size),
                                             lr=lr)

    models = (encoder, decoder)
    optims = (encoder_optim, decoder_optim)
    dataloaders = (train_dataloader, val_dataloader)
    epochs = int(args.num_epochs)
    criterion = nn.L1Loss()

    fit(models, optims, epochs, dataloaders, criterion,
        scaler, device, use_attention=use_attention)
    
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizerA_state_dict': encoder_optim.state_dict(),
        'optimizerB_state_dict': decoder_optim.state_dict(),
    }, args.model_file_path)
