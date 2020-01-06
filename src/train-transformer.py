#!/usr/bin/env python
# coding: utf-8

from transformers.training_utils import *
from transformers.transformers import *
from common.data_utils import *
from common.logging import logger
from pathlib import Path
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset, RandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import sys
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
    parser.add_argument('--num-heads',
			help='number of heads in Transformer Encoder')
    parser.add_argument('--dim-feedforward',
			help='number of dimensions in feedforward layer in Transformer Encoder')
    parser.add_argument('--dropout',
			help='dropout percentage in Transformer Encoder')
    parser.add_argument('--num-layers',
			help='number of layers in Transformer Encoder')

    args = parser.parse_args()

    if args.data_path is None:
        parser.print_help()

    return args   

def read_h5_file(h5_file_path, task, seq_length, stride):
    X, y = None, None
    h5_file = h5py.File(h5_file_path, 'r')
    for filename in h5_file.keys():
        X_temp = h5_file[filename]['X']
        X_temp = discard_remainder(X_temp, 2*seq_length)

        if task == 'prediction':
            X_temp, y_temp = split_sequences(X_temp, seq_length, stride)
        elif task == 'conversion':
            y_temp = h5_file[filename]['Y']
            y_temp = discard_remainder(y_temp, 2*seq_length)
        else:
            logger.error("Task must be either prediction or conversion, found {}".format(task))
            sys.exit()
        
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
    
    logger.info("Starting Transformer training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seq_length = int(args.seq_length)
    stride = int(args.stride)
    lr = float(args.learning_rate)
 
    train_file_path = args.data_path + '/training.h5'
    
    X, y = read_h5_file(train_file_path, args.task, seq_length, stride)

    logger.info("{}, {}".format(X.shape, y.shape))

    encoder_feature_size = X.shape[1]
    decoder_feature_size = y.shape[1]

    val_file_path = args.data_path + '/validation.h5'    
    
    X_val, y_val = read_h5_file(val_file_path, args.task, seq_length, stride)   

    scaler = None
    y = reshape_to_sequences(y, seq_length)

    y_val = reshape_to_sequences(y_val, seq_length)
    
    X_scaler = RobustScaler().fit(np.append(X, X_val, axis=0))
    X = X_scaler.transform(X)
    X -= np.mean(X, axis=0)
    X = reshape_to_sequences(X, seq_length)

    X_val = X_scaler.transform(X_val)
    X_val -= np.mean(X_val, axis=0)
    X_val = reshape_to_sequences(X_val, seq_length)

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
    
    num_heads = int(args.num_heads)
    dim_feedforward = int(args.dim_feedforward)
    dropout = float(args.dropout)
    num_layers = int(args.num_layers)

    model = MotionTransformer(encoder_feature_size, num_heads, dim_feedforward, dropout, num_layers, decoder_feature_size).double()
    epochs = int(args.num_epochs)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, 0.95)
    dataloaders = (train_dataloader, val_dataloader)
    criterion = nn.L1Loss()
    validation_criterion = nn.L1Loss()
    
    logger.info("Model for training: {}".format(str(model)))
    logger.info("Optimizer for training: {}".format(str(optimizer)))
    logger.info("Criterion for training: {}".format(str(criterion)))

    fit(model, optimizer, scheduler, epochs, dataloaders, criterion, validation_criterion, scaler, device)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, args.model_file_path)
