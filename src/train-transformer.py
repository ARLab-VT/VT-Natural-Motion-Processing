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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
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


if __name__ == "__main__":
    args = parse_args()
    
    for arg in vars(args):
        logger.info("{} - {}".format(arg, getattr(args, arg)))

    logger.info("Starting Transformer training...")
    
    logger.info("Device count: {}".format(str(torch.cuda.device_count())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on {}...".format(device))    
    seq_length = int(args.seq_length)
    stride = int(args.stride)
    lr = float(args.learning_rate)
 
    train_file_path = args.data_path + '/training.h5'
    
    X, y = read_variables(train_file_path, args.task, seq_length, stride)

    encoder_feature_size = X.shape[1]
    decoder_feature_size = y.shape[1]

    val_file_path = args.data_path + '/validation.h5'    
    
    X_val, y_val = read_variables(val_file_path, args.task, seq_length, stride)   
    
    scaler = None
    y = reshape_to_sequences(y, seq_length)
 
    y_val = reshape_to_sequences(y_val, seq_length)
    
    X = reshape_to_sequences(X, seq_length)

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

    model = MotionTransformer(encoder_feature_size, num_heads, dim_feedforward, dropout, num_layers, decoder_feature_size)

    logger.info("Device count: {}".format(str(torch.cuda.device_count())))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device).float()

    epochs = int(args.num_epochs)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)#, momentum=0.9) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, 1.0)
    dataloaders = (train_dataloader, val_dataloader)
    lambdas, stats = (1,1,1), (1,1)
    training_criterion = MotionLoss(lambdas, stats)
    validation_criterion = nn.L1Loss()
    
    logger.info("Model for training: {}".format(str(model)))
    logger.info("Optimizer for training: {}".format(str(optimizer)))
    logger.info("Criterion for training: {}".format(str(training_criterion)))

    fit(model, optimizer, scheduler, epochs, dataloaders, training_criterion, validation_criterion, scaler, device)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, args.model_file_path)
