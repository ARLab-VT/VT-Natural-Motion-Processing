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
	                    help='initial learning rate for training', default=0.001)
    parser.add_argument('--beta-one',
	                    help='beta1 for adam optimizer (momentum)', default=0.9)
    parser.add_argument('--beta-two',
	                    help='beta2 for adam optimizer', default=0.999)
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
    
    X_val, y_val = read_variables(val_file_path, args.task, seq_length, 80)   
    
    scaler = None
    X = X.view(-1, seq_length, X.shape[1])
    X_val = X_val.view(-1, seq_length, X_val.shape[1])

    y = y.view(-1, seq_length, y.shape[1])
    y_val = y_val.view(-1, seq_length, y_val.shape[1])

    logger.info("Training shapes (X, y): {}, {}".format(X.shape, y.shape))
    logger.info("Validation shapes (X, y): {}, {}".format(X_val.shape, y_val.shape))
    memory = X.element_size() * X.nelement() + y.element_size() * y.nelement() + X_val.element_size() * X_val.nelement() + y_val.element_size() * y_val.nelement()  
    logger.info("Memory requirement for training and validation: {}".format(memory))

    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = int(args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    num_heads = int(args.num_heads)
    dim_feedforward = int(args.dim_feedforward)
    dropout = float(args.dropout)
    num_layers = int(args.num_layers)

    if args.task == 'prediction':
        model = MotionTransformer(encoder_feature_size, num_heads, dim_feedforward, dropout, num_layers, decoder_feature_size)
    else:
        model = ConversionTransformer(encoder_feature_size, num_heads, dim_feedforward, dropout, num_layers, decoder_feature_size)    

    logger.info("Device count: {}".format(str(torch.cuda.device_count())))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device).float()

    epochs = int(args.num_epochs)
    beta1 = float(args.beta_one)
    beta2 = float(args.beta_two)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2)) 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,8,12], gamma=0.3)
    dataloaders = (train_dataloader, val_dataloader)
    training_criterion = MotionLoss(1.0, 0.0, 1.0)
    validation_criterion = nn.L1Loss()
    
    logger.info("Model for training: {}".format(str(model)))
    logger.info("Optimizer for training: {}".format(str(optimizer)))
    logger.info("Criterion for training: {}".format(str(training_criterion)))

    fit(model, optimizer, scheduler, epochs, dataloaders, training_criterion, validation_criterion, scaler, device, args.model_file_path) 
