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
    parser.add_argument('--representation',
                        help='data representation (quaternions, expmap, rotmat)', default='quaternions')
    parser.add_argument('--full-transformer',
                        help='will use Transformer with both encoder and decoder if true, will only use encoder if false', default=False, action='store_true')
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
                        help='sequence length for model, will be downsampled if downsample is provided', default=20)
    parser.add_argument('--downsample',
                        help='reduce sampling frequency of recorded data; default sampling frequency is 240 Hz', default=1)
    parser.add_argument('--stride',
                        help='stride used when reading data in for running prediction tasks', default=3)
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
    seq_length = int(args.seq_length)//int(args.downsample)
    lr = float(args.learning_rate)

    normalize = True    
    train_dataloader, norm_data = load_dataloader(args, "training", normalize)
    val_dataloader, _ = load_dataloader(args, "validation", normalize, norm_data=norm_data)
 
    encoder_feature_size = train_dataloader.dataset[0][0].shape[1]
    decoder_feature_size = train_dataloader.dataset[0][1].shape[1]

    num_heads = int(args.num_heads)
    dim_feedforward = int(args.dim_feedforward)
    dropout = float(args.dropout)
    num_layers = int(args.num_layers)

   
    if args.full_transformer:
        model = InferenceTransformer(decoder_feature_size, num_heads, dim_feedforward, dropout, num_layers, decoder_feature_size, quaternions=True)
    else:
        model = InferenceTransformerEncoder(encoder_feature_size, num_heads, dim_feedforward, dropout, num_layers, decoder_feature_size, quaternions=True)    
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device).double()

    epochs = int(args.num_epochs)
    beta1 = float(args.beta_one)
    beta2 = float(args.beta_two)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=0.03) 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,3], gamma=0.1)

    dataloaders = (train_dataloader, val_dataloader)
    training_criterion = nn.L1Loss()
    validation_criteria = [nn.L1Loss()]
    
    if args.representation == 'expmap':
        validation_criteria.append(ExpmapToQuatLoss())

    logger.info("Model for training: {}".format(str(model)))
    logger.info("Number of parameters: {}".format(str(num_params)))
    logger.info("Optimizer for training: {}".format(str(optimizer)))
    logger.info("Criterion for training: {}".format(str(training_criterion)))

    fit(model, optimizer, scheduler, epochs, dataloaders, training_criterion, validation_criteria, device, args.model_file_path, full_transformer=args.full_transformer) 

    logger.info("Completed Training...")
    logger.info("\n")
