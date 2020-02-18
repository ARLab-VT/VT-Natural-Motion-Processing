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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path',
                        help='path to h5 files containing data (must contain training.h5 and validation.h5)')
    parser.add_argument('--batch-size',
                        help='batch size for training', default=32)
    parser.add_argument('--seq-length',
                        help='sequence length for encoder/decoder', default=20)
    parser.add_argument('--stride',
                        help='stride used when running prediction tasks', default=3)          

    args = parser.parse_args()

    if args.data_path is None:
        parser.print_help()

    return args

def zero_velocity(dataloaders, criterion, seq_length):
    name = ["Training", "Validation", "Testing"]
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    
    # zero velocity
    for i in range(len(dataloaders)):
        avg_loss = 0
        for index, data in enumerate(dataloaders[i], 0):
            inputs, targets = data
            final = inputs[:,-1,:].unsqueeze(1)
            predictions = final.repeat(1,seq_length,1)
            loss = criterion(targets, predictions)
            avg_loss += loss
        avg_loss = avg_loss / len(dataloaders[i])
        logger.info("{} average loss: {}".format(name[i], avg_loss))


def load_dataloader(args, type): 
    file_path = args.data_path + '/' + type + '.h5'
    seq_length = int(args.seq_length)
    batch_size = int(args.batch_size)
    stride = int(args.stride)
    
    X, y = read_variables(file_path, 'prediction', seq_length, stride)

    logger.info("{} shapes (X, y): {}, {}".format(type, X.shape, y.shape))

    y = reshape_to_sequences(y, seq_length)
    X = reshape_to_sequences(X, seq_length)

    logger.info("Reshaped {} shapes (X, y):{}, {}".format(type, X.shape, y.shape))
    
    X, y = torch.tensor(X), torch.tensor(y)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    logger.info("Number of {} samples: {}".format(type, len(dataset)))

    return dataloader


if __name__ == "__main__":
    args = parse_args()
    
    for arg in vars(args):
        logger.info("{} - {}".format(arg, getattr(args, arg)))
    
    logger.info("Starting Zero velocity...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seq_length = int(args.seq_length)
    batch_size = int(args.batch_size)

    train_dataloader = load_dataloader(args, "training")
    val_dataloader = load_dataloader(args, "validation")
    test_dataloader = load_dataloader(args, "testing")

    criterion = ShapeLoss()
    logger.info("Criterion for zero velocity: {}".format(str(criterion)))
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    zero_velocity(dataloaders, criterion, seq_length)
