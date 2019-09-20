#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../..')

from src.common.training_utils import *
from src.common.models import *
from src.common.data_utils import *
from pathlib import Path
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
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

    parser.add_argument('-f', '--data-file',
                        help='path to h5 file for reading data')
    parser.add_argument('--batch-size',
                        help='batch size for training', default=32)
    parser.add_argument('--num-epochs',
                        help='number of epochs for training', default=1)
    parser.add_argument('--bidirectional',
                        help='will use bidirectional encoder', default=False, action='store_true')
    parser.add_argument('--attention',
                        help='will use decoder with attention', default=False, action='store_true')

    args = parser.parse_args()

    if args.data_file is None:
        parser.print_help()

    return args


if __name__ == "__main__":

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_file = Path(args.data_file)
    f = h5py.File(args.data_file, 'r')

    X, y = torch.tensor(f['X']), torch.tensor(f['y'])
    train_indices, val_indices = f['train_indices'], f['val_indices']

    y_size = tuple(y.size())
    y = y.view(y_size[0] * y_size[1], -1)
    scaler = RobustScaler().fit(y)
    y = torch.tensor(scaler.transform(y))
    y = y.view(y_size)

    dataset = TensorDataset(X, y)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    datasets = (train_dataset, val_dataset)

    batch_size = int(args.batch_size)
    dataloaders = setup_dataloaders(datasets, batch_size)

    print("Number of training samples:", len(dataloaders[0].dataset))
    print("Number of validation samples:", len(dataloaders[1].dataset))

    encoder_feature_size = X.shape[-1]
    decoder_feature_size = y.shape[-1]

    encoder, encoder_optim = get_encoder(
        encoder_feature_size, device, bidirectional=args.bidirectional)

    if args.attention:
        decoder, decoder_optim = get_attn_decoder(
            decoder_feature_size, 'dot', device, bidirectional_encoder=args.bidirectional)
    else:
        decoder, decoder_optim = get_decoder(decoder_feature_size, device)

    models = (encoder, decoder)
    optims = (encoder_optim, decoder_optim)
    epochs = int(args.num_epochs)
    criterion = nn.L1Loss()

    fit(models, optims, epochs, dataloaders, criterion,
        scaler, device, use_attention=args.attention)
