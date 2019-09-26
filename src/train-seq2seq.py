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

    parser.add_argument('-f', '--data-files-path',
                        help='path to h5 files containing data')
    parser.add_argument('--batch-size',
                        help='batch size for training', default=32)
    parser.add_argument('--encoder-feature-size',
                        help='encoder feature size for task', default=12)
    parser.add_argument('--decoder-feature-size',
                        help='decoder feature size for task', default=3)
    parser.add_argument('--num-epochs',
                        help='number of epochs for training', default=1)
    parser.add_argument('--bidirectional',
                        help='will use bidirectional encoder', default=False, action='store_true')
    parser.add_argument('--attention',
                        help='will use decoder with attention', default=False, action='store_true')

    args = parser.parse_args()

    if args.data_files_path is None:
        parser.print_help()

    return args


if __name__ == "__main__":

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_feature_size = int(args.encoder_feature_size)
    decoder_feature_size = int(args.decoder_feature_size)

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

    data_path = Path(args.data_files_path)
    filenames = os.listdir(data_path)

    for i, filename in enumerate(filenames):
        file_path = ('/'.join([args.data_files_path, filename]))
        data_file = Path(file_path)
        f = h5py.File(data_file, 'r')

        X, y = f['X'], f['y']
        train_indices, val_indices = f['train_indices'], f['val_indices']

        y_size = y.shape
        y = np.reshape(y, (y_size[0] * y_size[1], -1))
        scaler = RobustScaler().fit(y)
        y = scaler.transform(y)
        y = np.reshape(y, y_size)

        X_size = X.shape
        X = np.reshape(X, (X_size[0] * X_size[1], -1))
        X -= np.mean(X, axis=0)
        X = np.reshape(X, X_size)

        X, y = torch.tensor(X), torch.tensor(y)

        dataset = TensorDataset(X, y)

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        datasets = (train_dataset, val_dataset)

        batch_size = int(args.batch_size)
        dataloaders = setup_dataloaders(datasets, batch_size)

        print("File number:", i+1)
        print("Number of training samples:", len(dataloaders[0].dataset))
        print("Number of validation samples:", len(dataloaders[1].dataset))

        fit(models, optims, epochs, dataloaders, criterion,
            scaler, device, use_attention=args.attention)
