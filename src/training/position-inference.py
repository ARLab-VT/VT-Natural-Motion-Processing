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
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import math
import random
import os
import time
import argparse

torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--csv-files',
                        help='path to csv files for reading data')
    parser.add_argument('--bidirectional',
                        help='will use bidirectional encoder', default=False, action='store_true')
    parser.add_argument('--attention',
                        help='will use decoder with attention', default=False, action='store_true')

    args = parser.parse_args()

    if args.csv_files is None:
        parser.print_help()

    return args


if __name__ == "__main__":

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.csv_files)
    filenames = os.listdir(data_path)

    orientation_requests = {'Orientation': [
        'T12', 'RightUpperArm', 'RightForeArm']}
    position_requests = {'Position': ['Pelvis', 'RightForeArm']}
    num_files = 1
    seq_length = 20

    orientation, orientation_scaler = read_data(
        filenames, data_path, num_files, orientation_requests, seq_length)

    position, position_scaler = read_data(
        filenames, data_path, num_files, position_requests, seq_length, request_type='Position')

    position = np.flip(position, axis=1).copy()

    encoder_input_data, decoder_target_data = orientation, position

    batch_size = 32
    encoder_input_data = discard_remainder(encoder_input_data, batch_size)
    decoder_target_data = discard_remainder(decoder_target_data, batch_size)

    encoder_input_data = torch.tensor(encoder_input_data).to(device)
    decoder_target_data = torch.tensor(decoder_target_data).to(device)

    split_size = 0.8
    dataloaders = setup_dataloaders(
        encoder_input_data, decoder_target_data, batch_size, split_size)

    print("Number of training samples:", len(dataloaders[0].dataset))
    print("Number of validation samples:", len(dataloaders[1].dataset))

    encoder_feature_size = encoder_input_data.shape[-1]
    decoder_feature_size = decoder_target_data.shape[-1]

    encoder, encoder_optim = get_encoder(
        encoder_feature_size, device, bidirectional=args.bidirectional)

    if args.attention:
        decoder, decoder_optim = get_attn_decoder(
            decoder_feature_size, 'dot', device, bidirectional_encoder=args.bidirectional)
    else:
        decoder, decoder_optim = get_decoder(decoder_feature_size, device)

    models = (encoder, decoder)
    optims = (encoder_optim, decoder_optim)
    epochs = 1
    criterion = nn.L1Loss()

    fit(models, optims, epochs, dataloaders, criterion,
        position_scaler, use_attention=args.attention)
