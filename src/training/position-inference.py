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


torch.manual_seed(42)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path('../../../csv-files')
    filenames = os.listdir(data_path)

    orientation_requests = {'Orientation': [
        'T12', 'RightUpperArm', 'RightForeArm']}
    position_requests = {'Position': ['Pelvis', 'RightForeArm']}
    num_files = 1
    seq_length = 20

    orientation, orientation_scaler = read_data(
        filenames, data_path, num_files, orientation_requests, seq_length)

    position, position_scaler = read_data(
        filenames, data_path, num_files, position_requests,  seq_length, request_type='Position')

    position = np.flip(position, axis=1).copy()

    encoder_input_data, decoder_target_data = orientation, position

    batch_size = 32
    encoder_input_data = discard_remainder(encoder_input_data, batch_size)
    decoder_target_data = discard_remainder(decoder_target_data, batch_size)

    encoder_feature_size = encoder_input_data.shape[-1]
    decoder_feature_size = decoder_target_data.shape[-1]

    encoder_input_data = torch.tensor(encoder_input_data).to(device)
    decoder_target_data = torch.tensor(decoder_target_data).to(device)

    dataset = TensorDataset(encoder_input_data, decoder_target_data)

    train_size_approx = int(0.8 * len(dataset))
    train_size = train_size_approx - (train_size_approx % batch_size)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))

    epochs = 1

    dataloaders = (train_dataloader, val_dataloader)

    criterion = nn.L1Loss()

    encoder, encoder_opt = get_encoder(encoder_feature_size, device)
    decoder, decoder_opt = get_decoder(decoder_feature_size, device)

    models = (encoder, decoder)
    opts = (encoder_opt, decoder_opt)

    fit(models, opts, epochs, dataloaders, criterion, position_scaler)
