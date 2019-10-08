#!/usr/bin/env python
# coding: utf-8

from common.data_utils import *
import torch
from torch import nn, optim, Tensor
import h5py
from pathlib import Path
import argparse
import os
import random
import sys
import math


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--csv-files',
                        help='path to csv files for reading data')

    parser.add_argument('-o', '--output-file-path',
                        help='path to directory to save h5 file')

    parser.add_argument('--max-file-count',
                        help='max number of files to place into a single h5 file.',
                        default=5)

    parser.add_argument('--seq-length',
                        help='sequence length of prepared data.')

    parser.add_argument('--batch-size',
                        help='batch size for dataset',
                        default=32)

    parser.add_argument('--split-size',
                        help='split size for training and validation dataset',
                        default=0.8)

    args = parser.parse_args()

    if args.csv_files is None:
        parser.print_help()

    return args


if __name__ == "__main__":
    args = parse_args()

    data_path = Path(args.csv_files)
    filenames = os.listdir(data_path)
    random.shuffle(filenames)
    num_files = len(filenames)
    max_file_count = int(args.max_file_count)

    num_file_groups = math.ceil(num_files / max_file_count)
    offsets = [max_file_count*i for i in range(num_file_groups)]

    filename_groups = [[filename for filename in filenames[offset:offset +
                                                           max_file_count]] for offset in offsets]

    for i, filenames in enumerate(filename_groups):
        print("File group:", i+1)
        orientation_requests = {'Orientation': [
            'T12', 'RightUpperArm', 'RightForeArm']}
        position_requests = {'Position': ['Pelvis', 'RightForeArm']}
        seq_length = int(args.seq_length)

        orientation = read_data(
            filenames, data_path, orientation_requests, seq_length)

        position = read_data(
            filenames, data_path, position_requests, seq_length, request_type='Position')

        position = np.flip(position, axis=1).copy()

        batch_size = int(args.batch_size)
        orientation = discard_remainder(orientation, batch_size)
        position = discard_remainder(position, batch_size)

        orientation = torch.tensor(orientation)
        position = torch.tensor(position)

        split_size = float(args.split_size)
        dataset, train_indices, val_indices = setup_datasets(
            orientation, position, batch_size, split_size)

        h5_filename = args.output_file_path + "-" + str(i)
        f = h5py.File(h5_filename, 'w')
        f.create_dataset('X', data=orientation)
        f.create_dataset('y', data=position)
        f.create_dataset('train_indices', data=train_indices)
        f.create_dataset('val_indices', data=val_indices)
        f.close()
