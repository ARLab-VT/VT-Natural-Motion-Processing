#!/usr/bin/env python
# coding: utf-8

from common.data_utils import *
import h5py
from pathlib import Path
import argparse
import os
import random
import sys
import math
import glob


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training',
                        help='participant numbers for training data, space separated; e.g., "W1 W2 W3"')
    
    parser.add_argument('--validation',
                        help='participant numbers for validation data, space separated; e.g., "P1 P2 P3"')

    parser.add_argument('--testing',
                        help='participant numbers for testing data, space separated; e.g., "P4 P5 P6"')

    parser.add_argument('-f', '--data-path',
                        help='path to h5 files for reading data')

    parser.add_argument('-o', '--output-path',
                        help='path to directory to save h5 files for training, validation, and testing')

    args = parser.parse_args()

    if args.training is None or args.validation is None or args.testing is None:
        print("Participant numbers for training, validation, or testing dataset were not provided.")
        parser.print_help()
        sys.exit()

    if args.data_path is None or args.output_path is None:
        print("Data path or output path were not provided.")
        parser.print_help()
        sys.exit()

    return args

def setup_filepaths(data_path, participant_numbers):
    all_filepaths = []
    for participant_number in participant_numbers:
        filepaths = glob.glob(data_path + participant_number + "*.h5")
        all_filepaths += filepaths
    return all_filepaths

if __name__ == "__main__":
    args = parse_args()
    
    train_filepaths = setup_filepaths(args.data_path, args.training.split(" "))
    val_filepaths = setup_filepaths(args.data_path, args.validation.split(" "))
    test_filepaths = setup_filepaths(args.data_path, args.testing.split(" "))

    filepath_groups = [(train_filepaths, "training"), (val_filepaths, "validation"), (test_filepaths, "testing")]

    experiment_setup = {'sensorOrientation' : 'X', 'jointAngle' : 'y'}
    requests = {'sensorOrientation' : ['all'], 'jointAngle' : ['all']}

    for filepaths, group in filepath_groups:
        print("File group:", group)
        h5_filename = args.output_path + "/" + group + ".h5"
        h5_file = h5py.File(h5_filename, 'w')

        dataset = read_h5(filepaths, requests)

        for filename in dataset.keys():
            for label in experiment_setup.keys():
                h5_file.create_dataset(filename + '/' + experiment_setup[label], 
                                       data=dataset[filename][label])
        h5_file.close()
