#!/usr/bin/env python
# coding: utf-8

from common.data_utils import *
from common.logging import logger
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
    
    parser.add_argument('-x', '--task-input',
                        help='input type; e.g., "orientation", "relativePosition", or "jointAngle"')

    parser.add_argument('--input-label-request',
                        help='input label requests, space separated; e.g., "all" or "Pelvis RightForearm"')    

    parser.add_argument('-y', '--task-output',
                        help='output type; e.g., "orientation or "jointAngle"')

    parser.add_argument('--output-label-request',
                        help='output label requests, space separated; e.g., "all" or "jRightElbow"')
    

    args = parser.parse_args()

    if args.training is None or args.validation is None or args.testing is None:
        logger.info("Participant numbers for training, validation, or testing dataset were not provided.")
        parser.print_help()
        sys.exit()

    if args.data_path is None or args.output_path is None:
        logger.info("Data path or output path were not provided.")
        parser.print_help()
        sys.exit()

    if args.task_input is None or args.input_label_request is None or args.task_output is None:
        logger.info("Task input and label requests or task output were not given.")
        parser.print_help()
        sys.exit()

    if args.task_input == args.task_output:
        if args.output_label_request is None:
            logger.info("Will create h5 files with only input data for self-supervision tasks...")
        else:
            logger.error("Need output label requests to be None if task input is equal to task output.")
            parser.print_help()
            sys.exit()
    else:
        if args.output_label_request is None:
            logger.info("Label output requests were not given for the task.")
            parser.print_help()
            sys.exit()

    return args

def setup_filepaths(data_path, participant_numbers):
    all_filepaths = []
    for participant_number in participant_numbers:
        filepaths = glob.glob(data_path + "/" + participant_number + "_*.h5")
        all_filepaths += filepaths
    return all_filepaths

def map_requests(task, labels):
    requests = dict(map(lambda e: (e, labels), task))
    return requests

def write_dataset(filepath_groups, variable, experiment_setup, requests):
    for filepaths, group in filepath_groups:
        logger.info("Writing {} to the {} file group...".format(variable, group))
        h5_filename = args.output_path + "/" + group + ".h5"
        h5_file = h5py.File(h5_filename, 'w')
        dataset = read_h5(filepaths, requests)
        for filename in dataset.keys():
                temp_dataset = None
                for j, data_group in enumerate(experiment_setup[variable]):
                    if temp_dataset is None:
                        temp_dataset = dataset[filename][data_group]
                    else:
                        temp_dataset = np.append(temp_dataset, dataset[filename][data_group], axis=1)
                try:
                    h5_file.create_dataset(filename + '/' + variable, 
                                           data=temp_dataset)
                except KeyError:
                    logger.info("Dataset retrieved from {} does not contain {}".format(filename, data_group))
        h5_file.close()

if __name__ == "__main__":
    args = parse_args()
    
    train_filepaths = setup_filepaths(args.data_path, args.training.split(" "))
    val_filepaths = setup_filepaths(args.data_path, args.validation.split(" "))
    test_filepaths = setup_filepaths(args.data_path, args.testing.split(" "))

    filepath_groups = [(train_filepaths, "training"), (val_filepaths, "validation"), (test_filepaths, "testing")]

    task_input = args.task_input.split(" ")
    input_label_request = args.input_label_request.split(" ")

    task_output = args.task_output.split(" ")
    output_label_request = args.output_label_request.split(" ")

    if args.task_input == args.task_output:
        experiment_setup = {"X" : task_input}
        requests = map_requests(task_input, input_label_request)

        write_dataset(filepath_groups, "X", experiment_setup, requests)
    else:
        experiment_setup = {"X" : args.task_input.split(" "), "Y" : args.task_output.split(" ")}

        input_requests = map_requests(task_input, input_label_request)
        output_requests = map_requests(task_output, output_label_request)
        
        write_dataset(filepath_groups, "X", experiment_setup, input_requests)
        write_dataset(filepath_groups, "Y", experiment_setup, output_requests)         
    
