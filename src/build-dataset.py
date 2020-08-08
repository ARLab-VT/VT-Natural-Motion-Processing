#!/usr/bin/env python
# coding: utf-8

from common.data_utils import read_h5
from common.logging import logger
import h5py
import argparse
import sys
import glob
import numpy as np


def parse_args():
    """Parse arguments for module.

    Returns:
        argparse.Namespace: contains accessible arguments passed in to module
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--training",
                        help=("participants for training, space separated; "
                              "e.g., W1 W2"))
    parser.add_argument("--validation",
                        help=("participants for validation, space separated; "
                              "e.g., P1 P2"))
    parser.add_argument("--testing",
                        help=("participants for testing, space separated; "
                              "e.g., P4 P5"))
    parser.add_argument("-f",
                        "--data-path",
                        help="path to h5 files for reading data")
    parser.add_argument("-o", "--output-path",
                        help=("path to directory to save h5 files for "
                              "training, validation, and testing"))
    parser.add_argument("-x", "--task-input",
                        help=("input type; "
                              "e.g., orientation, relativePosition, "
                              "or jointAngle"))
    parser.add_argument("--input-label-request",
                        help=("input label requests, space separated; "
                              "e.g., all or Pelvis RightForearm"))
    parser.add_argument("-y", "--task-output",
                        help="output type; e.g., orientation or jointAngle")
    parser.add_argument("--output-label-request",
                        help=("output label requests, space separated; "
                              "e.g., all or jRightElbow"))
    parser.add_argument("--aux-task-output",
                        help=("auxiliary task output in addition "
                              "to regular task output"))
    parser.add_argument("--aux-output-label-request",
                        help="aux output label requests, space separated")

    args = parser.parse_args()

    if None in [args.training, args.validation, args.testing]:
        logger.info(("Participant numbers for training, validation, "
                     "or testing dataset were not provided."))
        parser.print_help()
        sys.exit()

    if None in [args.data_path, args.output_path]:
        logger.error("Data path or output path were not provided.")
        parser.print_help()
        sys.exit()

    if None in [args.task_input, args.input_label_request, args.task_output]:
        logger.error(("Task input and label requests "
                      "or task output were not given."))
        parser.print_help()
        sys.exit()

    if args.output_label_request is None:
        if args.task_input == args.task_output:
            logger.info("Will create h5 files with input data only.")
        else:
            logger.error("Label output requests were not given for the task.")
            parser.print_help()
            sys.exit()

    if args.aux_task_output == args.task_output:
        logger.error("Auxiliary task should not be the same as the main task.")
        parser.print_help()
        sys.exit()

    if (args.aux_task_output is not None and
            args.aux_output_label_request is None):
        logger.error("Need auxiliary output labels if using aux output task")
        parser.print_help()
        sys.exit()

    if args.task_input == args.task_output:
        if args.output_label_request is None:
            logger.info(("Will create h5 files with only input "
                         "data for self-supervision tasks..."))
        else:
            logger.info("Will create h5 files with input and output data.")

    return args


def setup_filepaths(data_path, participant_numbers):
    """Set up filepaths for reading in participant .h5 files.

    Args:
        data_path (str): path to directory containing .h5 files
        participant_numbers (list): participant numbers for filepaths

    Returns:
        list: filepaths to all of the .h5 files
    """
    all_filepaths = []
    for participant_number in participant_numbers:
        filepaths = glob.glob(data_path + "/" + participant_number + "_*.h5")
        all_filepaths += filepaths
    return all_filepaths


def map_requests(tasks, labels):
    """Generate a dict of tasks mapped to labels.

    Args:
        tasks (list): list of tasks/groups that will be mapped to labels
        labels (list): list of labels that will be the value for each task

    Returns:
        dict: dictionary mapping each task to the list of labels
    """
    requests = dict(map(lambda e: (e, labels), tasks))
    return requests


def write_dataset(filepath_groups, variable, experiment_setup, requests):
    """Write to data to training, validation, testing .h5 files.

    Args:
        filepath_groups (list): list of tuples associate files to data group
        variable (str): the machine learning variable X or Y to be written to
        experiment_setup (dict): map to reference task for variable
        requests (dict): requests to read from files to store with variable
    """
    for filepaths, group in filepath_groups:
        logger.info(f"Writing {variable} to the {group} set...")
        h5_filename = args.output_path + "/" + group + ".h5"
        with h5py.File(h5_filename, "a") as h5_file:
            dataset = read_h5(filepaths, requests)
            for filename in dataset.keys():
                temp_dataset = None
                for j, data_group in enumerate(experiment_setup[variable]):
                    if temp_dataset is None:
                        temp_dataset = dataset[filename][data_group]
                    else:
                        temp_dataset = np.append(temp_dataset,
                                                 dataset[filename][data_group],
                                                 axis=1)
                try:
                    h5_file.create_dataset(filename + "/" + variable,
                                           data=temp_dataset)
                except KeyError:
                    logger.info(f"{filename} does not contain {data_group}")


if __name__ == "__main__":
    args = parse_args()

    train_filepaths = setup_filepaths(args.data_path, args.training.split(" "))
    val_filepaths = setup_filepaths(args.data_path, args.validation.split(" "))
    test_filepaths = setup_filepaths(args.data_path, args.testing.split(" "))

    filepath_groups = [(train_filepaths, "training"),
                       (val_filepaths, "validation"),
                       (test_filepaths, "testing")]

    task_input = args.task_input.split(" ")
    input_label_request = args.input_label_request.split(" ")

    task_output = args.task_output.split(" ")
    if args.output_label_request is not None:
        output_label_request = args.output_label_request.split(" ")

    if (args.task_input == args.task_output
            and args.output_label_request is None):
        experiment_setup = {"X": task_input}
        requests = map_requests(task_input, input_label_request)

        write_dataset(filepath_groups, "X", experiment_setup, requests)
    else:
        experiment_setup = {"X": task_input, "Y": task_output}

        input_requests = map_requests(task_input, input_label_request)
        output_requests = map_requests(task_output, output_label_request)

        if args.aux_task_output is not None:
            aux_task_output = args.aux_task_output.split(" ")
            aux_output_label_request = args.aux_output_label_request.split(" ")
            experiment_setup["Y"] += aux_task_output
            aux_output_requests = map_requests(aux_task_output,
                                               aux_output_label_request)
            output_requests.update(aux_output_requests)

        write_dataset(filepath_groups, "X", experiment_setup, input_requests)
        write_dataset(filepath_groups, "Y", experiment_setup, output_requests)
