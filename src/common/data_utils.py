# Copyright (c) 2020-present, Assistive Robotics Lab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import warnings
import h5py
import math
import os
from .logging import logger
import sys
from torch.utils.data import TensorDataset, DataLoader


class XSensDataIndices:
    """XSensDataIndices helps with retrieving data.

    XSens has a data layout which requires interfacing data measurements
    (position, orientation, etc.) with places (RightLowerLeg, RightWrist)
    where those measurements are available.
    """

    def __init__(self):
        """Initialize object for use in reading data from .h5 files."""
        sensor_group = ["sensorFreeAcceleration",
                        "sensorMagneticField",
                        "sensorOrientation"]

        segment_group = ["position", "normPositions",
                         "velocity", "acceleration",
                         "normAcceleration", "sternumNormAcceleration",
                         "angularVelocity", "angularAcceleration",
                         "orientation", "normOrientation",
                         "sternumNormOrientation"]

        joint_group = ["jointAngle", "jointAngleXZY"]

        joint_ergo_group = ["jointAngleErgo", "jointAngleErgoXZY"]

        groups = [sensor_group, segment_group,
                  joint_group, joint_ergo_group]
        self._labels_to_items(groups)

    def __call__(self, requests):
        """Retrieve indices using a request.

        Requests are dicts that use groups like "position" as keys and
        labels like ["Pelvis"]. Requests can be passed to an instance of
        XSensDataIndices to retrieve the indices for the labels of each group.

        Args:
            requests (dict): maps groups to desired labels and will retrieve
                indices for those group labels.

        Returns:
            dict: a map of the groups to the indices for the labels.
        """
        label_indices = {}
        for label, items in requests.items():
            if label in self.label_items:
                label_indices[label] = self._request(label, items)
        return label_indices

    def _labels_to_items(self, groups):
        self.label_items = {}
        sensors = ["Pelvis", "T8", "Head", "RightShoulder", "RightUpperArm",
                   "RightForeArm", "RightHand", "LeftShoulder", "LeftUpperArm",
                   "LeftForeArm", "LeftHand", "RightUpperLeg", "RightLowerLeg",
                   "RightFoot", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot"]

        segments = ["Pelvis", "L5", "L3", "T12",
                    "T8", "Neck", "Head",
                    "RightShoulder", "RightUpperArm",
                    "RightForeArm", "RightHand",
                    "LeftShoulder", "LeftUpperArm",
                    "LeftForeArm", "LeftHand",
                    "RightUpperLeg", "RightLowerLeg",
                    "RightFoot", "RightToe",
                    "LeftUpperLeg", "LeftLowerLeg",
                    "LeftFoot", "LeftToe"]

        joints = ["jL5S1", "jL4L3", "jL1T12",
                  "jT9T8", "jT1C7", "jC1Head",
                  "jRightT4Shoulder", "jRightShoulder",
                  "jRightElbow", "jRightWrist",
                  "jLeftT4Shoulder", "jLeftShoulder",
                  "jLeftElbow", "jLeftWrist",
                  "jRightHip", "jRightKnee",
                  "jRightAnkle", "jRightBallFoot",
                  "jLeftHip", "jLeftKnee",
                  "jLeftAnkle", "jLeftBallFoot"]

        ergo_joints = ["T8_Head", "T8_LeftUpperArm", "T8_RightUpperArm",
                       "Pelvis_T8", "Vertical_Pelvis", "Vertical_T8"]

        item_groups = [sensors, segments, joints, ergo_joints]

        for index, group in enumerate(groups):
            for label in group:
                self.label_items[label] = item_groups[index]

    def _request(self, req_label, req_items):
        valid_items = self.label_items[req_label]

        if "all" in req_items:
            req_items = valid_items

        num_valid_items = len(valid_items)
        orientation_groups = ["orientation", "normOrientation",
                              "sternumNormOrientation"]
        dims = 4 if req_label in orientation_groups else 3

        indices = [list(range(i, i+dims))
                   for i in range(0, dims*num_valid_items, dims)]

        index_map = dict(zip(valid_items, indices))

        return self._find_indices(index_map, req_items)

    def _find_indices(self, index_map, items):
        mapped_indices = []

        for item in items:
            if item in index_map:
                mapped_indices.append(index_map[item])
            else:
                warnings.warn("Requested item {} not in file.".format(item))

        return mapped_indices


def discard_remainder(data, seq_length):
    """Discard data that does not fit inside sequence length.

    Args:
        data (np.ndarray): data to truncate
        seq_length (int): sequence length to find data that doesn"t fit into
            sequences

    Returns:
        np.ndarray: truncated data
    """
    new_row_num = data.shape[0] - (data.shape[0] % seq_length)
    data = data[:new_row_num]
    return data


def stride_downsample_sequences(data, seq_length, stride, downsample,
                                offset=0, in_out_ratio=1):
    """Build sequences with an array of data tensor.

    Args:
        data (np.ndarray): data to turn into sequences
        seq_length (int): sequence length of the original sequence (e.g., 30
            frames will be downsampled to 5 frames if downsample is 6.)
        stride (int): step size over data when looping over frames
        downsample (int): amount to downsample data (e.g., 6 will take 240 Hz
            to 40 Hz.)
        offset (int, optional): offset for index when looping; useful for the
            prediction task when making output data. Defaults to 0.
        in_out_ratio (int, optional): ratio of input to output; useful for the
            conversion task when making output data. Defaults to 1.

    Returns:
        np.ndarray: data broken into sequences
    """
    samples = []
    for i in range(0, data.shape[0] - 2*seq_length, stride):
        i_shift = i+offset
        sample = data[i_shift:i_shift+seq_length:downsample, :]

        ratio_shift = sample.shape[0] - sample.shape[0]//in_out_ratio
        sample = sample[ratio_shift:, :]
        samples.append(sample)
    samples = np.concatenate(samples, axis=0)
    return samples


def read_h5(filepaths, requests):
    """Read data from an h5 file and store in a dataset dict.

    Primarily used for building a dataset (see build-dataset.py)

    Args:
        filepaths (list): list of file paths to draw data from.
        requests (dict): dictionary of requests to make to files.

    Returns:
        dict: dictionary containing files mapped to labels mapped to data
    """
    xsensIndices = XSensDataIndices()
    indices = xsensIndices(requests)

    def flatten(l): return [item for sublist in l for item in sublist]

    h5_files = []
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, "r+")
        except OSError:
            logger.info(f"OSError: Unable to open file {filepath}")
            continue
        h5_files.append((h5_file, os.path.basename(filepath)))

    dataset = {}
    for h5_file, filename in h5_files:
        dataset[filename] = {}
        for label in indices:
            label_indices = flatten(indices[label])
            label_indices.sort()

            file_data = np.array(h5_file[label])
            file_data = file_data.reshape(file_data.shape[1],
                                          file_data.shape[0])

            data = np.array(file_data[:, label_indices])

            dataset[filename][label] = data

        h5_file.close()

    return dataset


def read_variables(h5_file_path, task, seq_length, stride, downsample,
                   in_out_ratio=1):
    """Read data from dataset and store in X and y variables.

    Args:
        h5_file_path (str): h5 file containing dataset built previously
        task (str): either prediction or conversion; task that will be modeled
            by the machine learning model
        seq_length (int): original sequence length before downsampling data
        stride (int): step size over data when building sequences
        downsample (int): amount to downsample data by (e.g., 6 to reduce
            240 Hz sampling rate to 40 Hz.)
        in_out_ratio (int, optional): input length compared to output length.
            Defaults to 1.

    Returns:
        tuple: returns a tuple of variables X and y for use in a machine
            learning task.
    """
    X, y = None, None
    h5_file = h5py.File(h5_file_path, "r")
    for filename in h5_file.keys():
        X_temp = h5_file[filename]["X"]
        X_temp = discard_remainder(X_temp, 2*seq_length)

        if task == "prediction":
            y_temp = stride_downsample_sequences(X_temp, seq_length, stride,
                                                 downsample, offset=seq_length)
        elif task == "conversion":
            y_temp = h5_file[filename]["Y"]
            y_temp = discard_remainder(y_temp, 2*seq_length)
            y_temp = stride_downsample_sequences(y_temp, seq_length, stride,
                                                 downsample,
                                                 in_out_ratio=in_out_ratio)
        else:
            logger.error(("Task must be either prediction or conversion, "
                          f"found {task}"))
            sys.exit()

        X_temp = stride_downsample_sequences(X_temp, seq_length,
                                             stride, downsample)

        assert not np.any(np.isnan(X_temp))
        assert not np.any(np.isnan(y_temp))

        if X is None and y is None:
            X = torch.tensor(X_temp)
            y = torch.tensor(y_temp)
        else:
            X = torch.cat((X, torch.tensor(X_temp)), dim=0)
            y = torch.cat((y, torch.tensor(y_temp)), dim=0)
    h5_file.close()
    return X, y


def load_dataloader(args, set_type, normalize, norm_data=None, shuffle=True):
    """Create dataloaders for PyTorch machine learning tasks.

    Args:
        args (argparse.Namespace): contains accessible arguments passed in
            to module
        set_type (str): set to read from when gathering data (either training,
            validation or testing sets)
        normalize (bool): whether to normalize the data before adding to
            dataloader
        norm_data (tuple, optional): if passed will contain mean and std_dev
            data to normalize input data with. Defaults to None.
        shuffle (bool, optional): whether to shuffle the data stored in the
            dataloader. Defaults to True.

    Returns:
        tuple: returns a tuple containing the DataLoader and the normalization
            data
    """
    file_path = args.data_path + "/" + set_type + ".h5"
    seq_length = int(args.seq_length)
    downsample = int(args.downsample)
    batch_size = int(args.batch_size)
    in_out_ratio = int(args.in_out_ratio)
    stride = int(args.stride) if set_type == "training" else seq_length//2

    logger.info((f"Retrieving {set_type} data "
                 f"for sequences {int(seq_length/240*1000)} ms long and "
                 f"downsampling to {240/downsample} Hz..."))

    X, y = read_variables(file_path, args.task, seq_length, stride, downsample,
                          in_out_ratio=in_out_ratio)

    if normalize:
        mean, std_dev = None, None
        if norm_data is None:
            mean, std_dev = X.mean(dim=0), X.std(dim=0)
            norm_data = (mean, std_dev)
            with h5py.File(args.data_path + "/normalization.h5", "w") as f:
                f["mean"], f["std_dev"] = mean, std_dev
        else:
            mean, std_dev = norm_data
        X = X.sub(mean).div(std_dev + 1e-8)

    logger.info(f"Data for {set_type} have shapes "
                f"(X, y): {X.shape}, {y.shape}")

    X = X.view(-1, math.ceil(seq_length/downsample), X.shape[1])
    y = y.view(-1, math.ceil(seq_length/(downsample*in_out_ratio)), y.shape[1])

    logger.info(f"Reshaped {set_type} shapes (X, y): {X.shape}, {y.shape}")

    dataset = TensorDataset(X, y)

    shuffle = True if set_type == "training" else False
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=True)

    logger.info(f"Number of {set_type} samples: {len(dataset)}")

    return dataloader, norm_data
