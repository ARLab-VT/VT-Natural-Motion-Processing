import numpy as np
import pandas as pd
import warnings
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import os
from .logging import *
import sys
from .conversions import *

class XSensDataIndices:
    def __init__(self):
        sensor_group = ['sensorFreeAcceleration',
                        'sensorMagneticField',
                        'sensorOrientation']

        segment_group = ['position', 'velocity', 'acceleration',
                         'angularVelocity', 'angularAcceleration',
                         'orientation', 'smoothedOrientation', 'normOrientation', 'relativePosition']

        joint_group = ['jointAngle', 'jointAngleExpmap', 'jointAngleXZY']

        joint_ergo_group = ['jointAngleErgo', 'jointAngleErgoXZY']

        groups = [sensor_group, segment_group,
                  joint_group, joint_ergo_group]
        self._labels_to_items(groups)

    def __call__(self, requests):
        label_indices = {}
        for label, items in requests.items():
            if label in self.label_items:
                label_indices[label] = self._request(label, items)
        return label_indices

    def _labels_to_items(self, groups):
        self.label_items = {}
        sensors = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm',
                   'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftUpperArm',
                   'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg',
                   'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']

        segments = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
                    'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
                    'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
                    'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe',
                    'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToe']

        joints = ['jL5S1', 'jL4L3', 'jL1T12', 'jT9T8', 'jT1C7', 'jC1Head',
                  'jRightT4Shoulder', 'jRightShoulder', 'jRightElbow', 'jRightWrist',
                  'jLeftT4Shoulder', 'jLeftShoulder', 'jLeftElbow', 'jLeftWrist',
                  'jRightHip', 'jRightKnee', 'jRightAnkle', 'jRightBallFoot',
                  'jLeftHip', 'jLeftKnee', 'jLeftAnkle', 'jLeftBallFoot']

        ergo_joints = ['T8_Head', 'T8_LeftUpperArm', 'T8_RightUpperArm',
                       'Pelvis_T8', 'Vertical_Pelvis', 'Vertical_T8']

        item_groups = [sensors, segments, joints, ergo_joints]

        for index, group in enumerate(groups):
            for label in group:
                self.label_items[label] = item_groups[index]

    def _request(self, req_label, req_items):
        valid_items = self.label_items[req_label]

        if 'all' in req_items:
            req_items = valid_items

        num_valid_items = len(valid_items)
        dims = 4 if req_label in ['orientation', 'smoothedOrientation', 'normOrientation'] else 3

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


def add_relative_position(filepaths):
    for filepath in filepaths:
        h5_file = h5py.File(filepath, 'r+')

        positions = np.array(h5_file['position'][:, :])
        positions = positions.reshape(positions.shape[1], positions.shape[0])
        pelvis = positions[:, :3]

        relative_positions = positions - np.tile(pelvis, np.array(positions.shape) //
                                                 np.array(pelvis.shape))
        
        relative_positions = relative_positions.reshape(relative_positions.shape[1], relative_positions.shape[0])

        try: 
            print("Writing to file {}".format(filepath))
            h5_file.create_dataset('relativePosition', data=relative_positions)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()


def add_joint_angle_expmap(filepaths):
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, 'r+')
        except OSError:
            print("OSError: Unable to open file {}".format(filepath))
            continue
 
        joint_angles = np.array(h5_file['jointAngle'])
        joint_angles = joint_angles.reshape(joint_angles.shape[1], joint_angles.shape[0])
        
        quat = euler_to_quaternion(joint_angles, 'zxy')
        expmap = quaternion_to_expmap(quat)
        expmap = expmap.reshape(expmap.shape[1], expmap.shape[0])

        try:
            print("Writing to file {}".format(filepath))
            h5_file.create_dataset('jointAngleExpmap', data=expmap)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()

def add_continuous_quaternions(filepaths, group_name, new_group_name):
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, 'r+')
        except OSError:
            print("OSError: Unable to open file {}".format(filepath))
            continue
 
        quat = np.array(h5_file[group_name][:, :])
        quat = quat.reshape(quat.shape[1], quat.shape[0])
        quat = quat.reshape(quat.shape[0], -1, 4)

        cont_quat = qfix(quat)

        cont_quat = cont_quat.reshape(cont_quat.shape[0], -1)
        cont_quat = cont_quat.reshape(cont_quat.shape[1], cont_quat.shape[0])
        
        try:            
            print("Writing to file {}".format(filepath))
            h5_file.create_dataset(new_group_name, data=cont_quat)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()

def add_normalized_quaternions(filepaths, group_name, new_group_name):
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, 'r+')
        except OSError:
            print("OSError: Unable to open file {}".format(filepath))
            continue

        quat = np.array(h5_file[group_name][:, :]) 
        quat = quat.reshape(quat.shape[1], quat.shape[0])
        quat = quat.reshape(quat.shape[0], -1, 4)

        norm_quat = np.zeros(quat.shape)

        pelvis = np.linalg.inv(quat_to_rotMat(torch.tensor(quat[:, 0, :])))
        for i in range(0, quat.shape[1]):
            rotMat = quat_to_rotMat(torch.tensor(quat[:, i, :])) 
            norm_rotMat = np.matmul(pelvis, rotMat)
            norm_quat[:, i, :] = rotMat_to_quat(norm_rotMat)

        norm_quat = norm_quat.reshape(norm_quat.shape[0], -1) 
        norm_quat = norm_quat.reshape(norm_quat.shape[1], norm_quat.shape[0])

        try:            
            print("Writing to file {}".format(filepath))
            nquat = h5_file[new_group_name]
            nquat[...] = norm_quat
            #h5_file.create_dataset(new_group_name, data=norm_quat)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()
        

def qfix(q):
    """
    Borrowed from QuaterNet: 
    https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py#L119    

    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result


def discard_remainder(data, seq_length):
    new_row_num = data.shape[0] - (data.shape[0] % seq_length)
    data = data[:new_row_num]
    return data


def reshape_to_sequences(data, seq_length):
    data = discard_remainder(data, seq_length)
    data = data.reshape(-1, seq_length, data.shape[1])
    return data


def pad_sequences(sequences, maxlen, start_char=False, padding='post'):
    if start_char:
        sequences = np.append(
            np.ones((1, sequences.shape[1])), sequences, axis=0)

    padded_sequences = np.zeros((maxlen, sequences.shape[1]))

    if padding == 'post':
        padded_sequences[:sequences.shape[0], :] = sequences
    elif padding == 'pre':
        offset = maxlen - sequences.shape[0]
        padded_sequences[offset:, :] = sequences
    return padded_sequences


def split_sequences(data, seq_length, stride):
    X, y = [], []
    for i in range(0, data.shape[0] - 2*seq_length, stride):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length:i+2*seq_length, :])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def read_h5(filepaths, requests):
    xsensIndices = XSensDataIndices()
    indices = xsensIndices(requests)

    def flatten(l): return [item for sublist in l for item in sublist]

    h5_files = []
    for filepath in filepaths:
        h5_file = h5py.File(filepath, 'r')
        h5_files.append((h5_file, os.path.basename(filepath)))

    dataset = {}
    for h5_file, filename in h5_files:
        dataset[filename] = {}
        for label in indices:
            label_indices = flatten(indices[label])
            label_indices.sort()
            
            file_data = np.array(h5_file[label])
            file_data = file_data.reshape(file_data.shape[1], file_data.shape[0])
            
            data = np.array(file_data[:, label_indices])

            dataset[filename][label] = data

        h5_file.close()

    return dataset


def read_variables(h5_file_path, task, seq_length, stride):
    X, y = None, None
    h5_file = h5py.File(h5_file_path, 'r')
    for filename in h5_file.keys():
        X_temp = h5_file[filename]['X']
        X_temp = discard_remainder(X_temp, 2*seq_length)

        if task == 'prediction':
            X_temp, y_temp = split_sequences(X_temp, seq_length, stride)
        elif task == 'conversion':
            y_temp = h5_file[filename]['Y']
            y_temp = discard_remainder(y_temp, 2*seq_length)
        else:
            logger.error("Task must be either prediction or conversion, found {}".format(task))
            sys.exit()
        
        if X is None and y is None:
            X = X_temp
            y = y_temp
        else:
            X = np.append(X, X_temp, axis=0)
            y = np.append(y, y_temp, axis=0)
    h5_file.close()
    return X, y
