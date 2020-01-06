import numpy as np
import pandas as pd
import warnings
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import os


class XSensDataIndices:
    def __init__(self):
        sensor_group = ['sensorFreeAcceleration',
                        'sensorMagneticField',
                        'sensorOrientation']

        segment_group = ['position', 'velocity', 'acceleration',
                         'angularVelocity', 'angularAcceleration',
                         'orientation', 'smoothedOrientation', 'relativePosition']

        joint_group = ['jointAngle', 'jointAngleXZY']

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
        dims = 4 if req_label in ['orientation', 'smoothedOrientation'] else 3

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
            h5_file.create_dataset('relativePosition', data=relative_positions)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()

def add_smooth_orientations(filepaths):
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, 'r+')
        except OSError:
            print("OSError: Unable to open file {}".format(filepath))
            continue
 
        orient = np.array(h5_file['orientation'][:, :])
        orient = orient.reshape(orient.shape[1], orient.shape[0])
        orient = orient.reshape(orient.shape[0], -1, 4)

        smooth_orient = qfix(orient)

        smooth_orient = smooth_orient.reshape(smooth_orient.shape[0], -1)
        smooth_orient = smooth_orient.reshape(smooth_orient.shape[1], smooth_orient.shape[0])
        
        try:
            h5_file.create_dataset('smoothedOrientation', data=smooth_orient)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()

def qfix(q):
    """
    Attribution: https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py#L119    

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

def get_body_info_map():

    num_segments = 23
    num_joints = 22
    num_sensors = 17

    orientation_indices = [list(range(i, i + 4)) for i in range(0, 89, 4)]
    position_indices = [list(range(i, i + 3)) for i in range(92, 159, 3)]
    joint_angle_indices = [list(range(i, i + 3)) for i in range(161, 225, 3)]

    segment_keys = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
                    'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
                    'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
                    'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe',
                    'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToe']
    joint_keys = ['jL5S1', 'jL4L3', 'jL1T12', 'jT9T8', 'jT1C7', 'jC1Head',
                  'jRightT4Shoulder', 'jRightShoulder', 'jRightElbow', 'jRightWrist',
                  'jLeftT4Shoulder', 'jLeftShoulder', 'jLeftElbow', 'jLeftWrist',
                  'jRightHip', 'jRightKnee', 'jRightAnkle', 'jRightBallFoot',
                  'jLeftHip', 'jLeftKnee', 'jLeftAnkle', 'jLeftBallFoot']

    orientation_map = dict(zip(segment_keys, orientation_indices))
    position_map = dict(zip(segment_keys, position_indices))
    joint_angle_map = dict(zip(joint_keys, joint_angle_indices))
    body_info_map = dict(zip(['Orientation', 'Position', 'Joint'],
                             [orientation_map, position_map, joint_angle_map]))

    return body_info_map


def request_indices(request_dict):
    indices = []
    body_info_map = get_body_info_map()
    for key in list(request_dict.keys()):
        if key in body_info_map:
            relevant_map = body_info_map[key]
            for request in request_dict[key]:
                indices += relevant_map[request]

    indices.sort()
    return indices


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
            data = np.array(h5_file[label][label_indices, :])

            data = data.reshape(data.shape[1], data.shape[0])

            dataset[filename][label] = data

        h5_file.close()

    return dataset
