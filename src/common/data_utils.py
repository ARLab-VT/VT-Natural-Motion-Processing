import numpy as np
import pandas as pd
import warnings
import h5py
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import os
from .logging import *
import sys
from .conversions import *
from torch.utils.data import TensorDataset, DataLoader

class XSensDataIndices:
    def __init__(self):
        sensor_group = ['sensorFreeAcceleration',
                        'sensorMagneticField',
                        'sensorOrientation',
                        'normSensorOrientation',
                        'normSensorAcceleration']

        segment_group = ['position', 'normPositions', 'velocity', 'acceleration', 'normAcceleration',
                         'sternumNormAcceleration', 'angularVelocity', 'angularAcceleration',
                         'orientation', 'normOrientation', 'normExpmapOrientation', 'sternumNormOrientation']

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
        dims = 4 if req_label in ['orientation', 'normOrientation', 'sternumNormOrientation', 'normSensorOrientation'] else 3

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


def add_expmap_orientations(filepaths, group_name, new_group_name):
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, 'r+')
        except OSError:
            print("OSError: Unable to open file {}".format(filepath))
            continue

        quat = torch.Tensor(h5_file[group_name][:, :]) 
        quat = quat.view(quat.shape[1], quat.shape[0])

        old_shape = quat.shape
        quat = quat.reshape(-1, 4)

        expmap = quat_to_expmap(quat)
        expmap = expmap.view(old_shape[0], old_shape[1]//4*3)
        expmap = expmap.view(expmap.shape[1], expmap.shape[0])

        try:    
            print("Writing to file {}".format(filepath))
            h5_file.create_dataset(new_group_name, data=expmap)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()


def add_normalized_positions(filepaths, new_group_name):
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, 'r+')
        except OSError:
            print("OSError: Unable to open file {}".format(filepath))
            continue

        quat = np.array(h5_file['orientation'][:, :]) 
        quat = quat.reshape(quat.shape[1], quat.shape[0])
        quat = quat.reshape(quat.shape[0], -1, 4)

        pos = np.array(h5_file['position'][:, :])
        pos = pos.reshape(pos.shape[1], pos.shape[0])
        pos = pos.reshape(pos.shape[0], -1, 3)

        quat = qfix(quat)

        norm_pos = np.zeros(pos.shape)

        pelvis_rot = np.linalg.inv(quat_to_rotMat(torch.tensor(quat[:, 0, :])))
        pelvis_pos = pos[:, 0, :]
        for i in range(0, quat.shape[1]):
            relative_pos = np.expand_dims(pos[:, i, :] - pelvis_pos, axis=2)
            norm_pos[:, i, :] = np.squeeze(np.matmul(pelvis_rot, relative_pos), axis=2)

        norm_pos = norm_pos.reshape(norm_pos.shape[0], -1) 
        norm_pos = norm_pos.reshape(norm_pos.shape[1], norm_pos.shape[0])

        try:            
            print("Writing to file {}".format(filepath))
            h5_file.create_dataset(new_group_name, data=norm_pos)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()

def add_normalized_accelerations(filepaths, group_name, new_group_name, root=0):
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, 'r+')
        except OSError:
            print("OSError: Unable to open file {}".format(filepath))
            continue

        quat = np.array(h5_file['orientation'][:, :]) 
        quat = quat.reshape(quat.shape[1], quat.shape[0])
        quat = quat.reshape(quat.shape[0], -1, 4)

        acc = np.array(h5_file[group_name][:, :])
        acc = acc.reshape(acc.shape[1], acc.shape[0])
        acc = acc.reshape(acc.shape[0], -1, 3)

        quat = qfix(quat)

        norm_acc = np.zeros(acc.shape)

        root_rot = np.linalg.inv(quat_to_rotMat(torch.tensor(quat[:, root, :])))
        root_acc = acc[:, root, :]
        for i in range(0, acc.shape[1]):
            relative_acc = np.expand_dims(acc[:, i, :] - root_acc, axis=2)
            norm_acc[:, i, :] = np.squeeze(np.matmul(root_rot, relative_acc), axis=2)

        norm_acc = norm_acc.reshape(norm_acc.shape[0], -1) 
        norm_acc = norm_acc.reshape(norm_acc.shape[1], norm_acc.shape[0])

        try:            
            print("Writing to file {}".format(filepath))
            h5_file.create_dataset(new_group_name, data=norm_acc)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()


def add_normalized_quaternions(filepaths, group_name, new_group_name, root=0):
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, 'r+')
        except OSError:
            print("OSError: Unable to open file {}".format(filepath))
            continue

        quat = np.array(h5_file[group_name][:, :]) 
        quat = quat.reshape(quat.shape[1], quat.shape[0])
        quat = quat.reshape(quat.shape[0], -1, 4)

        quat = qfix(quat)

        norm_quat = np.zeros(quat.shape)

        root_rotMat = np.linalg.inv(quat_to_rotMat(torch.tensor(quat[:, root, :])))
        for i in range(0, quat.shape[1]):
            rotMat = quat_to_rotMat(torch.tensor(quat[:, i, :])) 
            norm_rotMat = np.matmul(root_rotMat, rotMat)
            norm_quat[:, i, :] = rotMat_to_quat(norm_rotMat)

        norm_quat = norm_quat.reshape(norm_quat.shape[0], -1) 
        norm_quat = norm_quat.reshape(norm_quat.shape[1], norm_quat.shape[0])

        try:            
            print("Writing to file {}".format(filepath))
            h5_file.create_dataset(new_group_name, data=norm_quat)
        except RuntimeError:
            print("RuntimeError: Unable to create link (name already exists) in {}".format(
                  filepath))
        h5_file.close()


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


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


def stride_downsample_sequences(data, seq_length, stride, downsample, offset=0, in_out_ratio=1):
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


def read_variables(h5_file_path, task, seq_length, stride, downsample, in_out_ratio=1):
    X, y = None, None
    h5_file = h5py.File(h5_file_path, 'r')
    for filename in h5_file.keys():
        X_temp = h5_file[filename]['X']
        X_temp = discard_remainder(X_temp, 2*seq_length)

        if task == 'prediction':
            y_temp = stride_downsample_sequences(X_temp, seq_length, stride, downsample, offset=seq_length)
        elif task == 'conversion':
            y_temp = h5_file[filename]['Y']
            y_temp = discard_remainder(y_temp, 2*seq_length)
            y_temp = stride_downsample_sequences(y_temp, seq_length, stride, downsample, in_out_ratio=in_out_ratio)
        else:
            logger.error("Task must be either prediction or conversion, found {}".format(task))
            sys.exit()

        X_temp = stride_downsample_sequences(X_temp, seq_length, stride, downsample)
        
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


def load_dataloader(args, type, normalize, norm_data=None, shuffle=True): 
    file_path = args.data_path + '/' + type + '.h5'
    seq_length = int(args.seq_length)
    downsample = int(args.downsample)
    batch_size = int(args.batch_size)
    in_out_ratio = int(args.in_out_ratio)
    stride = int(args.stride) if type == 'training' else seq_length//2    

    logger.info("Retrieving {} data for sequences {} ms long and downsampling to {} Hz...".format(type, int(seq_length/240*1000), 240/downsample))

    X, y = read_variables(file_path, args.task, seq_length, stride, downsample, in_out_ratio=in_out_ratio)

    if normalize:
        mean, std_dev = None, None
        if norm_data is None:
            mean, std_dev = X.mean(dim=0), X.std(dim=0)
            norm_data = (mean, std_dev)
            with h5py.File(args.data_path + '/normalization.h5','w') as f:
                f['mean'], f['std_dev'] = mean, std_dev
        else:
            mean, std_dev = norm_data
        X = X.sub(mean).div(std_dev + 1e-8)

    logger.info("Data for {} have shapes (X, y): {}, {}".format(type, X.shape, y.shape))

    X = X.view(-1, math.ceil(seq_length/downsample), X.shape[1])
    y = y.view(-1, math.ceil(seq_length/(downsample*in_out_ratio)), y.shape[1])

    logger.info("Reshaped {} shapes (X, y): {}, {}".format(type, X.shape, y.shape))
    
    dataset = TensorDataset(X, y)
    
    shuffle = True if type == 'training' else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    logger.info("Number of {} samples: {}".format(type, len(dataset)))

    return dataloader, norm_data
