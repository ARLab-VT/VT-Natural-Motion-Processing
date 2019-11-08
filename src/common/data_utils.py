import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset, DataLoader, random_split


class XSensDataIndices:
    def __init__(self):
        sensor_group = ['sensorFreeAcceleration',
                        'sensorMagneticField',
                        'sensorOrientation']

        segment_group = ['position', 'velocity', 'acceleration',
                         'angularVelocity', 'angularAcceleration',
                         'orientation']

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

        num_valid_items = len(valid_items)
        dims = 4 if req_label == 'orientation' else 3

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


def reshape_to_sequences(data, seq_length=120):
    data = discard_remainder(data, seq_length)
    data = data.reshape(data.shape[0] // seq_length, seq_length, data.shape[1])
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


def split_sequences(data, seq_length=120):
    split_indices = seq_length // 2

    split_data = [(data[i, :split_indices, :], data[i, split_indices:, :])
                  for i in range(data.shape[0])]

    encoder_input_data = np.array([pad_sequences(
        sequences[0], seq_length // 2, padding='pre') for sequences in split_data])

    decoder_target_data = np.array(
        [pad_sequences(sequences[1], seq_length // 2) for sequences in split_data])

    return encoder_input_data, decoder_target_data


def read_h5(filenames, data_path, requests, seq_length, request_type=None):
    data = None

    for idx, file in enumerate(filenames):
        target


def read_data(filenames, data_path, requests, seq_length, request_type=None):
    data = None

    for idx, file in enumerate(filenames):
        target_columns = request_indices(requests)
        print("Index: " + str(idx + 1), end='\r')
        data_temp = pd.read_csv(data_path / file,
                                delimiter=',',
                                usecols=target_columns,
                                header=0).values

        data_temp = discard_remainder(data_temp, seq_length)

        if request_type == 'Position':
            data_temp[:, 3:] -= data_temp[:, :3]
            data_temp = data_temp[:, 3:]

        if idx == 0:
            data = data_temp
        else:
            data = np.concatenate((data, data_temp), axis=0)

    print("Done with reading files")
    print("Number of frames in dataset:", data.shape[0])
    print("Number of bytes:", data.nbytes)

    data = reshape_to_sequences(data, seq_length=seq_length)
    return data


def setup_datasets(encoder_input_data, decoder_target_data, batch_size, split_size):
    dataset = TensorDataset(encoder_input_data, decoder_target_data)

    train_size_approx = int(split_size * len(dataset))
    train_size = train_size_approx - (train_size_approx % batch_size)
    val_size = len(dataset) - train_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    return dataset, train_subset.indices, val_subset.indices


def setup_dataloaders(datasets, batch_size):
    train_dataloader = DataLoader(
        datasets[0], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        datasets[1], batch_size=batch_size, shuffle=False)

    dataloaders = (train_dataloader, val_dataloader)

    return dataloaders
