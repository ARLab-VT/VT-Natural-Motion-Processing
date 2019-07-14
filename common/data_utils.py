import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def get_body_info_map():

    orientation_indices = [list(range(i, i+4)) for i in range(0, 89, 4)]
    position_indices = [list(range(i, i+3)) for i in range(92, 159, 3)]
    joint_angle_indices = [list(range(i, i+3)) for i in range(161, 225, 3)]

    segment_keys = ['Pelvis','L5','L3','T12','T8','Neck','Head',
                    'RightShoulder','RightUpperArm','RightForeArm','RightHand',
                    'LeftShoulder','LeftUpperArm','LeftForeArm','LeftHand',
                    'RightUpperLeg','RightLowerLeg','RightFoot','RightToe',
                    'LeftUpperLeg','LeftLowerLeg','LeftFoot','LeftToe']
    joint_keys= ['jL5S1','jL4L3','jL1T12','jT9T8','jT1C7','jC1Head',
                 'jRightC7Shoulder','jRightShoulder','jRightElbow','jRightWrist',
                 'jLeftC7Shoulder','jLeftShoulder','jLeftElbow','jLeftWrist',
                 'jRightHip','jRightKnee','jRightAnkle','jRightBallFoot',
                 'jLeftHip','jLeftKnee','jLeftAnkle','jLeftBallFoot']

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
    data = data.reshape(data.shape[0]//seq_length, seq_length, data.shape[1])
    return data

def pad_sequences(sequences, maxlen, start_char=False, padding='post'):
    if start_char:
        sequences = np.append(np.ones((1, sequences.shape[1])), sequences, axis=0)

    padded_sequences = np.zeros((maxlen, sequences.shape[1]))

    if padding == 'post':
        padded_sequences[:sequences.shape[0], :] = sequences
    elif padding == 'pre':
        offset = maxlen - sequences.shape[0]
        padded_sequences[offset:, :] = sequences
    return padded_sequences

def split_sequences(data, seq_length=120):
    split_indices = seq_length//2

    split_data = [(data[i, :split_indices, :], data[i, split_indices:, :]) for i in range(data.shape[0])]

    encoder_input_data = np.array([pad_sequences(sequences[0], seq_length//2, padding='pre') for sequences in split_data])

    decoder_target_data = np.array([pad_sequences(sequences[1], seq_length//2) for sequences in split_data])

    return encoder_input_data, decoder_target_data


def read_data(filenames, data_path, n, requests, seq_length, request_type=None):
    data = None
    
    for idx, file in enumerate(filenames[:n]):
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

    scaler = RobustScaler().fit(data)
    data = scaler.transform(data)

    print("Number of frames in dataset:", data.shape[0])
    print("Number of bytes:", data.nbytes)

    data = reshape_to_sequences(data, seq_length=seq_length) 
    return data, scaler