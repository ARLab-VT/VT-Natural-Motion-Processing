# Copyright (c) 2020-present, Assistive Robotics Lab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import h5py
from .logging import logger
from .rotations import quat_to_rotMat, rotMat_to_quat
from .quaternion import quat_fix


def add_normalized_positions(filepaths, new_group_name):
    """Add position data normalized relative to the pelvis to h5 files.

    Args:
        filepaths (list): paths to files to add data to
        new_group_name (str): what the new group will be called in the h5
            file
    """
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, "r+")
        except OSError:
            logger.info(f"OSError: Unable to open file {filepath}")
            continue

        quat = np.array(h5_file["orientation"][:, :])
        quat = quat.reshape(quat.shape[1], quat.shape[0])
        quat = quat.reshape(quat.shape[0], -1, 4)

        pos = np.array(h5_file["position"][:, :])
        pos = pos.reshape(pos.shape[1], pos.shape[0])
        pos = pos.reshape(pos.shape[0], -1, 3)

        quat = quat_fix(quat)

        norm_pos = np.zeros(pos.shape)

        pelvis_rot = np.linalg.inv(
                        quat_to_rotMat(torch.tensor(quat[:, 0, :]))
                        )
        pelvis_pos = pos[:, 0, :]
        for i in range(0, quat.shape[1]):
            relative_pos = np.expand_dims(pos[:, i, :] - pelvis_pos, axis=2)
            norm_pos[:, i, :] = np.squeeze(np.matmul(pelvis_rot, relative_pos),
                                           axis=2)

        norm_pos = norm_pos.reshape(norm_pos.shape[0], -1)
        norm_pos = norm_pos.reshape(norm_pos.shape[1], norm_pos.shape[0])

        try:
            logger.info(f"Writing to file {filepath}")
            h5_file.create_dataset(new_group_name, data=norm_pos)
        except RuntimeError:
            logger.info(("RuntimeError: Unable to create link "
                         f"(name already exists) in {filepath}"))
        h5_file.close()


def add_normalized_accelerations(filepaths, group_name, new_group_name,
                                 root=0):
    """Add acceleration data normalized relative to a root to the h5 files.

    Args:
        filepaths (list): paths to files to add data to
        group_name (str): acceleration group to normalize
            (typically acceleration, but can also be sensorFreeAcceleration)
        new_group_name (str): new group name for normalized acceleration data
        root (int, optional): index of root (e.g., 0 is pelvis, 4 is sternum).
            Defaults to 0.
    """
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, "r+")
        except OSError:
            logger.info(f"OSError: Unable to open file {filepath}")
            continue

        quat = np.array(h5_file["orientation"][:, :])
        quat = quat.reshape(quat.shape[1], quat.shape[0])
        quat = quat.reshape(quat.shape[0], -1, 4)

        acc = np.array(h5_file[group_name][:, :])
        acc = acc.reshape(acc.shape[1], acc.shape[0])
        acc = acc.reshape(acc.shape[0], -1, 3)

        quat = quat_fix(quat)

        norm_acc = np.zeros(acc.shape)

        root_rot = np.linalg.inv(
                    quat_to_rotMat(torch.tensor(quat[:, root, :]))
                    )
        root_acc = acc[:, root, :]
        for i in range(0, acc.shape[1]):
            relative_acc = np.expand_dims(acc[:, i, :] - root_acc, axis=2)
            norm_acc[:, i, :] = np.squeeze(np.matmul(root_rot, relative_acc),
                                           axis=2)

        norm_acc = norm_acc.reshape(norm_acc.shape[0], -1)
        norm_acc = norm_acc.reshape(norm_acc.shape[1], norm_acc.shape[0])

        try:
            logger.info(f"Writing to file {filepath}")
            h5_file.create_dataset(new_group_name, data=norm_acc)
        except RuntimeError:
            logger.info(("RuntimeError: Unable to create link "
                         f"(name already exists) in {filepath}"))
        h5_file.close()


def add_normalized_quaternions(filepaths, group_name, new_group_name, root=0):
    """Add orientation data normalized relative to a root to the h5 files.

    Args:
        filepaths (list): paths to files to add data to
        group_name (str): orientation group to normalize
            (typically orientation, but can also be sensorOrientation)
        new_group_name (str): new group name for normalized orientation data
        root (int, optional): index of root (e.g., 0 is pelvis, 4 is sternum).
            Defaults to 0.
    """
    for filepath in filepaths:
        try:
            h5_file = h5py.File(filepath, "r+")
        except OSError:
            logger.info(f"OSError: Unable to open file {filepath}")
            continue

        quat = np.array(h5_file[group_name][:, :])
        quat = quat.reshape(quat.shape[1], quat.shape[0])
        quat = quat.reshape(quat.shape[0], -1, 4)

        quat = quat_fix(quat)

        norm_quat = np.zeros(quat.shape)

        root_rotMat = np.linalg.inv(
                        quat_to_rotMat(torch.tensor(quat[:, root, :]))
                        )
        for i in range(0, quat.shape[1]):
            rotMat = quat_to_rotMat(torch.tensor(quat[:, i, :]))
            norm_rotMat = np.matmul(root_rotMat, rotMat)
            norm_quat[:, i, :] = rotMat_to_quat(norm_rotMat)

        norm_quat = norm_quat.reshape(norm_quat.shape[0], -1)
        norm_quat = norm_quat.reshape(norm_quat.shape[1], norm_quat.shape[0])

        try:
            logger.info(f"Writing to file {filepath}")
            h5_file.create_dataset(new_group_name, data=norm_quat)
        except RuntimeError:
            logger.info(("RuntimeError: Unable to create link "
                         f"(name already exists) in {filepath}"))
        h5_file.close()
