# Copyright (c) 2020-present, Assistive Robotics Lab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def quat_to_rotMat(q):
    """Convert quaternions to rotation matrices.

    Using equation provided in XSens MVN Manual:
    https://www.xsens.com/hubfs/Downloads/usermanual/MVN_User_Manual.pdf

    Args:
        q (torch.Tensor): quaternion(s) to convert to rotation matrix format

    Returns:
        torch.Tensor: rotation matrix converted from quaternion format
    """
    if len(q.shape) != 2:
        q = q.unsqueeze(0)

    assert q.shape[1] == 4

    r0c0 = q[:, 0]**2 + q[:, 1]**2 - q[:, 2]**2 - q[:, 3]**2
    r0c1 = 2*q[:, 1]*q[:, 2] - 2*q[:, 0]*q[:, 3]
    r0c2 = 2*q[:, 1]*q[:, 3] + 2*q[:, 0]*q[:, 2]

    r1c0 = 2*q[:, 1]*q[:, 2] + 2*q[:, 0]*q[:, 3]
    r1c1 = q[:, 0]**2 - q[:, 1]**2 + q[:, 2]**2 - q[:, 3]**2
    r1c2 = 2*q[:, 2]*q[:, 3] - 2*q[:, 0]*q[:, 1]

    r2c0 = 2*q[:, 1]*q[:, 3] - 2*q[:, 0]*q[:, 2]
    r2c1 = 2*q[:, 2]*q[:, 3] + 2*q[:, 0]*q[:, 1]
    r2c2 = q[:, 0]**2 - q[:, 1]**2 - q[:, 2]**2 + q[:, 3]**2

    r0 = torch.stack([r0c0, r0c1, r0c2], dim=1)
    r1 = torch.stack([r1c0, r1c1, r1c2], dim=1)
    r2 = torch.stack([r2c0, r2c1, r2c2], dim=1)

    R = torch.stack([r0, r1, r2], dim=2)

    return R.permute(0, 2, 1)


def rotMat_to_quat(rotMat):
    """Convert rotation matrices back to quaternions.

    Ported from Matlab:
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4


    Args:
        rotMat (torch.Tensor): rotation matrix or matrices to convert to
            quaternion format.

    Returns:
        torch.Tensor: quaternion(s) converted from rotation matrix format
    """
    if len(rotMat.shape) != 3:
        rotMat = rotMat.unsqueeze(0)

    assert rotMat.shape[1] == 3 and rotMat.shape[2] == 3

    diffMat = rotMat - torch.transpose(rotMat, 1, 2)

    r = torch.zeros((rotMat.shape[0], 3), dtype=torch.float64)

    r[:, 0] = -diffMat[:, 1, 2]
    r[:, 1] = diffMat[:, 0, 2]
    r[:, 2] = -diffMat[:, 0, 1]

    sin_theta = torch.norm(r, dim=1)/2
    sin_theta = sin_theta.unsqueeze(1)

    r0 = r / (torch.norm(r, dim=1).unsqueeze(1) + 1e-9)

    cos_theta = (rotMat.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_theta = cos_theta.unsqueeze(1)

    theta = torch.atan2(sin_theta, cos_theta)

    theta = theta.squeeze(1)

    q = torch.zeros((rotMat.shape[0], 4), dtype=torch.float64)

    q[:, 0] = torch.cos(theta/2)
    q[:, 1:] = r0*torch.sin(theta/2).unsqueeze(1)

    return q
