# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the Attribution-NonCommercial
# 4.0 International license and is borrowed from the QuaterNet library.
# See https://github.com/facebookresearch/QuaterNet/blob/master/LICENSE for
# more details.
#

import torch
import numpy as np


def quat_fix(q):
    """Enforce quaternion continuity across the time dimension.

    Borrowed from QuaterNet:
    https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L119

    This function falls under the Attribution-NonCommercial 4.0 International
    license.

    Selects the representation (q or -q) with minimal distance
    (or, equivalently, maximal dot product) between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and
    J is the number of joints.
    Returns a tensor of the same shape.

    Args:
        q (np.ndarray): quaternions of size (L, J, 4) to enforce continuity.

    Returns:
        np.ndarray: quaternion of size (L, J, 4) that is continuous
            in time dimension.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def quat_mul(q, r):
    """Multiply quaternion(s) q with quaternion(s) r.

    Borrowed from QuaterNet:
    https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L13

    This function falls under the Attribution-NonCommercial 4.0 International
    license.

    Expects two equally-sized tensors of shape (*, 4), where * denotes any
    number of dimensions.
    Returns q*r as a tensor of shape (*, 4).

    Args:
        q (torch.Tensor): quaternions of size (*, 4)
        r (torch.Tensor): quaternions of size (*, 4)

    Returns:
        torch.Tensor: quaternions of size (*, 4)
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
