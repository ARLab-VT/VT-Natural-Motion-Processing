# Copyright (c) 2020-present, Assistive Robotics Lab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import math


class QuatDistance(nn.Module):
    """Loss function for calculating cosine similarity of quaternions."""

    def __init__(self):
        """Initialize QuatDistance loss."""
        super(QuatDistance, self).__init__()

    def forward(self, predictions, targets):
        """Forward pass through the QuatDistance loss.

        Args:
            predictions (torch.Tensor): the predictions from the model in
                quaternion form.
            targets (torch.Tensor): the targets in quaternion form.

        Returns:
            torch.Tensor: average angular difference in degrees between
            quaternions in predictions and targets.
        """
        predictions = predictions.contiguous().view(-1, 1, 4)
        targets = targets.contiguous().view(-1, 4, 1)

        inner_prod = torch.bmm(predictions, targets).view(-1)

        x = torch.clamp(torch.abs(inner_prod), min=0.0, max=1.0-1e-7)

        theta = torch.acos(x)

        return (360.0/math.pi)*torch.mean(theta)
