import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .conversions import *
from .logging import logger

class ExpmapToQuatLoss(nn.Module):
    def __init__(self):
        super(ExpmapToQuatLoss, self).__init__()

    def forward(self, predictions, targets):
        predictions = predictions.view(-1,3)
        targets = targets.contiguous().view(-1,3)

        predictions = rotMat_to_quat(expmap_to_rotMat(predictions.double()))
        targets = rotMat_to_quat(expmap_to_rotMat(targets.double()))

        return F.l1_loss(predictions, targets)

class QuatDistance(nn.Module):
    def __init__(self):
        super(QuatDistance, self).__init__()

    def forward(self, predictions, targets):
        predictions = predictions.contiguous().view(-1,1,4)
        targets = targets.contiguous().view(-1,4,1)

        inner_prod = torch.bmm(predictions, targets).view(-1)

        x = torch.clamp(torch.abs(inner_prod), min=0.0, max=1.0-1e-7)

        theta = torch.acos(x)

        return (360.0/math.pi)*torch.mean(theta)
