import torch
import numpy as np
import pandas as pd
import warnings
import h5py
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from scipy.linalg import norm
from .logging import logger

def euler_to_quaternion(joint_angles, order):
    """
    Euler angle conversion to quaternions.
    XSens stores joint angles in ZXY format typically unless using jointAngleXZY format.
    Wrapping scipy library for our purposes.
    """
    assert joint_angles.shape[1] % 3 == 0

    q = np.zeros((joint_angles.shape[0], 4 * (joint_angles.shape[1] // 3)))
       
    for i in range(joint_angles.shape[1] // 3):
        r = R.from_euler(order, joint_angles[:, 3*i:3*(i+1)], degrees=True)
        q[:, 4*i:4*(i+1)] = r.as_quat()

    return q

def quat_to_rotMat(q):
    """
    Converting quaternions to rotation matrices.
    Using equation provided in XSens MVN Manual.
    """
    if len(q.shape) != 2:
        q = q.unsqueeze(0)

    assert q.shape[1] == 4

    r0c0 = q[:,0]**2 + q[:,1]**2 - q[:,2]**2 - q[:,3]**2
    r0c1 = 2*q[:,1]*q[:,2] - 2*q[:,0]*q[:,3]
    r0c2 = 2*q[:,1]*q[:,3] + 2*q[:,0]*q[:,2]

    r1c0 = 2*q[:,1]*q[:,2] + 2*q[:,0]*q[:,3]
    r1c1 = q[:,0]**2 - q[:,1]**2 + q[:,2]**2 - q[:,3]**2
    r1c2 = 2*q[:,2]*q[:,3] - 2*q[:,0]*q[:,1]

    r2c0 = 2*q[:,1]*q[:,3] - 2*q[:,0]*q[:,2]
    r2c1 = 2*q[:,2]*q[:,3] + 2*q[:,0]*q[:,1]
    r2c2 = q[:,0]**2 - q[:,1]**2 - q[:,2]**2 + q[:,3]**2

    r0 = torch.stack([r0c0, r0c1, r0c2], dim=1)
    r1 = torch.stack([r1c0, r1c1, r1c2], dim=1)
    r2 = torch.stack([r2c0, r2c1, r2c2], dim=1)

    R = torch.stack([r0, r1, r2], dim=2)
    
    return R.permute(0, 2, 1)

def rotMat_to_quat(rotMat):
    """
    Converts rotation matrices back to quaternions.
    Ported from https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4
    """

    assert rotMat.shape[1] == 3 and rotMat.shape[2] == 3   
    diffMat = rotMat - torch.transpose(rotMat, 1, 2)

    r = torch.zeros((rotMat.shape[0], 3), dtype=torch.float64)

    r[:,0] = -diffMat[:, 1, 2]
    r[:,1] = diffMat[:, 0, 2]
    r[:,2] = -diffMat[:, 0, 1]

    sin_theta = torch.norm(r, dim=1)/2
    sin_theta = sin_theta.unsqueeze(1)
    
    r0 = r / (torch.norm(r, dim=1).unsqueeze(1) + np.finfo(np.float32).eps)

    cos_theta = (rotMat.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_theta = cos_theta.unsqueeze(1)

    theta = torch.atan2(sin_theta, cos_theta)

    theta = theta.squeeze(1)

    q = torch.zeros((rotMat.shape[0], 4), dtype=torch.float64)

    q[:, 0] = torch.cos(theta/2)
    q[:, 1:] = r0*torch.sin(theta/2).unsqueeze(1)
    
    return q 

def quat_to_expmap(q):
    """
    Quaternion conversion to exponential map.
    Assumes quaternions is of size N x 4*M where N is the number of frames and M is the number joints.
    Ported from https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m
    """
    assert q.shape[1] % 4 == 0
     
    expmap = np.zeros((q.shape[0], 3 * (q.shape[1] // 4)))

    for i in range(q.shape[1] // 4):
        joint_quat = q[:, 4*i:4*(i+1)]
        sin_half_theta = norm(joint_quat[:, 1:])
        cos_half_theta = joint_quat[:, 0]
    
        r0 = np.divide(joint_quat[:,1:], (sin_half_theta + np.finfo(np.float32).eps))
    
        theta = 2 * np.arctan2(sin_half_theta, cos_half_theta)
        theta = np.mod(theta + 2*np.pi, 2*np.pi)
    
        r0[theta > np.pi] *= -1
        theta[theta > np.pi] = 2*np.pi - theta[theta > np.pi]

        theta = np.expand_dims(theta, axis=1)

        expmap[:, 3*i:3*(i+1)] = np.multiply(theta, r0)

    return expmap
