import numpy as np
import pandas as pd
import warnings
import h5py
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from scipy.linalg import norm


def euler_to_quaternion(joint_angles, order):
    """
    Euler angle conversion to quaternions.
    XSens stores joint angles in ZXY format typically unless using jointAngleXZY format.
    Wrapping scipy library for our purposes.
    """
    assert joint_angles.shape[1] % 3 == 0

    quat = np.zeros((joint_angles.shape[0], 4 * (joint_angles.shape[1] // 3)))
       
    for i in range(joint_angles.shape[1] // 3):
        r = R.from_euler(order, joint_angles[:, 3*i:3*(i+1)], degrees=True)
        quat[:, 4*i:4*(i+1)] = r.as_quat()

    return quat


def quaternion_to_expmap(quat):
    """
    Quaternion conversion to exponential map.
    Assumes quaternions is of size N x 4*M where N is the number of frames and M is the number joints.
    Ported from https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m
    """
    assert quat.shape[1] % 4 == 0
     
    expmap = np.zeros((quat.shape[0], 3 * (quat.shape[1] // 4)))

    for i in range(quat.shape[1] // 4):
        joint_quat = quat[:, 4*i:4*(i+1)]
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
