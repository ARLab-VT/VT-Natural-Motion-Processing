#!/usr/bin/env python
# coding: utf-8

from transformers.training_utils import *
from transformers.transformers import *
from common.data_utils import *
from common.logging import logger
from common.losses import *
from pathlib import Path
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset, RandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import sys
import math
import random
import os
import time
import argparse
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task',
                        help='task for neural network to train on; either prediction or conversion')
    parser.add_argument('--data-path-parent',
                        help='path to parent folder of directories with h5 files (folders must contain normalization.h5 and validation.h5)')
    parser.add_argument('--figure-file-path',
                        help='path to where the figure should be saved') 
    parser.add_argument('--model-dir',
                        help='path to model file directory')
    parser.add_argument('--representation',
                        help='data representation (quaternions, expmap, rotmat)', default='quaternions')
    parser.add_argument('--full-transformer',
                        help='will use Transformer with both encoder and decoder if true, will only use encoder if false', default=False, action='store_true')
    parser.add_argument('--batch-size',
                        help='batch size for training', default=32)
    parser.add_argument('--seq-length',
                        help='sequence length for model, will be downsampled if downsample is provided', default=20)
    parser.add_argument('--downsample',
                        help='reduce sampling frequency of recorded data; default sampling frequency is 240 Hz', default=1)
    parser.add_argument('--in-out-ratio',
                        help='ratio of input/output; seq_length / downsample = input length = 10, output length = input length / in_out_ratio', default=1)
    parser.add_argument('--stride',
                        help='stride used when reading data in for running prediction tasks', default=3)
    parser.add_argument('--num-heads',
                        help='number of heads in Transformer Encoder')
    parser.add_argument('--dim-feedforward',
                        help='number of dimensions in feedforward layer in Transformer Encoder')
    parser.add_argument('--dropout',
                        help='dropout percentage in Transformer Encoder')
    parser.add_argument('--num-layers',
                        help='number of layers in Transformer Encoder')

    args = parser.parse_args()

    if args.data_path_parent is None:
        parser.print_help()

    return args   


if __name__ == "__main__":
    args = parse_args()
    
    for arg in vars(args):
        logger.info("{} - {}".format(arg, getattr(args, arg)))

    logger.info("Starting Transformer testing...")
    
    logger.info("Device count: {}".format(str(torch.cuda.device_count())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Testing on {}...".format(device))    
    seq_length = int(args.seq_length)//int(args.downsample)

    data_paths = [args.data_path_parent + '/' + name for name in os.listdir(args.data_path_parent) if os.path.isdir(args.data_path_parent + '/' + name)]
    model_paths = [args.model_dir + '/' + name for name in os.listdir(args.model_dir)]

    data_paths.sort()
    model_paths.sort()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i, data_path in enumerate(data_paths):
        args.data_path = data_path
        normalize = True    
        with h5py.File(args.data_path + "/normalization.h5", "r") as f:
            mean, std_dev = torch.Tensor(f["mean"]), torch.Tensor(f["std_dev"])
        norm_data = (mean, std_dev)

        test_dataloader, _ = load_dataloader(args, "testing", normalize, norm_data=norm_data)
     
        encoder_feature_size = test_dataloader.dataset[0][0].shape[1]
        decoder_feature_size = test_dataloader.dataset[0][1].shape[1]

        num_heads = int(args.num_heads) if args.full_transformer else encoder_feature_size
        dim_feedforward = int(args.dim_feedforward)
        dropout = float(args.dropout)
        num_layers = int(args.num_layers)
        quaternions = (args.representation == "quaternions")
       
        if args.full_transformer:
            model = InferenceTransformer(decoder_feature_size, num_heads, dim_feedforward, dropout, num_layers, quaternions=quaternions)
        else:
            num_heads = encoder_feature_size
            model = InferenceTransformerEncoder(encoder_feature_size, num_heads, dim_feedforward, dropout, num_layers, decoder_feature_size, quaternions=quaternions)    

        checkpoint = torch.load(model_paths[i], map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(device).double()

        criterion = QuatDistance()
     
        if args.full_transformer:    
            with torch.no_grad():
                inference_losses = [inference(model, data, criterion, device, average_batch=False)
                                    for _, data in enumerate(test_dataloader, 0)]
        else:
            with torch.no_grad():
                inference_losses = [loss_batch(model, None, data, criterion, device, full_transformer=args.full_transformer, average_batch=False)
                                    for _, data in enumerate(test_dataloader, 0)]

        def flatten(l): return [item for sublist in l for item in sublist]

        inference_losses = flatten(inference_losses)
    
        inference_loss = np.sum(inference_losses) / len(inference_losses)
        logger.info("Inference Loss: {}".format(inference_loss)) 
        
        ax.hist(inference_losses, bins=60, density=True, histtype=u'step')
        ax.set_xlim(0, 30)
        ax.set_xticks(range(0, 35, 5))
        ax.set_xticklabels(range(0, 35, 5))
    
    ax.set_xlabel('Sequence Angular Error in Degrees')
    ax.set_ylabel('Percentage')
    ax.legend(['Config. 1', 'Config. 2', 'Config. 3', 'Config. 4']) 
    figname = args.figure_file_path
    fig.savefig(figname, bbox_inches='tight')

    logger.info("Completed testing...")
    logger.info("\n")
