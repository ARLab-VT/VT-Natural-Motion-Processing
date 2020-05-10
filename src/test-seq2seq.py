#!/usr/bin/env python
# coding: utf-8

from seq2seq.training_utils import *
from seq2seq.seq2seq import *
from common.losses import *
from common.data_utils import *
from common.logging import *
from pathlib import Path
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset, RandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import random
import os
import time
import argparse
import h5py

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--task',
                        help='task for neural network to train on; either prediction or conversion')
    parser.add_argument('--data-path-parent',
                        help='path to parent of h5 files containing data (each subfolder must contain normalization.h5 and validation.h5)')
    parser.add_argument('--figure-file-path',
                        help='path to where the histogram plot should be saved')
    parser.add_argument('--model-dir',
                        help='path to model files')
    parser.add_argument('--representation',
                        help='will normalize if quaternions, will use expmap to quat validation loss if expmap', default='quaternion')
    parser.add_argument('--batch-size',
                        help='batch size for training', default=32)
    parser.add_argument('--seq-length',
                        help='sequence length for encoder/decoder', default=20)    
    parser.add_argument('--downsample',
                        help='reduce sampling frequency of recorded data; default sampling frequency is 240 Hz', default=1)
    parser.add_argument('--stride',
                        help='stride used when running prediction tasks', default=3)
    parser.add_argument('--hidden-size',
                        help='hidden size in both the encoder and decoder')
    parser.add_argument('--dropout',
                        help='dropout percentage in encoder and decoder', default=0.0)
    parser.add_argument('--bidirectional',
                        help='will use bidirectional encoder', default=False, action='store_true')
    parser.add_argument('--attention',
                        help='will use decoder with attention with this method', default='general')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    for arg in vars(args):
        logger.info("{} - {}".format(arg, getattr(args, arg)))
    
    logger.info("Starting seq2seq model training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seq_length = int(args.seq_length)
    stride = int(args.stride)
    batch_size = int(args.batch_size)

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
        val_dataloader, _ = load_dataloader(args, "validation", normalize, norm_data=norm_data)

        encoder_feature_size = val_dataloader.dataset[0][0].shape[1]
        decoder_feature_size = val_dataloader.dataset[0][1].shape[1]
     
        encoder = get_encoder(encoder_feature_size,
                              device,
                              hidden_size=int(args.hidden_size),
                              dropout=float(args.dropout),
                              bidirectional=args.bidirectional)

        use_attention = False
        if args.attention in ["add", "dot", "concat", "general", "activated-general", "biased-general"]:
            decoder = get_attn_decoder(decoder_feature_size,
                                       args.attention,
                                       device,
                                       hidden_size=int(args.hidden_size),
                                       bidirectional_encoder=args.bidirectional)
            use_attention = True
        else:
            decoder = get_decoder(decoder_feature_size,
                                  device,
                                  dropout=float(args.dropout),
                                  hidden_size=int(args.hidden_size))

        checkpoint = torch.load(model_paths[i], map_location=device)

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        decoder.batch_size = batch_size
        if use_attention: 
            decoder.attention.batch_size = batch_size

        encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)    
        decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

        models = (encoder.double(), decoder.double())
        criterion = QuatDistance()
        norm_quaternions = (args.representation == "quaternions")

        if args.representation == "expmap":
            validation_criteria.append(ExpmapToQuatLoss())

        with torch.no_grad():
            inference_losses = [loss_batch(data, models,
                                           None, criterion, device,
                                           use_attention=use_attention,
                                           norm_quaternions=norm_quaternions,
                                           average_batch=False)
                                           for _, data in enumerate(val_dataloader, 0)]


        def flatten(l): return [item for sublist in l for item in sublist]

        inference_losses = flatten(inference_losses)
            
        ax.hist(inference_losses, bins=60, density=True, histtype=u'step')
        ax.set_xlim(0, 30)
        ax.set_xticks(range(0, 35, 5))
        ax.set_xticklabels(range(0, 35, 5))
   
        inference_loss = np.sum(inference_losses) / len(inference_losses)
        logger.info("Inference Loss for {}: {}".format(data_path.split("/")[-1], inference_loss)) 
 
    ax.set_xlabel('Sequence Angular Error in Degrees')
    ax.set_ylabel('Percentage')
    ax.legend(['Config. 1', 'Config. 2', 'Config. 3', 'Config. 4'])
    figname = args.figure_file_path
    fig.savefig(figname, bbox_inches='tight')

    logger.info("Completed Testing...") 
    logger.info("\n")
