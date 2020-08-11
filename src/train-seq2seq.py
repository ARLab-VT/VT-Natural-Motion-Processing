# Copyright (c) 2020-present, Assistive Robotics Lab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from seq2seq.training_utils import (
    fit,
    get_encoder,
    get_decoder,
    get_attn_decoder
)
from common.losses import QuatDistance
from common.data_utils import load_dataloader
from common.logging import logger
import torch
from torch import nn, optim
import numpy as np
import argparse

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse arguments for module.

    Returns:
        argparse.Namespace: contains accessible arguments passed in to module
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--task",
                        help=("task for neural network to train on; "
                              "either prediction or conversion"))
    parser.add_argument("--data-path",
                        help=("path to h5 files containing data "
                              "(must contain training.h5 and validation.h5)"))
    parser.add_argument("--model-file-path",
                        help="path to model file for saving it after training")
    parser.add_argument("--representation",
                        help="data representation", default="quaternion")
    parser.add_argument("--auxiliary-acc",
                        help="will train on auxiliary acceleration if true",
                        default=False,
                        action="store_true")
    parser.add_argument("--batch-size",
                        help="batch size for training", default=32)
    parser.add_argument("--learning-rate",
                        help="learning rate for encoder and decoder",
                        default=0.001)
    parser.add_argument("--seq-length",
                        help="sequence length for encoder/decoder",
                        default=20)
    parser.add_argument("--downsample",
                        help="reduce sampling frequency of recorded data; "
                             "default sampling frequency is 240 Hz",
                        default=1)
    parser.add_argument("--in-out-ratio",
                        help=("ratio of input/output; "
                              "seq_length / downsample = input length = 10, "
                              "output length = input length / in_out_ratio"),
                        default=1)
    parser.add_argument("--stride",
                        help="stride used when running prediction tasks",
                        default=3)
    parser.add_argument("--num-epochs",
                        help="number of epochs for training", default=1)
    parser.add_argument("--hidden-size",
                        help="hidden size in both the encoder and decoder")
    parser.add_argument("--dropout",
                        help="dropout percentage in encoder and decoder",
                        default=0.0)
    parser.add_argument("--bidirectional",
                        help="will use bidirectional encoder",
                        default=False,
                        action="store_true")
    parser.add_argument("--attention",
                        help="will use decoder with given attention method",
                        default="general")

    args = parser.parse_args()

    if args.data_path is None:
        parser.print_help()

    return args


if __name__ == "__main__":
    args = parse_args()

    for arg in vars(args):
        logger.info(f"{arg} - {getattr(args, arg)}")

    logger.info("Starting seq2seq model training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_length = int(args.seq_length)
    stride = int(args.stride)
    lr = float(args.learning_rate)

    assert seq_length % int(args.in_out_ratio) == 0

    normalize = True
    train_dataloader, norm_data = load_dataloader(args, "training",
                                                  normalize, norm_data=None)
    val_dataloader, _ = load_dataloader(args, "validation",
                                        normalize, norm_data=norm_data)

    encoder_feature_size = train_dataloader.dataset[0][0].shape[1]
    decoder_feature_size = train_dataloader.dataset[0][1].shape[1]

    bidirectional = args.bidirectional
    encoder = get_encoder(encoder_feature_size,
                          device,
                          hidden_size=int(args.hidden_size),
                          dropout=float(args.dropout),
                          bidirectional=bidirectional)

    encoder_optim = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.05)
    encoder_sched = optim.lr_scheduler.MultiStepLR(encoder_optim,
                                                   milestones=[5],
                                                   gamma=0.1)

    use_attention = False
    attention_options = ["add", "dot", "concat",
                         "general", "activated-general", "biased-general"]
    if args.attention in attention_options:
        decoder = get_attn_decoder(decoder_feature_size,
                                   args.attention,
                                   device,
                                   hidden_size=int(args.hidden_size),
                                   bidirectional_encoder=bidirectional)
        use_attention = True
    else:
        decoder = get_decoder(decoder_feature_size,
                              device,
                              dropout=float(args.dropout),
                              hidden_size=int(args.hidden_size))

    decoder_optim = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.05)
    decoder_sched = optim.lr_scheduler.MultiStepLR(decoder_optim,
                                                   milestones=[5],
                                                   gamma=0.1)

    encoder_params = sum(p.numel()
                         for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel()
                         for p in decoder.parameters() if p.requires_grad)

    models = (encoder, decoder)
    optims = (encoder_optim, decoder_optim)
    dataloaders = (train_dataloader, val_dataloader)
    epochs = int(args.num_epochs)
    training_criterion = nn.L1Loss()
    validation_criteria = [nn.L1Loss(), QuatDistance()]
    norm_quaternions = (args.representation == "quaternions")

    schedulers = (encoder_sched, decoder_sched)

    logger.info(f"Encoder for training: {encoder}")
    logger.info(f"Decoder for training: {decoder}")
    logger.info(f"Number of parameters: {encoder_params + decoder_params}")
    logger.info(f"Optimizers for training: {encoder_optim}")
    logger.info(f"Criterion for training: {training_criterion}")

    fit(models, optims, epochs, dataloaders, training_criterion,
        validation_criteria, schedulers, device, args.model_file_path,
        use_attention=use_attention, norm_quaternions=norm_quaternions)

    logger.info("Completed Training...")
    logger.info("\n")
