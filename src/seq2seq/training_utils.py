# Copyright (c) 2020-present, Assistive Robotics Lab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F
import random
import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from common.logging import logger
from .seq2seq import (
    EncoderRNN,
    DecoderRNN,
    AttnDecoderRNN,
    Attention
)
plt.switch_backend("agg")
torch.manual_seed(0)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def get_encoder(feature_size, device, hidden_size=64,
                dropout=0.0, bidirectional=False):
    """Function to help set up an encoder.

    Args:
        feature_size (int): number of features in the input to this model
        device (torch.device): what device to put this encoder on
        hidden_size (int, optional): number of hidden units in the encoder.
            Defaults to 64.
        dropout (float, optional): dropout to apply in the encoder.
            Defaults to 0.0.
        bidirectional (bool, optional): whether to use a bidirectional encoder.
            Defaults to False.

    Returns:
        EncoderRNN: an encoder for use in seq2seq tasks
    """
    encoder = EncoderRNN(feature_size, hidden_size,
                         dropout=dropout, bidirectional=bidirectional)
    encoder = encoder.double().to(device)
    return encoder


def get_decoder(feature_size, device, hidden_size=64, dropout=0.0):
    """Function to help set up a decoder.

    Args:
        feature_size (int): number of features in the input to this model
        device (torch.device): what device to put this encoder on
        hidden_size (int, optional): number of hidden units in the encoder.
            Defaults to 64.
        dropout (float, optional): dropout to apply in the encoder.
            Defaults to 0.0.

    Returns:
        DecoderRNN: a decoder for use in seq2seq tasks
    """
    decoder = DecoderRNN(feature_size, hidden_size,
                         feature_size, dropout=dropout)
    decoder = decoder.double().to(device)
    return decoder


def get_attn_decoder(feature_size, method, device, batch_size=32,
                     hidden_size=64, bidirectional_encoder=False):
    """Function to help set up a decoder with attention.

    Args:
        feature_size (int): number of features in the input to this model
        method ([type]): [description]
        device (torch.device): what device to put this encoder on
        batch_size (int, optional): [description]. Defaults to 32.
        hidden_size (int, optional): number of hidden units in the encoder.
            Defaults to 64.
        dropout (float, optional): dropout to apply in the encoder.
            Defaults to 0.0.
        bidirectional_encoder (bool, optional): whether the encoder is
            bidirectional. Defaults to False.

    Returns:
        AttnDecoderRNN: a decoder with attention for use in seq2seq tasks
    """
    attn = Attention(hidden_size, batch_size, method,
                     bidirectional_encoder=bidirectional_encoder)
    decoder = AttnDecoderRNN(feature_size, feature_size,
                             hidden_size, hidden_size,
                             attn,
                             bidirectional_encoder=bidirectional_encoder)
    decoder = decoder.double().to(device)
    return decoder


def loss_batch(data, models, opts, criterion, device,
               teacher_forcing_ratio=0.0, use_attention=False,
               norm_quaternions=False, average_batch=True):
    """Train or validate encoder and decoder models on a single batch of data.

    Args:
        data (tuple): tuple containing input and target batches.
        models (tuple): tuple containing encoder and decoder for use during
            training.
        opts (tuple): tuple containing encoder and decoder optimizers
        criterion (nn.Module): criterion to use for training or validation
        device (torch.device): device to put batches on
        teacher_forcing_ratio (float, optional): percent of the time to use
            teacher forcing in the decoder. Defaults to 0.0.
        use_attention (bool, optional): whether the decoder uses attention.
            Defaults to False.
        norm_quaternions (bool, optional): whether the quaternion outputs
            should be normalized. Defaults to False.
        average_batch (bool, optional): For use during training; can be set to
            false for plotting histograms during testing. Defaults to True.

    Returns:
        float or list: will return a single loss or list of losses depending
            on the average_batch flag.
    """
    training = (opts is not None)

    encoder, decoder = models
    input_batch, target_batch = data

    input_batch = input_batch.to(device).double()
    target_batch = target_batch.to(device).double()

    if training:
        encoder.train(), decoder.train()
        encoder_opt, decoder_opt = opts
    else:
        encoder.eval(), decoder.eval()

    loss = 0
    seq_length = target_batch.shape[1]

    input = input_batch.permute(1, 0, 2)
    encoder_outputs, encoder_hidden = encoder(input)

    decoder_hidden = encoder_hidden
    decoder_input = torch.ones_like(target_batch[:, 0, :]).unsqueeze(0)
    EOS = torch.zeros_like(target_batch[:, 0, :]).unsqueeze(0)
    outputs = torch.zeros_like(target_batch)

    use_teacher_forcing = (True if random.random() < teacher_forcing_ratio
                           else False)

    if not average_batch:
        if training:
            logger.warning("average_batch must be true when training")
            sys.exit()
        losses = [0 for i in range(target_batch.shape[0])]

    for t in range(seq_length):

        if use_attention:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        else:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

        target = target_batch[:, t, :].unsqueeze(0)

        output = decoder_output

        if norm_quaternions:
            original_shape = output.shape

            output = output.contiguous().view(-1, 4)
            output = F.normalize(output, p=2, dim=1).view(original_shape)

        outputs[:, t, :] = output

        if use_teacher_forcing:
            decoder_input = target
        else:
            if torch.all(torch.eq(decoder_output, EOS)):
                break
            decoder_input = output.detach()

    loss = criterion(outputs, target_batch)

    if training:
        loss.backward()

        encoder_opt.step()
        encoder_opt.zero_grad()

        decoder_opt.step()
        decoder_opt.zero_grad()

    if average_batch:
        return loss.item()
    else:
        losses = []
        for b in range(outputs.shape[0]):
            sample_loss = criterion(outputs[b, :], target_batch[b, :])
            losses.append(sample_loss.item())

        return losses


def fit(models, optims, epochs, dataloaders, training_criterion,
        validation_criteria, schedulers, device, model_file_path,
        teacher_forcing_ratio=0.0, use_attention=False,
        norm_quaternions=False, schedule_rate=1.0):
    """Fit a seq2seq model to data, logging training and validation loss.

    Args:
        models (tuple): tuple containing the encoder and decoder
        optims (tuple): tuple containing the encoder and decoder optimizers for
            training
        epochs (int): number of epochs to train for
        dataloaders (tuple): tuple containing the training dataloader and val
            dataloader.
        training_criterion (nn.Module): criterion for backpropagation during
            training
        validation_criteria (list): list of criteria for validation
        schedulers (list): list of schedulers to control learning rate for
            optimizers
        device (torch.device): device to place data on
        model_file_path (str): where to save the model when validation loss
            reaches new minimum
        teacher_forcing_ratio (float, optional): percent of the time to use
            teacher forcing in the decoder. Defaults to 0.0.
        use_attention (bool, optional): whether decoder uses attention or not.
            Defaults to False.
        norm_quaternions (bool, optional): whether quaternions should be
            normalized after they are output from the decoder.
            Defaults to False.
        schedule_rate (float, optional): rate to increase or decrease teacher
            forcing ratio. Defaults to 1.0.
    """

    train_dataloader, val_dataloader = dataloaders

    min_val_loss = math.inf
    for epoch in range(epochs):
        losses = []
        total_time = 0

        logger.info(f"Epoch {epoch+1} / {epochs}")

        for index, data in enumerate(train_dataloader, 0):
            with Timer() as timer:
                loss = loss_batch(data, models,
                                  optims, training_criterion, device,
                                  use_attention=use_attention,
                                  norm_quaternions=norm_quaternions)

                losses.append(loss)
            total_time += timer.interval
            if index % (len(train_dataloader) // 10) == 0:
                logger.info((f"Total time elapsed: {total_time} - "
                             f"Batch Number: {index} / {len(train_dataloader)}"
                             f" - Training loss: {loss}"
                             ))
        val_loss = []
        for validation_criterion in validation_criteria:
            with torch.no_grad():
                val_losses = [loss_batch(data, models,
                                         None, validation_criterion, device,
                                         use_attention=use_attention,
                                         norm_quaternions=norm_quaternions)
                              for _, data in enumerate(val_dataloader, 0)]

            val_loss.append(np.sum(val_losses) / len(val_losses))

        loss = np.sum(losses) / len(losses)

        for scheduler in schedulers:
            scheduler.step()

        val_loss_strs = ", ".join(map(str, val_loss))
        logger.info(f"Training Loss: {loss} - Val Loss: {val_loss_strs}")

        teacher_forcing_ratio *= schedule_rate
        if val_loss[0] < min_val_loss:
            min_val_loss = val_loss[0]
            logger.info(f"Saving model to {model_file_path}")
            torch.save({
                "encoder_state_dict": models[0].state_dict(),
                "decoder_state_dict": models[1].state_dict(),
                "optimizerA_state_dict": optims[0].state_dict(),
                "optimizerB_state_dict": optims[1].state_dict(),
            }, model_file_path)
