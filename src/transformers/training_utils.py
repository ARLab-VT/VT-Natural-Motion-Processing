#!/usr/bin/env python
# coding: utf-8
import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from common.logging import logger
plt.switch_backend("agg")
torch.manual_seed(0)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def loss_batch(model, optimizer, data, criterion, device,
               full_transformer=False, average_batch=True):
    """Run a batch through the Transformer Encoder or Transformer for training.

    Used in the fit function to train the models.

    Args:
        model (nn.Module): model to pass batch through
        optimizer (optim.Optimizer): optimizer for training
        data (tuple): contains input and targets to use for inference
        criterion (nn.Module): criterion to evaluate model on
        device (torch.device): device to put data on
        full_transformer (bool, optional): whether the model is a full
            transformer; the forward pass will operate differently if so.
            Defaults to False.
        average_batch (bool, optional): whether to average the batch; useful
            for plotting histograms if average_batch is false.
            Defaults to True.

    Returns:
        float or list: returns a single loss or a list of losses depending on
            average_batch argument.
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    inputs, targets = data
    inputs = inputs.permute(1, 0, 2).to(device).double()
    targets = targets.permute(1, 0, 2).to(device).double()

    outputs = None
    if full_transformer:
        SOS = torch.zeros_like(targets[0, :]).unsqueeze(0)
        tgt = torch.cat((SOS, targets), dim=0)

        if tgt.shape[2] > inputs.shape[2]:
            padding = torch.zeros((inputs.shape[0], inputs.shape[1],
                                   tgt.shape[2] - inputs.shape[2]))
            padding = padding.to(device).double()
            inputs = torch.cat((inputs, padding), dim=2)

        outputs = model(inputs, tgt[:-1, :])
    else:
        outputs = model(inputs)

    loss = criterion(outputs, targets)

    if optimizer is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()

    if average_batch:
        return loss.item()
    else:
        losses = []
        for b in range(outputs.shape[1]):
            sample_loss = criterion(outputs[:, b, :], targets[:, b, :])
            losses.append(sample_loss.item())

        return losses


def inference(model, data, criterion, device, average_batch=True):
    """Run inference with full Transformer.

    Used to determine actual validation loss that will be seen in practice
    since full Transformers use teacher forcing.


    Args:
        model (nn.Module): model to run inference with
        data (tuple): contains input and targets to use for inference
        criterion (nn.Module): criterion to evaluate model on
        device (torch.device): device to put data on
        average_batch (bool, optional): whether to average the batch; useful
            for plotting histograms if average_batch is false.
            Defaults to True.

    Returns:
        float or list: returns a single loss or a list of losses depending on
            average_batch argument.
    """
    model.eval()

    inputs, targets = data
    inputs = inputs.permute(1, 0, 2).to(device).double()
    targets = targets.permute(1, 0, 2).to(device).double()

    pred = torch.zeros((targets.shape[0]+1,
                        targets.shape[1],
                        targets.shape[2])).to(device).double()

    if pred.shape[2] > inputs.shape[2]:
        padding = torch.zeros((inputs.shape[0],
                               inputs.shape[1],
                               pred.shape[2] - inputs.shape[2]))
        padding = padding.to(device).double()
        inputs = torch.cat((inputs.clone(), padding), dim=2)

    memory = model.encoder(model.pos_decoder(inputs))

    for i in range(pred.shape[0]-1):
        next_pred = model.inference(memory, pred[:i+1, :].clone())
        pred[i+1, :] = pred[i+1, :].clone() + next_pred[-1, :].clone()

    if average_batch:
        loss = criterion(pred[1:, :], targets)
        return loss.item()
    else:
        losses = []
        for b in range(pred.shape[1]):
            loss = criterion(pred[1:, b, :], targets[:, b, :])
            losses.append(loss.item())
        return losses


def fit(model, optimizer, scheduler, epochs, dataloaders, training_criterion,
        validation_criteria, device, model_file_path,
        full_transformer=False, min_val_loss=math.inf):
    """Fit a Transformer model to data, logging training and validation loss.

    Args:
        model (nn.Module): the model to train
        optimizer (tuple): the optimizer for training
        scheduler (list): scheduler to control learning rate for
            optimizers
        epochs (int): number of epochs to train for
        dataloaders (tuple): tuple containing the training dataloader and val
            dataloader.
        training_criterion (nn.Module): criterion for backpropagation during
            training
        validation_criteria (list): list of criteria for validation
        device (torch.device): device to place data on
        model_file_path (str): where to save the model when validation loss
            reaches new minimum
        full_transformer (bool): whether the model is a full transformer and
            needs to run inference for evaluation
        min_val_loss (float, optional): minimum validation loss

    Returns:
        float: minimum validation loss reached during training
    """
    train_dataloader, val_dataloader = dataloaders
    total_time = 0

    for epoch in range(epochs):
        losses = 0
        logger.info("Epoch {}".format(epoch + 1))
        avg_loss = 0
        for index, data in enumerate(train_dataloader, 0):
            with Timer() as timer:
                loss = loss_batch(model, optimizer, data, training_criterion,
                                  device, full_transformer=full_transformer)
            losses += loss
            avg_loss += loss
            total_time += timer.interval
            if index % (len(train_dataloader) // 10) == 0 and index != 0:
                avg_training_loss = avg_loss / (len(train_dataloader) // 10)
                logger.info((f"Total time elapsed: {total_time}"
                             " - "
                             f"Batch number: {index} / {len(train_dataloader)}"
                             " - "
                             f"Training loss: {avg_training_loss}"
                             " - "
                             f"LR: {optimizer.param_groups[0]['lr']}"
                             ))
                avg_loss = 0

        val_loss = []
        for validation_criterion in validation_criteria:
            with torch.no_grad():
                val_losses = [loss_batch(model, None, data,
                                         validation_criterion, device,
                                         full_transformer=full_transformer)
                              for _, data in enumerate(val_dataloader, 0)]

            val_loss.append(np.sum(val_losses) / len(val_losses))

        loss = losses / len(train_dataloader)

        scheduler.step()
        val_loss_str = ", ".join(map(str, val_loss))
        logger.info(f"Epoch {epoch+1} - "
                    f"Training Loss: {loss} - "
                    f"Val Loss: {val_loss_str}")

        if full_transformer:
            inference_loss = []
            for validation_criterion in validation_criteria:
                with torch.no_grad():
                    inference_losses = [inference(model, data,
                                                  validation_criterion, device)
                                        for _, data in
                                        enumerate(val_dataloader, 0)]
                inference_loss.append(
                    np.sum(inference_losses) / len(inference_losses)
                    )
            inference_loss_str = ", ".join(map(str, inference_loss))
            logger.info(f"Inference Loss: {inference_loss_str}")

        if val_loss[0] < min_val_loss:
            min_val_loss = val_loss[0]
            logger.info(f"Saving model to {model_file_path}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, model_file_path)

    return min_val_loss
