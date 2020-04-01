import matplotlib.ticker as ticker
import torch
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
import torch.nn.functional as F
from common.logging import logger
from common.conversions import *

# importing models
plt.switch_backend('agg')
torch.manual_seed(0)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


class ExpmapToQuatLoss(nn.Module):
    def __init__(self):
        super(ExpmapToQuatLoss, self).__init__()        

    def forward(self, predictions, targets):
        predictions = predictions.view(-1,3)
        targets = targets.view(-1,3)

        predictions = rotMat_to_quat(expmap_to_rotMat(predictions.double()))
        targets = rotMat_to_quat(expmap_to_rotMat(targets.double()))

        return F.l1_loss(predictions, targets)

def train(model, optimizer, data, training_criterion, device, full_transformer=False):
    model.train()
    inputs, targets = data
    inputs = inputs.permute(1, 0, 2).to(device).double()
    targets = targets.permute(1, 0, 2).to(device).double()

    if full_transformer:
        last_input = inputs[-1, :].unsqueeze(0)
        tgt = torch.cat((last_input, targets), dim=0)
        output = model(inputs, tgt[:-1, :])
    else:
        output = model(inputs)

    loss = training_criterion(output, targets)
  
    loss.backward()
    
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

def loss_batch(model, optimizer, data, criterion, device, full_transformer=False):

    if optimizer is None:
        model.eval()
    else:
        model.train()   

    inputs, targets = data
    inputs = inputs.permute(1, 0, 2).to(device).double()
    targets = targets.permute(1, 0, 2).to(device).double()
    
    output = None
    if full_transformer:
        SOS = torch.zeros_like(targets[0, :]).unsqueeze(0)
        tgt = torch.cat((SOS, targets), dim=0)
        
        if tgt.shape[2] > inputs.shape[2]:
            padding = torch.zeros((inputs.shape[0], inputs.shape[1], tgt.shape[2] - inputs.shape[2])).to(device).double()
            inputs = torch.cat((inputs, padding), dim=2)
        
        output = model(inputs, tgt[:-1,:])
    else:
        output = model(inputs)    
    
    loss = criterion(output, targets)
            
    if optimizer is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

def inference(model, data, criterion, device):
    model.eval()

    inputs, targets = data
    inputs = inputs.permute(1, 0, 2).to(device).double()
    targets = targets.permute(1, 0, 2).to(device).double()
    
    pred = torch.zeros((targets.shape[0]+1, targets.shape[1], targets.shape[2])).to(device).double()

    if pred.shape[2] > inputs.shape[2]:
        padding = torch.zeros((inputs.shape[0], inputs.shape[1], pred.shape[2] - inputs.shape[2])).to(device).double()
        inputs = torch.cat((inputs, padding), dim=2)
    
    for i in range(pred.shape[0]-1):
        next_pred = model(inputs, pred[:i+1,:])
        pred[i+1,:] = next_pred[-1, :]
    
    loss = criterion(pred[1:,:], targets)
    
    return loss.item()


def fit(model, optimizer, scheduler, epochs, dataloaders, training_criterion, validation_criteria, device, model_file_path, full_transformer=False):
    train_dataloader, val_dataloader = dataloaders
    total_time = 0
    min_val_loss = math.inf

    for epoch in range(epochs):
        losses = 0
        logger.info("Epoch {}".format(epoch + 1))
        avg_loss = 0
        for index, data in enumerate(train_dataloader, 0):
            with Timer() as timer:
                loss = loss_batch(model, optimizer, data, training_criterion, device, full_transformer=full_transformer)
            losses += loss
            avg_loss += loss
            total_time += timer.interval
            if index % (len(train_dataloader) // 10) == 0 and index != 0:
                logger.info("Total time elapsed: {} - Batch number: {} / {} - Training loss: {} - Learning Rate: {}".format(str(total_time), 
                                                                                                                            str(index), 
                                                                                                                            str(len(train_dataloader)), 
                                                                                                                            str(avg_loss / (len(train_dataloader) // 10)),
                                                                                                                            str(optimizer.param_groups[0]['lr'])))
                avg_loss = 0

        val_loss = []
        for validation_criterion in validation_criteria:
            with torch.no_grad():
                val_losses = [loss_batch(model, None, data, validation_criterion, device, full_transformer=full_transformer)
                                         for _, data in enumerate(val_dataloader, 0)]

            val_loss.append(np.sum(val_losses) / len(val_losses))
            
        loss = losses / len(train_dataloader)

        scheduler.step()
        logger.info("Epoch {} - Training Loss: {} - Val Loss: {}".format(str(epoch+1), str(loss), ", ".join(map(str, val_loss))))
        
        if full_transformer:
            with torch.no_grad():
                inference_losses = [inference(model, data, training_criterion, device)
                                    for _, data in enumerate(val_dataloader, 0)]
            inference_loss = np.sum(inference_losses) / len(inference_losses)
            logger.info("Inference Loss: {}".format(str(inference_loss)))

        if val_loss[0] < min_val_loss:
            min_val_loss = val_loss[0]
            logger.info("Saving model to {}".format(model_file_path))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_file_path)

    
