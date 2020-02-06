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

def savePlot(points, epochs, filename):
    points = np.array(points)
    plt.figure()
    fig, ax = plt.subplots()
    x = range(1, epochs + 1)
    plt.plot(x, points[:, 0], 'b-')
    plt.plot(x, points[:, 1], 'r-')
    plt.legend(['training loss', 'val loss'])
    plt.savefig(filename)


def plotLossesOverTime(losses_over_time):
    losses_over_time = np.array(losses_over_time)
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(losses_over_time)
    plt.xlabel('frames')
    plt.xticks(np.arange(1, 21))
    plt.ylabel('scaled MAE loss')

class MotionLoss(nn.Module):
    def __init__(self, lambdas, stats):
        super(MotionLoss, self).__init__()
        self.l_mae = lambdas[0]
        self.l_angle = lambdas[1]
        self.l_anthro = lambdas[2]
        self.mu = stats[0]
        self.cov = stats[1]

    def maeLoss(self, prediction, target):
        return F.l1_loss(prediction, target, reduction='mean')
 
    def angleLoss(self, prediction, target, device):
        prediction = prediction.contiguous().view(-1, 4)
        target = target.contiguous().view(-1, 4) 

        R_targ = quat_to_rotMat(prediction)
        R_pred = quat_to_rotMat(target)

        R = R_pred.bmm(torch.transpose(R_targ, 1, 2))

        loss = torch.mean(torch.abs(torch.log(F.relu(R.diagonal(dim1=-2, dim2=-1)) + 1e-8))) 

        return loss
    #def anthropometricLoss(self, prediction):
        

    def forward(self, prediction, target):
        device = prediction.device
        
        mae = self.maeLoss(prediction, target)
        angle = self.angleLoss(prediction, target, device)
        #anthro = self.anthropometricLoss(prediction)
        
        return self.l_mae*mae + self.l_angle*angle #+ self.l_anthro*anthro

def train(model, optimizer, data, training_criterion, device):
    model.train()
    inputs, targets = data

    inputs = inputs.permute(1, 0, 2).to(device).float()
    targets = targets.permute(1, 0, 2).to(device).float()
    output = model(inputs)

    loss = training_criterion(output, targets)
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

def evaluate(model, data, validation_criterion, scaler, device):
    model.eval()
    inputs, targets = data
    inputs = inputs.permute(1, 0, 2).to(device).float()
    targets = targets.permute(1, 0, 2).to(device).float()
    output = model(inputs)

    loss = validation_criterion(output, targets)
    scaled_loss = 0
    for i in range(output.shape[1]):
        if scaler is None:
            break
        target_i = targets[:, i, :]
        output_i = output[:, i, :]
        
        scaled_loss_temp = validation_criterion(torch.tensor(scaler.inverse_transform(output_i.cpu())),
                                                torch.tensor(scaler.inverse_transform(target_i.cpu())))
        scaled_loss += scaled_loss_temp    

    return loss.item(), scaled_loss / output.shape[1] 

def fit(model, optimizer, scheduler, epochs, dataloaders, training_criterion, validation_criterion, scaler, device):
    train_dataloader, val_dataloader = dataloaders
    total_time = 0
    for epoch in range(epochs):
        losses = 0
        logger.info("Epoch {}".format(epoch))
        for index, data in enumerate(train_dataloader, 0):
            with Timer() as timer:
                loss = train(model, optimizer, data, training_criterion, device) 
            losses += loss
            total_time += timer.interval
            if index % (len(train_dataloader) // 10) == 0:
                logger.info("Total time elapsed: {} - Batch number: {} / {} - Training loss: {} - Learning Rate: {}".format(str(total_time), 
                                                                                                                            str(index), 
                                                                                                                            str(len(train_dataloader)), 
                                                                                                                            str(loss),
                                                                                                                            str(scheduler.get_lr()[0])))
        
        with torch.no_grad():
            val_losses, scaled_val_losses = zip(
                *[evaluate(model, data, validation_criterion, scaler, device)
                  for _, data in enumerate(val_dataloader, 0)]
            )
            
        loss = losses / len(train_dataloader)
        val_loss = np.sum(val_losses) / len(val_losses)
        scaled_val_loss = np.sum(scaled_val_losses) / len(scaled_val_losses)
        scheduler.step()
        logger.info("Epoch {} - Training Loss: {} - Val Loss: {} - Scaled Val Loss: {}".format(str(epoch), str(loss), str(val_loss), str(scaled_val_loss)))
