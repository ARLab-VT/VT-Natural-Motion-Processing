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
    def __init__(self, l_mae, l_angle, l_shape):
        super(MotionLoss, self).__init__()
        self.l_mae = l_mae
        self.l_angle = l_angle
        self.l_shape = l_shape

    def maeLoss(self, prediction, target):
        return F.l1_loss(prediction, target, reduction='mean')
 
    def angleLoss(self, prediction, target):
        prediction = prediction.contiguous().view(-1, 4)
        target = target.contiguous().view(-1, 4) 

        R_targ = quat_to_rotMat(prediction)
        R_pred = quat_to_rotMat(target)

        R = R_pred.bmm(torch.transpose(R_targ, 1, 2))

        loss = torch.mean(torch.abs(torch.log(F.relu(R.diagonal(dim1=-2, dim2=-1)) + 1e-8))) 

        return loss
    
    def shapeBasedLoss(self, prediction, target):
        prediction = torch.nn.functional.normalize(prediction, 0, 1)
        target = torch.nn.functional.normalize(target, 0, 1)
    
        prediction = prediction.permute(1, 0, 2)
        target = target.permute(1, 0, 2)
    
        prediction = prediction.unsqueeze(3)
        target = target.unsqueeze(3)
    
        prediction = torch.cat((prediction, torch.zeros_like(prediction)), dim=3)
        target = torch.cat((target, torch.zeros_like(target)), dim=3)
    
        pred_fft = torch.fft(prediction, 1)
        targ_fft = torch.fft(target, 1)
    
        fft_mult_real = pred_fft[...,0]*targ_fft[...,0] + pred_fft[...,1]*targ_fft[...,1]
        fft_mult_imag = pred_fft[...,1]*targ_fft[...,0] - pred_fft[...,0]*targ_fft[...,1]
    
        fft_mult = torch.cat((fft_mult_real.unsqueeze(3), fft_mult_imag.unsqueeze(3)), dim=3)
    
        cc = torch.ifft(fft_mult, 1)[...,0]
    
        pred_norm = torch.norm(prediction[...,0], dim=(1,2)).unsqueeze(1)
        targ_norm = torch.norm(target[...,0], dim=(1,2)).unsqueeze(1)
    
        max_cc_norm, _ = torch.max(torch.sum(cc, 1) / (pred_norm * targ_norm), 1)
    
        loss = torch.abs(torch.mean(torch.ones_like(max_cc_norm, requires_grad=True) - max_cc_norm))
    
        return loss

    def forward(self, prediction, target):        
        mae = self.maeLoss(prediction, target)
        angle = self.angleLoss(prediction, target)
        shape = self.shapeBasedLoss(prediction, target)
        
        return self.l_mae*mae + self.l_angle*angle + self.l_shape*shape

class ShapeLoss(nn.Module):
    def __init__(self):
        super(ShapeLoss, self).__init__()
    
    def shapeBasedLoss(self, prediction, target):
        prediction = torch.nn.functional.normalize(prediction, 0, 1)
        target = torch.nn.functional.normalize(target, 0, 1)
    
        prediction = prediction.permute(1, 0, 2)
        target = target.permute(1, 0, 2)
    
        prediction = prediction.unsqueeze(3)
        target = target.unsqueeze(3)
    
        prediction = torch.cat((prediction, torch.zeros_like(prediction)), dim=3)
        target = torch.cat((target, torch.zeros_like(target)), dim=3)
    
        pred_fft = torch.fft(prediction, 1)
        targ_fft = torch.fft(target, 1)
    
        fft_mult_real = pred_fft[...,0]*targ_fft[...,0] + pred_fft[...,1]*targ_fft[...,1]
        fft_mult_imag = pred_fft[...,1]*targ_fft[...,0] - pred_fft[...,0]*targ_fft[...,1]
    
        fft_mult = torch.cat((fft_mult_real.unsqueeze(3), fft_mult_imag.unsqueeze(3)), dim=3)
    
        cc = torch.ifft(fft_mult, 1)[...,0]
    
        pred_norm = torch.norm(prediction[...,0], dim=(1,2)).unsqueeze(1)
        targ_norm = torch.norm(target[...,0], dim=(1,2)).unsqueeze(1)
    
        max_cc_norm, _ = torch.max(torch.sum(cc, 1) / (pred_norm * targ_norm), 1)
    
        loss = torch.abs(torch.mean(torch.ones_like(max_cc_norm, requires_grad=True) - max_cc_norm))
    
        return loss

    def forward(self, prediction, target): 
        return self.shapeBasedLoss(prediction, target)

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

def fit(model, optimizer, scheduler, epochs, dataloaders, training_criterion, validation_criterion, scaler, device, model_file_path):
    train_dataloader, val_dataloader = dataloaders
    total_time = 0
    min_val_loss = math.inf

    for epoch in range(epochs):
        losses = 0
        logger.info("Epoch {}".format(epoch + 1))
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
                                                                                                                            str(optimizer.param_groups[0]['lr'])))

        with torch.no_grad():
            val_losses, scaled_val_losses = zip(
                *[evaluate(model, data, validation_criterion, scaler, device)
                  for _, data in enumerate(val_dataloader, 0)]
            )
            
        loss = losses / len(train_dataloader)
        val_loss = np.sum(val_losses) / len(val_losses)
        scaled_val_loss = np.sum(scaled_val_losses) / len(scaled_val_losses)
        scheduler.step()
        logger.info("Epoch {} - Training Loss: {} - Val Loss: {} - Scaled Val Loss: {}".format(str(epoch+1), str(loss), str(val_loss), str(scaled_val_loss)))
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            logger.info("Saving model to {}".format(model_file_path))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_file_path)
