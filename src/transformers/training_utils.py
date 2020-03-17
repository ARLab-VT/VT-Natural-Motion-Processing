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

def train(model, optimizer, data, training_criterion, device):
    model.train()
    inputs, targets = data

    inputs = inputs.permute(1, 0, 2).to(device).float()
    targets = targets.permute(1, 0, 2).to(device).float()
    output = model(inputs)

    loss = training_criterion(output, targets)
    
    loss.backward()
    
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
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

def test(model, dataloader, criterion, device):
    with torch.no_grad():
        test_losses, scaled_test_losses = zip(
            *[evaluate(model, data, criterion, None, device)
                for _, data in enumerate(dataloader, 0)]
        )
 
    test_loss = np.sum(test_losses) / len(test_losses)

    
    logger.info("Test Loss: {}".format(str(test_loss)))


def fit(model, optimizer, scheduler, epochs, dataloaders, training_criterion, validation_criteria, scaler, device, model_file_path):
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
        val_loss = []
        for validation_criterion in validation_criteria:
            with torch.no_grad():
                val_losses, scaled_val_losses = zip(
                    *[evaluate(model, data, validation_criterion, scaler, device)
                      for _, data in enumerate(val_dataloader, 0)]
                )

            val_loss.append(np.sum(val_losses) / len(val_losses))
            
        loss = losses / len(train_dataloader)
        scaled_val_loss = np.sum(scaled_val_losses) / len(scaled_val_losses)
        scheduler.step()
        logger.info("Epoch {} - Training Loss: {} - Val Loss: {} - Scaled Val Loss: {}".format(str(epoch+1), str(loss), ", ".join(map(str, val_loss)), str(scaled_val_loss)))
        
        if val_loss[0] < min_val_loss:
            min_val_loss = val_loss[0]
            logger.info("Saving model to {}".format(model_file_path))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_file_path)
