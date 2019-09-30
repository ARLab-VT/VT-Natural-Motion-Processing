import matplotlib.ticker as ticker
import torch
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from torch import optim


#importing models
from .models import EncoderRNN, DecoderRNN, AttnDecoderRNN, Attention

plt.switch_backend('agg')
torch.manual_seed(0)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


class ProgressBar:
    def __init__(self, size, step_size):
        self.size = size
        self.step_size = step_size

    def print_progress_bar(self, index, total_time, loss):
        amount_complete = math.floor(
            (index + 1) / self.size * 30)
        amount_incomplete = (30 - amount_complete)
        progress_bar = "[" + amount_complete * \
            "=" + amount_incomplete * "-" + "]"

        print("\r%d/%d" % (self.step_size * (index + 1), self.step_size * self.size),
              progress_bar,
              "- Time elapsed: %.2fs" % total_time,
              "- Loss: %.8f" % loss,
              end=""
              )


def get_encoder(num_features, device, hidden_size=64, lr=0.001, dropout=0.0, bs=32, bidirectional=False):
    encoder = EncoderRNN(num_features, hidden_size, bs,
                         dropout=dropout, bidirectional=bidirectional).to(device)
    return encoder, optim.Adam(encoder.parameters(), lr=lr)


def get_decoder(num_features, device, hidden_size=64, lr=0.001, dropout=0.0, bs=32):
    decoder = DecoderRNN(num_features, hidden_size,
                         num_features, bs, dropout=dropout).to(device)
    return decoder, optim.Adam(decoder.parameters(), lr=lr)


def get_attn_decoder(num_features, method, device, hidden_size=64, lr=0.001, bs=32, bidirectional_encoder=False):
    attn = Attention(hidden_size, bs, method,
                     bidirectional_encoder=bidirectional_encoder)
    decoder = AttnDecoderRNN(num_features, num_features, hidden_size, hidden_size,
                             attn, bidirectional_encoder=bidirectional_encoder).to(device)
    return decoder, optim.Adam(decoder.parameters(), lr=lr)


def showPlot(points, epochs):
    points = np.array(points)
    plt.figure()
    fig, ax = plt.subplots()
    x = range(1, epochs + 1)
    plt.plot(x, points[:, 0], 'b-')
    plt.plot(x, points[:, 1], 'r-')
    plt.legend(['training loss', 'val loss'])


def plotLossesOverTime(losses_over_time):
    losses_over_time = np.array(losses_over_time)
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(losses_over_time)
    plt.xlabel('frames')
    plt.xticks(np.arange(1, 21))
    plt.ylabel('scaled MAE loss')


def loss_batch(data, models, opts, criterion, device, scaler=None, teacher_forcing_ratio=0.0, use_attention=False):
    encoder, decoder = models
    input_batch, target_batch = data

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    if opts is not None:
        encoder.train(), decoder.train()
        encoder_opt, decoder_opt = opts
    else:
        encoder.eval(), decoder.eval()

    loss = 0
    scaled_loss = 0
    seq_length = input_batch.shape[1]
    scaled_loss_over_time = torch.zeros(seq_length)

    input = input_batch.permute(1, 0, 2).float()
    encoder_outputs, encoder_hidden = encoder(input)

    decoder_hidden = encoder_hidden
    decoder_input = torch.ones_like(target_batch[:, 0, :]).unsqueeze(0).float()
    EOS = torch.zeros_like(target_batch[:, 0, :]).unsqueeze(0).float()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for t in range(seq_length):

        if use_attention:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        else:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

        target = target_batch[:, t, :].unsqueeze(0).float()

        loss += criterion(decoder_output, target)

        if opts is None and scaler is not None:
            scaled_loss_temp = criterion(torch.tensor(scaler.inverse_transform(decoder_output.squeeze(0).cpu())),
                                         torch.tensor(scaler.inverse_transform(target.squeeze(0).cpu())))
            scaled_loss += scaled_loss_temp
            scaled_loss_over_time[t] = scaled_loss_temp

        if use_teacher_forcing:
            decoder_input = target
        else:
            if torch.all(torch.eq(decoder_output, EOS)):
                break
            decoder_input = decoder_output.detach()

    if opts is not None:
        loss.backward()

        encoder_opt.step()
        encoder_opt.zero_grad()

        decoder_opt.step()
        decoder_opt.zero_grad()
    else:
        scaled_loss = scaled_loss.item() / seq_length

    return loss.item() / seq_length, scaled_loss, scaled_loss_over_time


def fit(models, optims, epochs, dataloaders, criterion, scaler, device,
        teacher_forcing_ratio=0.0,
        update_learning_rates=None,
        use_attention=False,
        schedule_rate=1.0):

    train_dataloader, val_dataloader = dataloaders
    scaled_val_losses_over_time = None

    num_batches = len(train_dataloader)
    batch_size = len(train_dataloader.dataset[0])

    progress_bar = ProgressBar(num_batches, batch_size)

    plot_losses = []
    for epoch in range(epochs):
        losses = []
        total_time = 0

        if update_learning_rates is not None:
            optims = update_learning_rates(optims, epoch)

        print("Epoch", str(epoch + 1) + "/" + str(epochs))

        for index, data in enumerate(train_dataloader, 0):
            with Timer() as timer:
                loss, _, _ = loss_batch(data, models,
                                        optims, criterion, device,
                                        use_attention=use_attention)

                losses.append(loss)
            total_time += timer.interval
            progress_bar.print_progress_bar(index, total_time, loss)

        with torch.no_grad():
            val_losses, scaled_val_losses, scaled_val_losses_over_time = zip(
                *[loss_batch(data, models, None, criterion, device, use_attention=use_attention, scaler=scaler)
                  for _, data in enumerate(val_dataloader, 0)]
            )

        loss = np.sum(losses) / len(losses)
        val_loss = np.sum(val_losses) / len(val_losses)
        scaled_val_loss = np.sum(scaled_val_losses) / len(scaled_val_losses)

        if epoch == epochs - 1:
            scaled_val_losses_over_time = torch.stack(
                scaled_val_losses_over_time)
            scaled_val_loss_over_time = scaled_val_losses_over_time.mean(0)

        plot_losses.append((loss, val_loss))

        print()
        print("Training Loss: %.8f - Val Loss: %.8f - Scaled Val Loss: %.8f" %
              (loss, val_loss, scaled_val_loss))

        teacher_forcing_ratio *= schedule_rate

    showPlot(plot_losses, epochs)
    plotLossesOverTime(scaled_val_loss_over_time)
