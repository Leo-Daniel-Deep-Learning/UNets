"""
Reference: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from unet_mods import DoubleConv, Down, Up, OutConv

import numpy as np
import random, time, os


class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        if bilinear:
            factor = 2
        else:
            factor = 1

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def bce_dice_loss(self, x, y, smooth=1e-5):
        x = x.view(-1)
        y = y.view(-1)

        intersection = (x * y).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (x.sum() + y.sum() + smooth)

        BCE = F.binary_cross_entropy(x, y, reduction='mean')

        return 0.5 * BCE + 1 * dice_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.bce_dice_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

