import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import prepare_batch

"""a UNet architecture based on the original paper's implementation"""
"""added batchnorm though"""

class Unet_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet_ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Unet_DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet_DownSample, self).__init__()
        self.downsample = nn.MaxPool2d(2)
        self.ConvBlock = Unet_ConvBlock(in_channels, out_channels)

    def forward(self, x):
        y = self.downsample(x)
        return self.ConvBlock(y)

class Unet_UpSample(nn.Module):
    """using deconvolution here even though bilinear interpolation might be a bit better"""
    def __init__(self, in_channels, out_channels):
        super(Unet_UpSample, self).__init__()
        """out channels of the conv2dtranspose here is half number of in channels because we are concating relevant decoder layer"""
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.ConvBlock = Unet_ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        """format is batch_size, channels, h, w"""
        x = torch.cat([x2, x1], dim=1)
        return self.ConvBlock(x)

class Unet(nn.Module):
    def __init__(self, n_classes):
        super(Unet, self).__init__()
        self.n_classes = n_classes

        #self.inc = self.Unet_ConvBlock(3, 64)

        self.inc = Unet_ConvBlock(3, 64)
        self.down1 = Unet_DownSample(64, 128)
        self.down2 = Unet_DownSample(128,256)
        self.down3 = Unet_DownSample(256, 512)
        self.down4 = Unet_DownSample(512, 1024)

        self.up1 = Unet_UpSample(1024, 512)
        self.up2 = Unet_UpSample(512, 256)
        self.up3 = Unet_UpSample(256, 128)
        self.up4 = Unet_UpSample(128, 64)

        self.out = nn.Conv2d(64, self.n_classes, kernel_size=3, padding=1, bias=False)

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
        logits = self.out(x)
        return logits