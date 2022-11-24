import torch
import torch.nn as nn
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
