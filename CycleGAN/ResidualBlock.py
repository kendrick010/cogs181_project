import torch
import torch.nn as nn
from . import ConvolutionalBlock

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(ConvolutionalBlock(channels, channels, kernel_size=3, padding=1),
                                   ConvolutionalBlock(channels, channels, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.block(x)