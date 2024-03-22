import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_downsampling=True, **kwargs):
        super().__init__()
        
        if is_downsampling:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs), 
                                      nn.InstanceNorm2d(out_channels),
                                      nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                                      nn.InstanceNorm2d(out_channels),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)