import torch
from torch import nn
import torch.nn.functional as F

from .blocks import DoubleConvBlock3d, ResBlock3d


class Encoder3D(nn.Module):
    def __init__(self, in_channels, encoder_channels, residual=False):
        """Encoder for UNet.

        Args:
            downsample (int): how many times the max pooling is applied.
        """
        super().__init__()

        if residual:
            conv_block = lambda c_in, c_out: ResBlock3d(c_in, c_out, kernel_size=3, padding=1)
        else:
            conv_block = lambda c_in, c_out: DoubleConvBlock3d(c_in, c_out, kernel_size=3, padding=1)

        self.first_conv = nn.Conv3d(in_channels, encoder_channels[0], kernel_size=3, padding=1)
        self.encoder_blocks = nn.ModuleList([
            conv_block(in_c, out_c)
            for in_c, out_c in zip(encoder_channels, encoder_channels[1:])
        ])
        self.downsample = nn.MaxPool3d(kernel_size=2, ceil_mode=True)
        self.out_channels = encoder_channels[-1]

    def forward(self, x, return_encoder_fmaps=False):
        x = self.first_conv(x)

        encoder_fmaps = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_fmaps.insert(0, x)
            x = self.downsample(x)

        if return_encoder_fmaps:
            return x, encoder_fmaps

        return x