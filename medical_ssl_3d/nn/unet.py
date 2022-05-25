import torch
from torch import nn
import torch.nn.functional as F

from .blocks import DoubleConvBlock3d, ResBlock3d
from .encoder import Encoder3D


class UNet3d(nn.Module):
    def __init__(self, in_channels, encoder_channels, decoder_channels, residual=False):
        super().__init__()

        assert len(encoder_channels) == len(decoder_channels)

        if residual:
            conv_block = lambda c_in, c_out: ResBlock3d(c_in, c_out, kernel_size=3, padding=1)
        else:
            conv_block = lambda c_in, c_out: DoubleConvBlock3d(c_in, c_out, kernel_size=3, padding=1)

        # encoder
        self.first_conv = nn.Conv3d(in_channels, encoder_channels[0], kernel_size=3, padding=1)
        self.encoder_blocks = nn.ModuleList([
            conv_block(in_c, out_c)
            for in_c, out_c in zip(encoder_channels, encoder_channels[1:])
        ])
        self.downsample = nn.MaxPool3d(kernel_size=2, ceil_mode=True)

        self.bridge = conv_block(encoder_channels[-1], decoder_channels[0])

        # decoder
        self.decoder_blocks = nn.ModuleList([
            conv_block(down_c + left_c, out_c)
            for down_c, left_c, out_c in zip(decoder_channels, encoder_channels[::-1], decoder_channels[1:])
        ])

    def forward(self, x):
        x = self.first_conv(x)

        encoder_fmaps = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_fmaps.insert(0, x)
            x = self.downsample(x)

        x = self.bridge(x)

        for block, fmap in zip(self.decoder_blocks, encoder_fmaps):
            x = F.interpolate(x, size=fmap.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat((x, fmap), dim=1)
            x = block(x)

        return x


class UNet3d_v2(nn.Module):
    def __init__(self, in_channels, encoder_channels, decoder_channels, residual=False) -> None:
        super().__init__()
        
        assert len(encoder_channels) == len(decoder_channels)

        self.encoder = Encoder3D(in_channels, encoder_channels, residual=residual)

        if residual:
            conv_block = lambda c_in, c_out: ResBlock3d(c_in, c_out, kernel_size=3, padding=1)
        else:
            conv_block = lambda c_in, c_out: DoubleConvBlock3d(c_in, c_out, kernel_size=3, padding=1

        self.bridge = conv_block(encoder_channels[-1], decoder_channels[0])

        self.decoder_blocks = nn.ModuleList([
            conv_block(down_c + left_c, out_c)
            for down_c, left_c, out_c in zip(decoder_channels, encoder_channels[::-1], decoder_channels[1:])
        ])

    @staticmethod
    def _merge(left, down):
        return torch.add(*layers.interpolate_to_left(left, down, 'trilinear'))

    def forward(self, x):
        x, encoder_levels_outputs = self.encoder(x, return_levels_outputs=True)
        x = self.bridge(x)
        for output, level in zip(encoder_levels_outputs, self.decoder_blocks):
            x = level(self._merge(output, self.upsample(x)))

        return x


    

