from torch import nn
import torch.nn.functional as F


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, **conv_params):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, **conv_params)

    def forward(self, x):
        return self.conv(F.leaky_relu(self.norm(x)))


class DoubleConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, **conv_params):
        super().__init__()
        intermediate_channels = min(in_channels, out_channels)  # for less memory consumption
        self.conv_block1 = ConvBlock3d(in_channels, intermediate_channels, **conv_params)
        self.conv_block2 = ConvBlock3d(intermediate_channels, out_channels, **conv_params)

    def forward(self, x):
        return self.conv_block2(self.conv_block1(x))


class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, **conv_params):
        super().__init__()
        self.double_conv_block = DoubleConvBlock3d(in_channels, out_channels, **conv_params)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        return self.double_conv_block(x) + self.skip(x)


class Normalize(nn.Module):
    def __init__(self, dim, p=2):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)
