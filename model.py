import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = x + residual

        return x


class KLABaseline(nn.Module):

    def __init__(self):

        super().__init__()

        self.head = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1
        )

        self.residual1 = ResidualBlock(64)
        self.residual2 = ResidualBlock(64)
        self.residual3 = ResidualBlock(64)
        self.residual4 = ResidualBlock(64)

        self.upsample = nn.Conv2d(
            64,
            64 * 4,
            kernel_size=3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.tail = nn.Conv2d(
            64,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        x = self.head(x)

        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)

        x = self.upsample(x)

        x = self.pixel_shuffle(x)

        x = self.tail(x)

        return x