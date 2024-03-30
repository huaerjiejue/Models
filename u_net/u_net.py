#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/26 17:35
# @Author : ZhangKuo
import torch
import torch.nn as nn
from torchinfo import summary
from torchvision.transforms.functional import center_crop


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv_block(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.nn_conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, 2, stride=2
        )
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.nn_conv_transpose2d(x)
        skip = center_crop(skip, x.shape[-2:])
        x = torch.cat((x, skip), dim=1)
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=None):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if features is None:
            features = [64, 128, 256, 512, 1024]

        self.input = ConvBlock(in_channels, features[0])
        for i in range(len(features) - 1):
            self.encoder.append(DownSample(features[i], features[i + 1]))

        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(UpSample(features[i], features[i - 1]))

        self.output = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        x = self.input(x)
        skips = []
        for down in self.encoder:
            skips.append(x)
            x = down(x)

        for up, skip in zip(self.decoder, reversed(skips)):
            x = up(x, skip)

        return self.output(x)


if __name__ == "__main__":
    model = UNet(3, 1)
    summary(model, input_size=(1, 3, 256, 256), device="cpu")
