#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/27 8:19
# @Author : ZhangKuo
import torch
import pytest
import numpy as np
from u_net.u_net import ConvBlock, DownSample, UpSample, UNet


class TestConvBlock:
    def test_forward(self):
        conv_block = ConvBlock(3, 64)
        x = torch.randn(2, 3, 128, 128)
        assert conv_block(x).shape == (2, 64, 124, 124)
        conv_block = ConvBlock(64, 128)
        x = torch.randn(2, 64, 124, 124)
        assert conv_block(x).shape == (2, 128, 120, 120)


class TestDownSample:
    def test_forward(self):
        # b, c_in, h, w -> b, c_out, h//2-4, w//2-4
        down_sample = DownSample(3, 64)
        x = torch.randn(2, 3, 128, 128)
        assert down_sample(x).shape == (
            2,
            64,
            60,
            60,
        )  # 128//2 -2 -2，因为conv_block中有两个卷积层
        down_sample = DownSample(64, 128)
        x = torch.randn(2, 64, 60, 60)
        assert down_sample(x).shape == (2, 128, 26, 26)


class TestUpSample:
    def test_forward(self):
        # b, c_in, h, w -> b, c_out, h*2-4, w*2-4
        up_sample = UpSample(128, 64)
        x = torch.randn(2, 128, 26, 26)
        skip = torch.randn(2, 64, 60, 60)
        """
        x: 2, 128, 26, 26 -> 2, 64, 52, 52
        skip 2, 64, 60, 60 -> 2, 64, 52, 52
        x: 2, 64, 52, 52 -> 2, 128, 52, 52
        x: 2, 128, 52, 52 -> 2, 64, 48, 48
        """
        assert up_sample(x, skip).shape == (2, 64, 48, 48)
        up_sample = UpSample(64, 32)
        x = torch.randn(2, 64, 48, 48)
        skip = torch.randn(2, 32, 124, 124)
        """
        x: 2, 64, 48, 48 -> 2, 32, 96, 96
        skip 2, 32, 124, 124 -> 2, 32, 96, 96
        x: 2, 32, 96, 96 -> 2, 64, 96, 96
        x: 2, 64, 96, 96 -> 2, 32, 92, 92 
        """
        assert up_sample(x, skip).shape == (2, 32, 92, 92)


class TestUNet:
    def test_forward(self):
        unet = UNet(3, 1)
        x = torch.randn(2, 3, 256, 256)
        assert unet(x).shape == (2, 1, 68, 68)

