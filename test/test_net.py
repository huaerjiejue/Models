#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/30 22:06
# @Author : ZhangKuo
import torch
import torch.nn as nn

from torchinfo import summary

from diffusion.net import SelfAttention, ConvBlock


class TestSelfAttention:
    def test_forward(self):
        self_attention = SelfAttention(64, 8)
        x = torch.randn(2, 64, 8, 8)
        assert self_attention(x).shape == (2, 64, 8, 8)
        self_attention = SelfAttention(128, 16)
        x = torch.randn(2, 128, 16, 16)
        assert self_attention(x).shape == (2, 128, 16, 16)


class TestConvBlock:
    def test_forward(self):
        conv_block = ConvBlock(64, 128, 8)
        x = torch.randn(2, 64, 8, 8)
        t = torch.randn(2, 64, 8, 8)
        assert conv_block(x, t).shape == (2, 128, 8, 8)
        conv_block = ConvBlock(128, 256, 16)
        x = torch.randn(2, 128, 16, 16)
        t = torch.randn(2, 128, 16, 16)
        assert conv_block(x, t).shape == (2, 256, 16, 16)

