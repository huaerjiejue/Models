#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/30 22:06
# @Author : ZhangKuo
import torch
import torch.nn as nn

from torchinfo import summary

from diffusion.net import SelfAttention, ConvBlockWithT, ConvBlock, Up, Down


class TestSelfAttention:
    def test_forward(self):
        self_attention = SelfAttention(64, 8)
        x = torch.randn(2, 64, 8, 8)
        assert self_attention(x).shape == (2, 64, 8, 8)
        self_attention = SelfAttention(128, 16)
        x = torch.randn(2, 128, 16, 16)
        assert self_attention(x).shape == (2, 128, 16, 16)


class TestConvBlockWithT:
    def test_forward(self):
        conv_block = ConvBlockWithT(64, 128, 8)
        x = torch.randn(2, 64, 10, 10)
        t = torch.randn(2, 64, 10, 10)
        assert conv_block(x, t).shape == (2, 128, 8, 8)
        conv_block = ConvBlockWithT(128, 256, 16)
        x = torch.randn(2, 128, 18, 18)
        t = torch.randn(2, 128, 18, 18)
        assert conv_block(x, t).shape == (2, 256, 16, 16)


class TestConvBlock:
    def test_forward(self):
        conv_block = ConvBlock(64, 32)
        x = torch.randn(2, 64, 10, 10)
        assert conv_block(x).shape == (2, 32, 8, 8)
        conv_block = ConvBlock(128, 64)
        x = torch.randn(2, 128, 18, 18)
        assert conv_block(x).shape == (2, 64, 16, 16)


class TestUp:
    def test_forward(self):
        up = Up(64, 32)
        x = torch.randn(2, 64, 10, 10)
        skip = torch.randn(2, 32, 20, 20)
        assert up(x, skip).shape == (2, 32, 18, 18)
        up = Up(128, 64)
        x = torch.randn(2, 128, 18, 18)
        skip = torch.randn(2, 64, 36, 36)
        assert up(x, skip).shape == (2, 64, 34, 34)


class TestDown:
    def test_forward(self):
        down = Down(128, 256)
        x = torch.randn(2, 128, 18, 18)
        t = torch.randn(2, 128, 9, 9)
        assert down(x, t).shape == (2, 256, 7, 7)
        down = Down(64, 128)
        x = torch.randn(2, 64, 10, 10)
        t = torch.randn(2, 64, 5, 5)
        assert down(x, t).shape == (2, 128, 3, 3)

