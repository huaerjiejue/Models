#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/30 14:38
# @Author : ZhangKuo
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torchinfo import summary
from einops import rearrange, repeat


class SelfAttention(nn.Module):
    # b, c, h, w -> b, c, h, w
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        self.multi = MultiheadAttention(embed_size, heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        # x.shape = (batch_size, channels, height, width)
        n, c, h, w = x.size()
        x = rearrange(x, "n c h w -> n (h w) c", h=h, w=w)
        attn_out, _ = self.multi(x, x, x, attn_mask=mask)
        attn_out = self.norm(attn_out + x) # residual connection
        attn_out = rearrange(attn_out, "n (h w) c -> n c h w", h=h, w=w)
        return attn_out


class ConvBlock(nn.Module):
    # b, c_in, h, w -> b, c_out, h, w
    def __init__(self, in_channels, out_channels, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.self_attn = SelfAttention(out_channels, dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm([out_channels, dim, dim])

    def forward(self, x, t):
        assert x.shape == t.shape, f"The shape of x({x.shape}) and t({t.shape}) must be the same"
        x = self.conv1(x)
        t = repeat(t, "b c h w -> b (c c_new) h w", c_new=x.shape[1] // t.shape[1])
        x += t
        x = self.self_attn(x)
        x = self.conv2(x)
        x = self.gelu(x)
        return self.layer_norm(x)


if __name__ == "__main__":
    pass
