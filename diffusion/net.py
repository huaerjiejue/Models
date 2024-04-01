#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/30 14:38
# @Author : ZhangKuo
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torchinfo import summary
from torchvision.transforms.functional import center_crop
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
        attn_out = self.norm(attn_out + x)  # residual connection
        attn_out = rearrange(attn_out, "n (h w) c -> n c h w", h=h, w=w)
        return attn_out


class ConvBlockWithT(nn.Module):
    # b, c_in, h, w -> b, c_out, h-2, w-2
    # 用于unet的decoder部分
    def __init__(
        self, in_channels, out_channels, heads, kernel_size=3, stride=1, padding=1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.self_attn = SelfAttention(out_channels, heads)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding=0
        )
        self.gelu = nn.GELU()

    def forward(self, x, t):
        assert (
            x.shape == t.shape
        ), f"The shape of x({x.shape}) and t({t.shape}) must be the same"
        x = self.conv1(x)
        t = repeat(t, "b c h w -> b (c c_new) h w", c_new=x.shape[1] // t.shape[1])
        x += t
        x = self.self_attn(x)
        x = self.conv2(x)
        x = self.gelu(x)
        b, c, h, w = x.size()
        x = nn.LayerNorm([c, h, w])(x)
        return x


class ConvBlock(nn.Module):
    # b, c_in, h, w -> b, c_out, h-2, w-2
    # 用于unet的encoder部分
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=0),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.size()
        x = nn.LayerNorm([c, h, w])(x)
        return x


class Up(nn.Module):
    # b, c_in, h, w + b, c_out, 2h, 2w -> b, c_out, 2h-2, 2w-2
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up_conv(x)
        skip = center_crop(skip, x.shape[-2:])
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class Down(nn.Module):
    # b, c_in, h, w -> b, c_out, h//2-2, w//2-2
    def __init__(self, in_channels, out_channels, heads=8):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv = ConvBlockWithT(in_channels, out_channels, heads)

    def forward(self, x, t):
        x = self.max_pool(x)
        x = self.conv(x, t)
        return x


class Net(nn.Module):
    def __init__(self,  features=None):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 3
        self.features = features
        if self.features is None:
            self.features = [64, 128, 256, 512, 1024]
        self.heads = [feature//8 for feature in self.features]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skips = []
        self.input = ConvBlock(self.in_channels, self.features[0])
        for i in range(len(self.features) - 1):
            self.encoder.append(
                Down(self.features[i], self.features[i + 1], self.heads[i])
            )
        for i in range(len(self.features) - 1, 0, -1):
            self.decoder.append(
                Up(self.features[i], self.features[i - 1], self.heads[i])
            )

    def forward(self, x, t):
        x = self.input(x)
        for i, down in enumerate(self.encoder):
            b, c, h, w = x.size()
            t = repeat(t, "1 -> b, c, h, w", b=b, c=c, h=h, w=w)
            x = down(x, t)
            self.skips.append(x)
        self.skips = self.skips[::-1]
        for i, up in enumerate(self.decoder):
            x = up(x, self.skips[i])
        return x

if __name__ == "__main__":
    model = Net()
    summary(model, [(3, 256, 256), 1])
