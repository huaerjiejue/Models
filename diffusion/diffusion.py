#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/30 14:38
# @Author : ZhangKuo
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader


class Diffusion:
    def __init__(self, model, datafile_name, num_steps):
        self.model = model
        self.datafile_name = datafile_name
        self.num_steps = num_steps
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.betas = torch.linspace(1e-4, 2e-3, self.num_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        self.imgs = self.read_file()

    def read_file(self):
        imgs = [
            os.path.join(self.datafile_name, img)
            for img in os.listdir(self.datafile_name)
        ]
        return imgs

    def show_img(self, imgs):
        len_imgs = len(imgs)
        fig, axs = plt.subplots(1, len_imgs, figsize=(20, 20))
        for i in range(len_imgs):
            img = plt.imread(imgs[i])
            axs[i].imshow(img)
            axs[i].axis("off")

    def q_x(self, x_0, t, noise):
        assert 0 <= t < self.num_steps
        # noise = torch.randn_like(x_0).to(self.device)
        alphas_cumprod = self.alphas_cumprod[t]
        one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt[t]
        x_t = alphas_cumprod * x_0 + one_minus_alphas_bar_sqrt * noise
        return x_t

    def train(self, epochs=1000, batch_size=32):
        dataset = load_dataset(self.datafile_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                t = torch.randint(0, self.num_steps, (1, batch_size)).to(self.device)
                x_0 = data.to(self.device)
                noise = torch.randn_like(x_0).to(self.device)
                x_t = self.q_x(x_0, t, noise)
                pred = self.model(x_t)
                loss = self.criterion(pred, t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
        print("Training Finished!")

    def p_sample(self, x_t, t):
        pred_noise = self.model(x_t)
        pure_noise = torch.randn_like(x_t).to(self.device)
        x_t_prev = (
            1
            / self.alphas[t]
            * (x_t - self.betas[t] * pred_noise / self.one_minus_alphas_bar_sqrt[t])
            + torch.sqrt(self.betas[t]) * pure_noise
        )
        x_t_prev = (x_t_prev.clamp(0, 1) * 255).to(torch.uint8)
        return x_t_prev

    def test(self, x_t, ts):
        for t in ts:
            x_t = self.p_sample(x_t, t)
            plt.imshow(x_t[0].cpu().numpy().transpose(1, 2, 0))
            plt.axis("off")
            plt.show()
