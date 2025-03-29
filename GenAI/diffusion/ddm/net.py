#%% 
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-colorblind")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMAGE_SIZE = 64
NOISE_EMBEDDING_SIZE = 32
PLOT_DIFFUSION_STEPS = 20

def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=diffusion_times.dtype, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=diffusion_times.dtype, device=diffusion_times.device))
    
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    
    return noise_rates, signal_rates

def sinusoidal_embedding(x):
    """
    Generates sinusoidal embeddings for the input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Sinusoidal embeddings.
    """
    # 每个维度的频率
    frequencies = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), NOISE_EMBEDDING_SIZE // 2, device=x.device, dtype=x.dtype)
    )
    # 每个维度的角速度
    angular_speeds = 2.0 * math.pi * frequencies
    # sin 和 cosine 拼起来
    sin_emb = torch.sin(x * angular_speeds)
    cos_emb = torch.cos(x * angular_speeds)
    embeddings = torch.cat([sin_emb, cos_emb], dim=-1)
    return embeddings

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, x):  # x: (B, 1, 1, 1)
        x = x.view(x.size(0), 1)  # (B, 1)
        half_dim = self.embedding_size // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half_dim, device=x.device)
        )
        angles = x * freqs * 2 * math.pi  # (B, D)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, D)
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        emb = emb.expand(-1, -1, IMAGE_SIZE, IMAGE_SIZE)  # (B, D, H, W)
        return emb


# ========== ResidualBlock ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.proj(x)
        x = self.norm(x)
        x = F.silu(self.conv1(x))
        x = self.conv2(x)
        return x + residual

# ========== DownBlock ==========
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualBlock(in_channels, out_channels))
        for _ in range(block_depth - 1):
            self.blocks.append(ResidualBlock(out_channels, out_channels))
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        # 连续执行所有残差块
        for block in self.blocks:
            x = block(x)
        # 只在所有子块都完成后，append一次 skip
        skips.append(x)
        # 池化
        x = self.pool(x)
        return x, skips

# ========== UpBlock ==========
class UpBlock(nn.Module):
    def __init__(self, x_channels, skip_channels, out_channels, block_depth):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.blocks = nn.ModuleList()

        # 拼接后通道是 (x_channels + skip_channels)
        self.blocks.append(ResidualBlock(x_channels + skip_channels, out_channels))

        for _ in range(block_depth - 1):
            self.blocks.append(ResidualBlock(out_channels, out_channels))

    def forward(self, x, skips):
        x = self.upsample(x)
        # 每个 UpBlock 只 pop 一次 skip
        skip = skips.pop()
        x = torch.cat([x, skip], dim=1)
        # 依次通过残差块
        for block in self.blocks:
            x = block(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始卷积，将 (3 通道图像) -> 32 通道
        self.init_conv = nn.Conv2d(3, 32, kernel_size=1)
        # 噪声嵌入
        self.noise_embedding = SinusoidalEmbedding(NOISE_EMBEDDING_SIZE)

        # 下采样路径 (block_depth=2)
        self.down1 = DownBlock(32 + NOISE_EMBEDDING_SIZE, 32, block_depth=2)
        self.down2 = DownBlock(32, 64, block_depth=2)
        self.down3 = DownBlock(64, 96, block_depth=2)

        # Bottleneck
        self.res1 = ResidualBlock(96, 128)
        self.res2 = ResidualBlock(128, 128)

        # 上采样路径
        self.up1 = UpBlock(x_channels=128, skip_channels=96, out_channels=96, block_depth=2)
        self.up2 = UpBlock(x_channels=96, skip_channels=64, out_channels=64, block_depth=2)
        self.up3 = UpBlock(x_channels=64, skip_channels=32, out_channels=32, block_depth=2)

        # 输出层 (将通道数恢复到 3)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, noisy_images, noise_variances):
        """
        noisy_images  : (B, 3, H, W)
        noise_variances : (B, 1) 或 (B, 1, 1, 1)，表示噪声方差，可以根据需要传入
        """
        # 初始卷积
        x = self.init_conv(noisy_images)

        # 噪声嵌入，并拼接到通道维度
        noise_emb = self.noise_embedding(noise_variances)
        x = torch.cat([x, noise_emb], dim=1)  # (B, 32 + NOISE_EMBEDDING_SIZE, H, W)

        # 下采样 (并收集 skip)
        skips = []
        x, skips = self.down1(x, skips)  # out: 32
        x, skips = self.down2(x, skips)  # out: 64
        x, skips = self.down3(x, skips)  # out: 96

        # bottleneck
        x = self.res1(x)  # out: 128
        x = self.res2(x)  # out: 128

        # 上采样 (依次使用 skip)
        x = self.up1(x, skips)  # out: 96
        x = self.up2(x, skips)  # out: 64
        x = self.up3(x, skips)  # out: 32

        # 输出层
        x = self.out_conv(x)    # (B, 3, H, W)
        return x



class DiffusionModel(nn.Module):
    def __init__(self, unet_class, scheduler, EMA=0.999):
        super(DiffusionModel, self).__init__()
        self.normalizer = transforms.Normalize(mean=[0.5], std=[0.5])
        self.network = unet_class()
        self.ema_network = unet_class()
        self.diffusion_schedule = scheduler
        self.EMA = EMA

    def denormalize(self, images):
        mean = torch.tensor([0.5]).to(images.device)
        std = torch.tensor([0.5]).to(images.device)
        images = mean + images * std
        return torch.clamp(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network(noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps: int):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise # (B,C,H,W)
        for step in range(diffusion_steps):
            diffusion_times = torch.ones((num_images, 1, 1, 1)).to(initial_noise.device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images, 3, 64, 64)).to(next(self.parameters()).device)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images, optimizer, criterion):
        self.train()
        images = self.normalizer(images)
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1)).to(images.device)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * images + noise_rates * noises

        optimizer.zero_grad()
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        noise_loss = criterion(pred_noises, noises)
        noise_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for param, ema_param in zip(self.network.parameters(), self.ema_network.parameters()):
                ema_param.data.mul_(self.EMA).add_(param.data, alpha=1 - self.EMA)

        return noise_loss.item()

    def test_step(self, images, criterion):
        self.eval()
        with torch.no_grad():
            images = self.normalizer(images)
            noises = torch.randn_like(images).to(images.device)

            diffusion_times = torch.rand((images.size(0), 1, 1, 1)).to(images.device)
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            noisy_images = signal_rates * images + noise_rates * noises
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            noise_loss = criterion(pred_noises, noises)

        return noise_loss.item()