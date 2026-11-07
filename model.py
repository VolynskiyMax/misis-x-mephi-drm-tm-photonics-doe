"""
Архитектура нейросети U-Net для генерации фазового ДОЭ из 4 изображений.
Разработана для Человека 2 (Специалист по нейросетям).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from physics import FresnelMultiplanePropagator


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetDOE(nn.Module):
    """
    U-Net генератор фазового ДОЭ.
    - Вход:  [Batch, 4, 128, 128] — 4 целевых изображения (слои 3D сцены)
    - Выход: [Batch, 1, 128, 128] — фаза ДОЭ в радианах от -pi до +pi
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_filters: int = 32):
        super().__init__()
        f = base_filters
        
        # Энкодер (сжатие)
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))

        # Боттлнек
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))

        # Декодер (восстановление)
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(f * 2, f)

        # Выходной слой
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bot(x4)

        u1 = self.up1(xb)
        u1 = self.conv1(torch.cat([u1, x4], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv2(torch.cat([u2, x3], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv3(torch.cat([u3, x2], dim=1))

        u4 = self.up4(u3)
        u4 = self.conv4(torch.cat([u4, x1], dim=1))

        # Tanh дает диапазон [-1, 1], домножаем на pi -> [-pi, pi]
        phase = self.act(self.outc(u4)) * torch.pi
        return phase


class PhysicsInformedLoss(nn.Module):
    """
    Комплексная функция потерь для обучения ДОЭ:
    Loss = alpha * MSE + beta * (1 - PCC)
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred_layers: torch.Tensor, target_layers: torch.Tensor) -> torch.Tensor:
        # pred_layers, target_layers: [Batch, 4, 128, 128] в диапазоне [0, 1]
        loss_mse = self.mse(pred_layers, target_layers)

        # Корреляция Пирсона по батчу и каналам
        p_flat = pred_layers.flatten(start_dim=2)   # [B, 4, 128*128]
        t_flat = target_layers.flatten(start_dim=2) # [B, 4, 128*128]

        p_diff = p_flat - p_flat.mean(dim=-1, keepdim=True)
        t_diff = t_flat - t_flat.mean(dim=-1, keepdim=True)

        num = (p_diff * t_diff).sum(dim=-1)
        den = torch.sqrt((p_diff**2).sum(dim=-1) * (t_diff**2).sum(dim=-1) + 1e-8)
        pcc = (num / den).mean()

        loss_pcc = 1.0 - pcc
        total_loss = self.alpha * loss_mse + self.beta * loss_pcc
        return total_loss, loss_mse, pcc
