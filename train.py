"""
Скрипт быстрого запуска обучения (Proof-of-Concept) на синтетических данных.
Запускается сразу из коробки: python train.py
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from physics import FresnelMultiplanePropagator, phase_to_image, evaluate_all_layers
from model import UNetDOE, PhysicsInformedLoss


# Простой генератор синтетических четверок (круги, полосы, прямоугольники)
class SyntheticPhotonicsDataset(Dataset):
    def __init__(self, size: int = 200, grid_size: int = 128):
        self.size = size
        self.grid_size = grid_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Генерируем 4 случайных контрастных слоя
        layers = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(4):
            cx, cy = np.random.randint(30, 98, size=2)
            r = np.random.randint(15, 30)
            y, x = np.ogrid[:self.grid_size, :self.grid_size]
            shape_type = np.random.randint(0, 3)
            if shape_type == 0:  # круг
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                layers[i, mask] = 1.0
            elif shape_type == 1:  # прямоугольник
                w, h = np.random.randint(10, 30, size=2)
                layers[i, max(0, cy-h):min(128, cy+h), max(0, cx-w):min(128, cx+w)] = 1.0
            else:  # крест
                layers[i, cy-4:cy+4, :] = 1.0
                layers[i, :, cx-4:cx+4] = 1.0
        return torch.from_numpy(layers)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Используемое устройство: {device}")

    # Инициализация моделей
    model = UNetDOE(in_channels=4, out_channels=1, base_filters=16).to(device)
    propagator = FresnelMultiplanePropagator().to(device)
    criterion = PhysicsInformedLoss(alpha=1.0, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = SyntheticPhotonicsDataset(size=100)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("Старт тестового обучения (10 эпох)...")
    model.train()
    for epoch in range(1, 11):
        epoch_loss = 0.0
        epoch_pcc = 0.0
        for batch in loader:
            batch = batch.to(device) # [B, 4, 128, 128]
            optimizer.zero_grad()

            # 1. Нейросеть предсказывает фазу одного ДОЭ
            phase_doe = model(batch) # [B, 1, 128, 128]

            # 2. Физический слой восстанавливает 4 слоя
            rec_layers = propagator(phase_doe) # [B, 4, 128, 128]

            # 3. Расчет функции потерь
            loss, mse, pcc = criterion(rec_layers, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pcc += pcc.item()

        print(f"Эпоха {epoch:2d}/10 | Loss: {epoch_loss/len(loader):.4f} | Средний PCC: {epoch_pcc/len(loader):.4f}")

    # Сохраняем веса
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/unet_doe_poc.pth")
    print("Веса модели сохранены в weights/unet_doe_poc.pth")


if __name__ == "__main__":
    main()
