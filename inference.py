"""
Финальный инференс-скрипт строго по требованиям ТЗ хакатона:
1. Принимает 4 исходных изображения PNG (128x128, 8-bit, grayscale).
2. Генерирует фазовый ДОЭ и сохраняет его как 8-bit PNG: doe.png (-pi -> 0, +pi -> 255).
3. Восстанавливает 4 слоя методом дифракции Френеля: rec_1.png ... rec_4.png.
4. Вычисляет и выводит значения 2 метрик (Пирсон KK и SSIM ИСС).

Использование:
    python inference.py --img1 path1.png --img2 path2.png --img3 path3.png --img4 path4.png [--weights weights/best.pth]
"""

import argparse
import os
import numpy as np
from PIL import Image
import torch

from physics import (
    FresnelMultiplanePropagator,
    phase_to_image,
    reconstruct_layers_numpy,
    save_reconstructed_images,
    evaluate_all_layers,
    DISTANCES
)
from model import UNetDOE


def load_input_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((128, 128))
    return np.array(img, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Инференс нейросетевого синтеза ДОЭ под 4 плоскости")
    parser.add_argument("--img1", type=str, required=True, help="Путь к изображению слоя 1 (z = 24.1 мм)")
    parser.add_argument("--img2", type=str, required=True, help="Путь к изображению слоя 2 (z = 25.3 мм)")
    parser.add_argument("--img3", type=str, required=True, help="Путь к изображению слоя 3 (z = 26.5 мм)")
    parser.add_argument("--img4", type=str, required=True, help="Путь к изображению слоя 4 (z = 27.7 мм)")
    parser.add_argument("--weights", type=str, default=None, help="Путь к весам модели (.pth)")
    parser.add_argument("--out_dir", type=str, default="results", help="Папка для сохранения результатов")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Загрузка 4 входных изображений
    print("Загрузка 4 входных изображений...")
    inputs = [
        load_input_image(args.img1),
        load_input_image(args.img2),
        load_input_image(args.img3),
        load_input_image(args.img4),
    ]

    # Подготовка тензора для сети [1, 4, 128, 128]
    input_tensors = [torch.from_numpy(img).float() / 255.0 for img in inputs]
    input_batch = torch.stack(input_tensors, dim=0).unsqueeze(0)

    # 2. Инференс нейросети
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = UNetDOE(in_channels=4, out_channels=1, base_filters=16).to(device)

    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Загружены веса из {args.weights}")
    else:
        print("Внимание: веса не указаны, расчет выполняется на тестовой модели.")

    model.eval()
    with torch.no_grad():
        phase_tensor = model(input_batch.to(device))
        phase_doe = phase_tensor.squeeze().cpu().numpy()

    # 3. Сохранение ДОЭ по ТЗ (-pi -> 0, +pi -> 255)
    doe_path = os.path.join(args.out_dir, "doe.png")
    doe_img = phase_to_image(phase_doe)
    doe_img.save(doe_path)
    print(f"\n[РЕЗУЛЬТАТ 1] Голограмма ДОЭ сохранена: {doe_path}")

    # 4. Восстановление 4 изображений методом дифракции Френеля
    rec_layers = reconstruct_layers_numpy(phase_doe)
    for i, layer in enumerate(rec_layers, start=1):
        rec_path = os.path.join(args.out_dir, f"rec_{i}.png")
        Image.fromarray(layer, mode="L").save(rec_path)
    print(f"[РЕЗУЛЬТАТ 2] 4 восстановленных изображения сохранены в папку: {args.out_dir}")

    # 5. Вычисление метрик (Пирсон KK и SSIM ИСС)
    print("\n[РЕЗУЛЬТАТ 3] Метрики качества восстановления:")
    print("=" * 65)
    metrics = evaluate_all_layers(inputs, rec_layers)
    print("=" * 65)


if __name__ == "__main__":
    main()
