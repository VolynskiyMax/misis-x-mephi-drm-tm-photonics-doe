import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. НАСТРОЙКИ, УСТРОЙСТВО И ФИЗИЧЕСКИЕ КОНСТАНТЫ (ПО ТЗ МИФИ)
# ==============================================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Ускорение на Apple Silicon (Mac)
else:
    DEVICE = torch.device("cpu")

N = 128
WAVELENGTH = 532e-9        # 532 нм
PIXEL_SIZE = 10e-6         # 10 мкм
DISTANCES = [24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3]  # z1, z2, z3, z4 в метрах


# ==============================================================================
# 2. АРХИТЕКТУРА НЕЙРОСЕТИ (СТРОГО СООТВЕТСТВУЕТ ВЕСАМ mephi_doe_model.pth)
# ==============================================================================
class DOE_GeneratorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.block(4, 64)
        self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256)
        self.bottleneck = self.block(256, 256)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = self.block(256 + 256, 128)
        self.dec2 = self.block(128 + 128, 64)
        self.dec1 = self.block(64 + 64, 32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        # Выход в радианах [-pi, pi]
        return torch.tanh(self.out_conv(d1)) * np.pi


# ==============================================================================
# 3. ФИЗИЧЕСКИЙ СЛОЙ ДИФРАКЦИИ ФРЕНЕЛЯ (1-FFT)
# ==============================================================================
class WinningFresnel(nn.Module):
    def __init__(self, N, wavelength, pixel_size, distances):
        super().__init__()
        coords = (torch.arange(N, dtype=torch.float32) - N // 2) * pixel_size
        U, V = torch.meshgrid(coords, coords, indexing="ij")
        u2_v2 = U**2 + V**2
        chirps = [
            torch.exp(1j * (np.pi / (wavelength * z)) * u2_v2)
            for z in distances
        ]
        self.register_buffer("chirps", torch.stack(chirps, dim=0))

    def forward(self, phase_doe, return_intensity=False):
        """
        Прямой расчет дифракции Френеля.
        По ТЗ хакатона: сохраняются МОДУЛИ значений (|K|), а не интенсивность (|K|^2).
        """
        H = torch.exp(1j * phase_doe)
        field_inside = H * self.chirps.unsqueeze(0)
        field_shifted = torch.fft.ifftshift(field_inside, dim=(-2, -1))
        K = torch.fft.fft2(field_shifted, dim=(-2, -1))
        K = torch.fft.fftshift(K, dim=(-2, -1))

        amp = torch.abs(K)

        # Робастная нормализация (отсечение выбросов спеклов на 99.8% квантиле)
        flat = amp.view(amp.shape[0], 4, -1)
        max_val = torch.quantile(flat, 0.998, dim=-1).view(-1, 4, 1, 1) + 1e-8
        norm_amp = torch.clamp(amp / max_val, 0.0, 1.0)

        if return_intensity:
            return norm_amp**2
        return norm_amp


# ==============================================================================
# 4. МЕТРИКИ (СТРОГО ПО ФОРМУЛАМ ИЗ СЛАЙДА 3 БЕЗ ВНЕШНИХ ЗАВИСИМОСТЕЙ)
# ==============================================================================
def ssim_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Индекс структурного сходства (ИСС / SSIM) строго по формуле из презентации:
    ИСС = [ (2*<A>*<B> + c1) * (2*sigma_AB + c2) ] /
          [ (<A>^2 + <B>^2 + c1) * (sigma_A^2 + sigma_B^2 + c2) ]
    """
    # Значения в диапазоне [0, 1]
    L = 1.0
    c1 = (0.01 * L)**2
    c2 = (0.03 * L)**2

    mu_A = A.mean(dim=(-2, -1), keepdim=True)
    mu_B = B.mean(dim=(-2, -1), keepdim=True)

    var_A = ((A - mu_A)**2).mean(dim=(-2, -1), keepdim=True)
    var_B = ((B - mu_B)**2).mean(dim=(-2, -1), keepdim=True)
    cov_AB = ((A - mu_A) * (B - mu_B)).mean(dim=(-2, -1), keepdim=True)

    num = (2.0 * mu_A * mu_B + c1) * (2.0 * cov_AB + c2)
    den = (mu_A**2 + mu_B**2 + c1) * (var_A + var_B + c2)
    iss = (num / (den + 1e-12)).mean()
    return iss


def calc_metrics_and_loss(reconstructed: torch.Tensor, target: torch.Tensor):
    """
    Вычисление коэффициента корреляции Пирсона (КК) и ИСС (SSIM).
    """
    mean_A = reconstructed.mean(dim=(-2, -1), keepdim=True)
    mean_B = target.mean(dim=(-2, -1), keepdim=True)

    Am = reconstructed - mean_A
    Bm = target - mean_B

    num = (Am * Bm).sum(dim=(-2, -1))
    den = torch.sqrt((Am**2).sum(dim=(-2, -1)) * (Bm**2).sum(dim=(-2, -1))) + 1e-8
    kk = (num / den).mean()

    iss = ssim_torch(reconstructed, target)
    loss = 4.0 * (1.0 - kk) + 1.2 * (1.0 - iss)
    return loss, kk.item(), iss.item()


# ==============================================================================
# 5. ОСНОВНАЯ ФУНКЦИЯ ВЫЧИСЛЕНИЯ И СОХРАНЕНИЯ
# ==============================================================================
def run_pipeline(
    weights_path: str,
    four_image_paths: list,
    output_folder: str = "jury_result",
    steps: int = 1200,
    return_intensity: bool = False
):
    start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)

    # 1. Загрузка обученной нейросети
    print(f"Используемое устройство: {DEVICE}")
    model = DOE_GeneratorNet().to(DEVICE)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print("✅ Обученная модель успешно загружена!")

    # 2. Чтение 4 входных картинок
    imgs = []
    print("Чтение входных изображений:")
    for i, p in enumerate(four_image_paths, start=1):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Файл изображения {p} не найден!")
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"Не удалось открыть изображение: {p}")
        im = cv2.resize(im, (128, 128))
        imgs.append(im.astype(np.float32) / 255.0)
        print(f"  [{i}/4] {p} (128x128)")

    target_imgs = torch.tensor(np.stack(imgs, axis=0)).unsqueeze(0).to(DEVICE)

    # 3. Инференс нейросети (мгновенное получение фазы ДОЭ)
    prop = WinningFresnel(N, WAVELENGTH, PIXEL_SIZE, DISTANCES).to(DEVICE)

    with torch.no_grad():
        # Чистый прогноз нейросети без шума
        init_phase = model(target_imgs)
        raw_rec = prop(init_phase, return_intensity=return_intensity)
        _, raw_kk, raw_iss = calc_metrics_and_loss(raw_rec, target_imgs)
        print(f"⚡ Чистый вывод нейросети (без итераций) -> КК: {raw_kk:.4f} | ИСС: {raw_iss:.4f}")

    # 4. Если steps > 0: тонкая доводка для максимизации метрик
    if steps > 0:
        print(f"⏳ Дополнительная оптимизация фазы ({steps} шагов Adam)...")
        editable_phase = nn.Parameter(init_phase.clone())
        optimizer = torch.optim.Adam([editable_phase], lr=0.09)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.005)

        fg = (target_imgs >= 0.05).float()
        bg = (target_imgs < 0.05).float()
        bg_frac = bg.mean().item()

        w_bg = 2.0 + 8.0 * (bg_frac**3)
        w_ssim = 1.5 + 2.5 * (bg_frac**3)
        w_kk = 3.5

        for step in range(steps):
            optimizer.zero_grad()
            rec = prop(editable_phase, return_intensity=return_intensity)

            Am = rec - rec.mean(dim=(-2, -1), keepdim=True)
            Bm = target_imgs - target_imgs.mean(dim=(-2, -1), keepdim=True)
            kk = ((Am * Bm).sum(dim=(-2, -1)) / (torch.sqrt((Am**2).sum(dim=(-2, -1)) * (Bm**2).sum(dim=(-2, -1))) + 1e-8)).mean()
            iss = ssim_torch(rec, target_imgs)

            fg_loss = (((rec - target_imgs) * fg)**2).mean()
            bg_loss = ((rec * bg)**2).mean()

            total_loss = w_kk * (1.0 - kk) + w_ssim * (1.0 - iss) + 2.0 * fg_loss + w_bg * bg_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        final_phase_tensor = editable_phase.detach()
    else:
        final_phase_tensor = init_phase.detach()

    # Приводим фазу строго к диапазону [-pi, pi]
    final_phase = torch.remainder(final_phase_tensor + np.pi, 2 * np.pi) - np.pi

    with torch.no_grad():
        final_rec = prop(final_phase, return_intensity=return_intensity)
        _, final_kk, final_iss = calc_metrics_and_loss(final_rec, target_imgs)

    # 5. Сохранение ДОЭ по ТЗ (-pi -> 0, +pi -> 255)
    doe_np = final_phase.cpu().squeeze().numpy()
    doe_uint8 = np.clip(np.round((doe_np + np.pi) / (2.0 * np.pi) * 255.0), 0, 255).astype(np.uint8)

    doe_path = os.path.join(output_folder, "doe.png")
    cv2.imwrite(doe_path, doe_uint8)
    cv2.imwrite(os.path.join(output_folder, "DOE.png"), doe_uint8) # дублируем для надежности

    # 6. Сохранение 4 восстановленных слоев (модули значений по ТЗ)
    rec_np = final_rec.cpu().squeeze().numpy()
    layer_metrics = []
    print("\n" + "="*65)
    print("📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ ВОССТАНОВЛЕННЫХ СЛОЕВ С ОРИГИНАЛАМИ:")
    print("="*65)
    z_mms = [24.1, 25.3, 26.5, 27.7]
    for i in range(4):
        rec_uint8 = np.clip(np.round(rec_np[i] * 255.0), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f"rec_{i+1}.png"), rec_uint8)
        
        # Индивидуальные метрики для плоскости
        t_i = target_imgs[:, i:i+1]
        r_i = final_rec[:, i:i+1]
        _, kk_i, iss_i = calc_metrics_and_loss(r_i, t_i)
        layer_metrics.append((kk_i, iss_i))
        print(f"  • Слой {i+1} (z = {z_mms[i]} мм): SSIM (ИСС) = {iss_i:.4f} | Pearson (КК) = {kk_i:.4f}")

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"ИТОГОВОЕ СРЕДНЕЕ ПО 4 СЛОЯМ:")
    print(f"  ⭐ SSIM (Индекс структурного сходства):  {final_iss:.4f}")
    print(f"  ⭐ Pearson (Коэффициент корреляции):     {final_kk:.4f}")
    print(f"⏱️ ВРЕМЯ РАБОТЫ МОДЕЛИ:                    {total_time:.3f} сек")
    print("="*65)
    print(f"🎉 Голограмма сохранена: {doe_path}")
    print(f"📁 Все 4 восстановленных слоя сохранены в: '{output_folder}'\n")
    return final_kk, final_iss


# ==============================================================================
# 6. ТОЧКА ВХОДА (CLI АРГУМЕНТЫ ИЛИ ДЕФОЛТ)
# ==============================================================================
if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description="Синтез фазового ДОЭ на основе обученной нейросети")
    parser.add_argument("images", nargs="*", default=None, help="Пути к 4 изображениям (позиционные аргументы)")
    parser.add_argument("--weights", type=str, default="mephi_doe_model.pth", help="Путь к весам .pth")
    parser.add_argument("--input_dir", type=str, default=None, help="Папка с 4 входными изображениями комплекта")
    parser.add_argument("--img1", type=str, default=None, help="Изображение 1 (z = 24.1 мм)")
    parser.add_argument("--img2", type=str, default=None, help="Изображение 2 (z = 25.3 мм)")
    parser.add_argument("--img3", type=str, default=None, help="Изображение 3 (z = 26.5 мм)")
    parser.add_argument("--img4", type=str, default=None, help="Изображение 4 (z = 27.7 мм)")
    parser.add_argument("--steps", type=int, default=1200, help="Количество шагов оптимизации (0 = мгновенный вывод 5 мс, 1200 = максимальное качество за 3-4 сек)")
    parser.add_argument("--out", type=str, default="output", help="Папка для сохранения выходных файлов (doe.png и rec_*.png)")
    parser.add_argument("--intensity", action="store_true", help="Возвращать интенсивность (|K|²) вместо модуля")
    args = parser.parse_args()

    # Определение путей к 4 картинкам комплекта
    jury_images = None
    
    # 1. Если передана папка с комплектом (--input_dir)
    if args.input_dir and os.path.exists(args.input_dir):
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        found = sorted([
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if f.lower().endswith(valid_exts) and not f.startswith("doe") and not f.startswith("rec")
        ])
        if len(found) >= 4:
            jury_images = found[:4]

    # 2. Если переданы позиционные аргументы (например: python run.py 1.png 2.png 3.png 4.png)
    if not jury_images and args.images and len(args.images) >= 4:
        jury_images = args.images[:4]

    # 3. Если переданы именованные флаги --img1 .. --img4
    if not jury_images and args.img1 and args.img2 and args.img3 and args.img4:
        jury_images = [args.img1, args.img2, args.img3, args.img4]

    # 4. Если в текущей папке есть 1.png .. 4.png
    if not jury_images and os.path.exists("1.png") and os.path.exists("2.png") and os.path.exists("3.png") and os.path.exists("4.png"):
        jury_images = ["1.png", "2.png", "3.png", "4.png"]

    # 5. Автопоиск любых 4 изображений в текущей папке
    if not jury_images:
        cur_pngs = sorted([
            f for f in os.listdir(".") if f.endswith(".png") 
            and not f.startswith("doe") and not f.startswith("rec") and not f.startswith("DOE")
        ])
        if len(cur_pngs) >= 4:
            jury_images = cur_pngs[:4]
            print(f"Автоматически выбраны файлы комплекта: {jury_images}")
        else:
            jury_images = ["1.png", "2.png", "3.png", "4.png"]

    # Проверка пути к весам
    weights_file = args.weights
    if not os.path.exists(weights_file):
        alt_path = os.path.join(os.path.dirname(__file__), "mephi_doe_model.pth")
        if os.path.exists(alt_path):
            weights_file = alt_path

    run_pipeline(
        weights_path=weights_file,
        four_image_paths=jury_images,
        output_folder=args.out,
        steps=args.steps,
        return_intensity=args.intensity
    )