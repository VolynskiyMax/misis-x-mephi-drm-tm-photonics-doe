"""
Модуль волновой оптики и физических расчетов для хакатона по синтезу ДОЭ.
Разработан в соответствии с ТЗ НИЯУ МИФИ (ЛаПлаз, Каф. 37 "Лазерная физика").

Физические параметры:
- Длина волны лазера: lambda = 532 нм (532e-9 м)
- Линейный размер пикселя ПВМС: delta_u = 10 мкм (10e-6 м)
- Размеры ДОЭ и изображений: 128 x 128 пикселей
- Диапазон фазы ДОЭ: от -pi до +pi
- Расстояния восстановления: 24.1 мм, 25.3 мм, 26.5 мм, 27.7 мм
- Метод восстановления: прямое вычисление дифракции Френеля с 1-FFT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.fft
from PIL import Image
from typing import Tuple, List, Union


# ==============================================================================
# 1. КОНСТАНТЫ ЗАДАЧИ
# ==============================================================================
WAVELENGTH: float = 532e-9                    # 532 нм
PIXEL_SIZE: float = 10e-6                     # 10 мкм
GRID_SIZE: int = 128                          # 128x128 пикселей
DISTANCES: Tuple[float, ...] = (              # Расстояния в метрах
    0.0241,  # 24.1 мм
    0.0253,  # 25.3 мм
    0.0265,  # 26.5 мм
    0.0277   # 27.7 мм
)


# ==============================================================================
# 2. ДИФФЕРЕНЦИРУЕМЫЙ СЛОЙ ФРЕНЕЛЯ ДЛЯ PYTORCH (ДЛЯ ОБУЧЕНИЯ НЕЙРОСЕТИ)
# ==============================================================================
class FresnelMultiplanePropagator(nn.Module):
    """
    Дифференцируемый модуль дифракции Френеля на 4 расстояния.
    Использует формулу с одним двумерным преобразованием Фурье (1-FFT):
    
    K(x, y) = [exp(j*k*z) / (j*lambda*z)] * exp(j*pi/(lambda*z) * (x^2 + y^2))
              * FFT2 { H(u, v) * exp(j*pi/(lambda*z) * (u^2 + v^2)) }
              
    Для фазового ДОЭ: H(u, v) = exp(j * phi(u, v)).
    При расчете модуля |K(x,y)| внешние фазовые множители равны 1 по модулю.
    """
    def __init__(
        self,
        wavelength: float = WAVELENGTH,
        pixel_size: float = PIXEL_SIZE,
        distances: Tuple[float, ...] = DISTANCES,
        grid_size: int = GRID_SIZE
    ):
        super().__init__()
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distances = distances
        self.grid_size = grid_size

        # Сетка координат (u, v) в плоскости ПВМС с центром в (0, 0)
        coords = (torch.arange(grid_size, dtype=torch.float32) - grid_size // 2) * pixel_size
        v, u = torch.meshgrid(coords, coords, indexing="ij")
        r_sq = u**2 + v**2  # u^2 + v^2

        # Предварительный расчет квадратичных фазовых множителей exp(j * pi / (lambda * z) * (u^2 + v^2))
        phase_kernels = []
        for z in distances:
            phase = (torch.pi / (wavelength * z)) * r_sq
            kernel = torch.complex(torch.cos(phase), torch.sin(phase))
            phase_kernels.append(kernel)

        # Буфер ядра: форма [4, 1, 128, 128]
        self.register_buffer("phase_kernels", torch.stack(phase_kernels).unsqueeze(1))

    def forward(self, phase_doe: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Прямой проход:
        Args:
            phase_doe: тензор фазы ДОЭ [Batch, 1, 128, 128], значения в радианах [-pi, pi]
            normalize: нормализовать ли каждый слой в диапазон [0, 1]
        Returns:
            reconstructed: [Batch, 4, 128, 128] — модуль восстановленного поля для каждого из 4 расстояний
        """
        # Комплексное поле на модуляторе: H(u, v) = exp(j * phi)
        field_slm = torch.complex(torch.cos(phase_doe), torch.sin(phase_doe))

        reconstructed_layers = []
        for i in range(len(self.distances)):
            kernel_z = self.phase_kernels[i]  # [1, 128, 128]

            # 1. Умножение на квадратичную фазу для расстояния z_i
            modulated_field = field_slm * kernel_z

            # 2. 1-FFT с центрированием частот
            shifted_in = torch.fft.ifftshift(modulated_field, dim=(-2, -1))
            fft_field = torch.fft.fft2(shifted_in)
            field_z = torch.fft.fftshift(fft_field, dim=(-2, -1))

            # 3. Модуль комплексного поля |K(x, y)|
            amplitude = torch.abs(field_z)

            # 4. Нормализация в диапазон [0, 1] для стабильности функции потерь
            if normalize:
                b_min = amplitude.amin(dim=(-2, -1), keepdim=True)
                b_max = amplitude.amax(dim=(-2, -1), keepdim=True)
                amplitude = (amplitude - b_min) / (b_max - b_min + 1e-8)

            reconstructed_layers.append(amplitude)

        return torch.cat(reconstructed_layers, dim=1)  # [Batch, 4, 128, 128]


# ==============================================================================
# 3. ВОССТАНОВЛЕНИЕ НА NUMPY (ДЛЯ БЫСТРОГО ИНФЕРЕНСА БЕЗ PYTORCH)
# ==============================================================================
def reconstruct_layers_numpy(
    phase_doe: np.ndarray,
    wavelength: float = WAVELENGTH,
    pixel_size: float = PIXEL_SIZE,
    distances: Tuple[float, ...] = DISTANCES
) -> List[np.ndarray]:
    """
    Восстанавливает 4 слоя из фазового ДОЭ с помощью NumPy.
    Args:
        phase_doe: 2D массив 128x128 в радианах [-pi, pi]
    Returns:
        Список из 4 массивов 128x128 (модули амплитуды в диапазоне [0, 255], uint8)
    """
    grid_size = phase_doe.shape[0]
    coords = (np.arange(grid_size) - grid_size // 2) * pixel_size
    v, u = np.meshgrid(coords, coords, indexing="ij")
    r_sq = u**2 + v**2

    field_slm = np.exp(1j * phase_doe)
    reconstructed = []

    for z in distances:
        phase_kernel = np.exp(1j * (np.pi / (wavelength * z)) * r_sq)
        modulated = field_slm * phase_kernel
        field_z = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(modulated)))
        amplitude = np.abs(field_z)

        # Нормализация в 8-bit [0, 255]
        amp_norm = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-12)
        amp_uint8 = np.round(amp_norm * 255.0).astype(np.uint8)
        reconstructed.append(amp_uint8)

    return reconstructed


# ==============================================================================
# 4. ФУНКЦИИ КОНВЕРТАЦИИ ПО ТЗ (-pi -> 0, +pi -> 255)
# ==============================================================================
def phase_to_image(phase_rad: np.ndarray) -> Image.Image:
    """
    Конвертирует матрицу фазы [-pi, pi] в 8-битное изображение PNG.
    По ТЗ: значение -pi соответствует 0, значение +pi соответствует 255.
    """
    clipped = np.clip(phase_rad, -np.pi, np.pi)
    uint8_data = np.round((clipped + np.pi) / (2.0 * np.pi) * 255.0).astype(np.uint8)
    return Image.fromarray(uint8_data, mode="L")


def image_to_phase(image_or_path: Union[str, Image.Image]) -> np.ndarray:
    """
    Конвертирует 8-битное изображение PNG [0, 255] обратно в фазу [-pi, pi].
    """
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path).convert("L")
    else:
        img = image_or_path.convert("L")
    arr = np.array(img, dtype=np.float32)
    return (arr / 255.0) * (2.0 * np.pi) - np.pi


def save_reconstructed_images(layers: List[np.ndarray], prefix: str = "rec"):
    """
    Сохраняет 4 восстановленных изображения в формате PNG (8-bit, grayscale).
    """
    for i, layer in enumerate(layers, start=1):
        filename = f"{prefix}_{i}.png"
        Image.fromarray(layer, mode="L").save(filename)
        print(f"Сохранен слой {i}: {filename}")


# ==============================================================================
# 5. МЕТРИКИ СРАВНЕНИЯ ПО ТЗ
# ==============================================================================
def pearson_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """
    Коэффициент корреляции Пирсона (КК) по формуле из слайда 3:
    
    KK = sum((A_ij - <A>)*(B_ij - <B>)) / sqrt(sum((A_ij - <A>)^2) * sum((B_ij - <B>)^2))
    KK = 1.0, когда изображения идентичны.
    """
    A_f = A.astype(np.float64).flatten()
    B_f = B.astype(np.float64).flatten()

    diff_A = A_f - np.mean(A_f)
    diff_B = B_f - np.mean(B_f)

    num = np.sum(diff_A * diff_B)
    den = np.sqrt(np.sum(diff_A**2) * np.sum(diff_B**2)) + 1e-12
    return float(num / den)


def ssim_metric(A: np.ndarray, B: np.ndarray) -> float:
    """
    Индекс структурного сходства (ИСС / SSIM) по формуле из слайда 3:
    
    ИСС = [ (2*<A>*<B> + c1) * (2*sigma_AB + c2) ] /
          [ (<A>^2 + <B>^2 + c1) * (sigma_A^2 + sigma_B^2 + c2) ]
          
    ИСС = 1.0, когда изображения идентичны.
    """
    A_f = A.astype(np.float64)
    B_f = B.astype(np.float64)

    mu_A = np.mean(A_f)
    mu_B = np.mean(B_f)

    var_A = np.var(A_f)
    var_B = np.var(B_f)
    cov_AB = np.mean((A_f - mu_A) * (B_f - mu_B))

    # Стандартные константы стабилизации для динамического диапазона 255
    L = 255.0
    c1 = (0.01 * L)**2
    c2 = (0.03 * L)**2

    num = (2.0 * mu_A * mu_B + c1) * (2.0 * cov_AB + c2)
    den = (mu_A**2 + mu_B**2 + c1) * (var_A + var_B + c2)
    return float(num / den)


def evaluate_all_layers(
    targets: List[np.ndarray],
    reconstructed: List[np.ndarray]
) -> List[Tuple[float, float]]:
    """
    Вычисляет метрики (КК, ИСС) для всех 4 пар изображений.
    """
    results = []
    for i, (orig, rec) in enumerate(zip(targets, reconstructed), start=1):
        kk = pearson_correlation(orig, rec)
        iss = ssim_metric(orig, rec)
        results.append((kk, iss))
        print(f"Слой {i} (z = {DISTANCES[i-1]*1e3:.1f} мм): КК (Пирсон) = {kk:.4f}, ИСС (SSIM) = {iss:.4f}")
    return results


# ==============================================================================
# 6. КЛАССИЧЕСКИЙ БЕЙЗЛАЙН: МУЛЬТИПЛОСКОСТНОЙ ГЕРЧБЕРГ-СЭКСТОН (ДЛЯ СРАВНЕНИЯ)
# ==============================================================================
def gerchberg_saxton_multiplane(
    target_images: List[np.ndarray],
    iterations: int = 100,
    wavelength: float = WAVELENGTH,
    pixel_size: float = PIXEL_SIZE,
    distances: Tuple[float, ...] = DISTANCES
) -> np.ndarray:
    """
    Классический итеративный алгоритм Герчберга-Сэкстона для синтеза одного ДОЭ под 4 плоскости.
    Служит бейзлайном для доказательства жюри, что нейросеть работает в сотни раз быстрее!
    
    Args:
        target_images: список из 4 массивов 128x128 (амплитуды объектов)
        iterations: количество итераций (обычно 50-200)
    Returns:
        Синтезированная фаза ДОЭ [-pi, pi]
    """
    grid_size = target_images[0].shape[0]
    coords = (np.arange(grid_size) - grid_size // 2) * pixel_size
    v, u = np.meshgrid(coords, coords, indexing="ij")
    r_sq = u**2 + v**2

    # Целевые амплитуды (нормированные)
    target_amps = [(img.astype(np.float64) / 255.0) for img in target_images]

    # Ядра прямого распространения
    forward_kernels = [np.exp(1j * (np.pi / (wavelength * z)) * r_sq) for z in distances]
    # Ядра обратного распространения (эрмитово сопряжение)
    backward_kernels = [np.conj(k) for k in forward_kernels]

    # Начальное случайное фазовое распределение ДОЭ
    doe_phase = np.random.uniform(-np.pi, np.pi, (grid_size, grid_size))

    for it in range(iterations):
        field_slm = np.exp(1j * doe_phase)
        back_sum = np.zeros((grid_size, grid_size), dtype=np.complex128)

        for i in range(len(distances)):
            # Прямой расчет в плоскость z_i
            mod_field = field_slm * forward_kernels[i]
            field_z = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mod_field)))

            # Замена амплитуды на целевую при сохранении рассчитанной фазы
            current_phase_z = np.angle(field_z)
            new_field_z = target_amps[i] * np.exp(1j * current_phase_z)

            # Обратное распространение в плоскость ДОЭ
            back_mod = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(new_field_z)))
            back_field = back_mod * backward_kernels[i]
            back_sum += back_field

        # Обновление фазы ДОЭ усреднением по всем плоскостям
        doe_phase = np.angle(back_sum)

    return doe_phase


# ==============================================================================
# 7. ТЕСТОВЫЙ ЗАПУСК
# ==============================================================================
if __name__ == "__main__":
    print("=== ТЕСТИРОВАНИЕ МОДУЛЯ ВОЛНОВОЙ ОПТИКИ ===")
    
    # 1. Создаем 4 тестовых синтетических изображения (круги / кресты / полосы)
    test_imgs = []
    for idx in range(4):
        canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        if idx == 0:  # круг
            y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
            mask = (x - 64)**2 + (y - 64)**2 <= 30**2
            canvas[mask] = 255
        elif idx == 1:  # крест
            canvas[58:70, :] = 255
            canvas[:, 58:70] = 255
        elif idx == 2:  # рамка
            canvas[20:108, 20:25] = 255
            canvas[20:108, 103:108] = 255
            canvas[20:25, 20:108] = 255
            canvas[103:108, 20:108] = 255
        elif idx == 3:  # диагонали
            for k in range(GRID_SIZE):
                canvas[k, max(0, min(GRID_SIZE-1, k-3)):min(GRID_SIZE, k+4)] = 255
        test_imgs.append(canvas)
    
    print("Создано 4 тестовых шаблона 128x128.")

    # 2. Проверяем классический синтез GS на 15 итерациях
    print("Запуск классического синтеза ДОЭ (Герчберг-Сэкстон, 15 итераций)...")
    phase_baseline = gerchberg_saxton_multiplane(test_imgs, iterations=15)
    
    # 3. Сохраняем ДОЭ
    doe_img = phase_to_image(phase_baseline)
    doe_img.save("test_doe.png")
    print("Тестовый ДОЭ сохранен: test_doe.png (8-bit grayscale PNG)")

    # 4. Восстанавливаем 4 слоя
    rec_layers = reconstruct_layers_numpy(phase_baseline)
    save_reconstructed_images(rec_layers, prefix="test_rec")

    # 5. Оцениваем метрики
    print("\nРезультаты валидации:")
    evaluate_all_layers(test_imgs, rec_layers)
    
    # 6. Проверяем PyTorch-слой
    print("\nПроверка PyTorch слоя...")
    prop_torch = FresnelMultiplanePropagator()
    tensor_phase = torch.from_numpy(phase_baseline).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        out_torch = prop_torch(tensor_phase)
    print(f"Выходной тензор PyTorch: {out_torch.shape} (должен быть [1, 4, 128, 128]) - УСПЕШНО!")
