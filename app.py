"""
Интерактивный веб-интерфейс на Gradio для хакатона по нейросетевому синтезу ДОЭ.
Запуск:
    python app.py
или
    uv run --with gradio --with torch --with torchvision --with numpy --with opencv-python python3 app.py
"""

import os
import time
import cv2
import numpy as np
import torch
import gradio as gr
from run import DOE_GeneratorNet, WinningFresnel, calc_metrics_and_loss, DEVICE, N, WAVELENGTH, PIXEL_SIZE, DISTANCES

# Загрузка обученной нейросети и физического слоя
MODEL_PATH = "mephi_doe_model.pth"
if not os.path.exists(MODEL_PATH):
    # Попробуем найти в родительской папке или Downloads
    if os.path.exists("/Users/maksim/Downloads/mephi_doe_model.pth"):
        MODEL_PATH = "/Users/maksim/Downloads/mephi_doe_model.pth"

model = DOE_GeneratorNet().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Модель успешно загружена из {MODEL_PATH}")
else:
    print(f"⚠️ Файл весов {MODEL_PATH} не найден!")

prop = WinningFresnel(N, WAVELENGTH, PIXEL_SIZE, DISTANCES).to(DEVICE)


def process_images(img1, img2, img3, img4, steps, return_intensity):
    start_time = time.time()
    
    if img1 is None or img2 is None or img3 is None or img4 is None:
        return None, None, None, None, None, "⚠️ Пожалуйста, загрузите все 4 изображения!", ""

    # Приводим к 128x128 grayscale [0, 1]
    raw_imgs = [img1, img2, img3, img4]
    processed_imgs = []
    for im in raw_imgs:
        if isinstance(im, str):
            arr = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        elif isinstance(im, np.ndarray):
            if len(im.shape) == 3:
                arr = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            else:
                arr = im
        else:
            arr = np.array(im)
            if len(arr.shape) == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
        arr = cv2.resize(arr, (128, 128))
        processed_imgs.append(arr.astype(np.float32) / 255.0)

    target_imgs = torch.tensor(np.stack(processed_imgs, axis=0)).unsqueeze(0).to(DEVICE)

    # 1. Инференс нейросети
    with torch.no_grad():
        init_phase = model(target_imgs)

    # 2. Оптимизация (если steps > 0)
    if steps > 0:
        editable = torch.nn.Parameter(init_phase.clone())
        opt = torch.optim.Adam([editable], lr=0.08)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.005)
        
        bg_mask = (target_imgs < 0.05).float()
        fg_mask = (target_imgs >= 0.05).float()

        for s in range(int(steps)):
            opt.zero_grad()
            rec = prop(editable, return_intensity=return_intensity)
            loss_m, _, _ = calc_metrics_and_loss(rec, target_imgs)
            fg_loss = (((rec - target_imgs) * fg_mask)**2).sum() / (fg_mask.sum() + 1e-8)
            bg_loss = ((rec * bg_mask)**2).mean()
            tot = loss_m + fg_loss * 2.5 + bg_loss * 8.0
            tot.backward()
            opt.step()
            sched.step()
            
        final_phase = editable.detach()
    else:
        final_phase = init_phase.detach()

    # Фаза строго в [-pi, pi]
    final_phase = torch.remainder(final_phase + np.pi, 2 * np.pi) - np.pi

    # 3. Восстановление 4 слоев
    with torch.no_grad():
        rec_layers = prop(final_phase, return_intensity=return_intensity)

    calc_time = time.time() - start_time

    # 4. Преобразование ДОЭ в изображение (uint8)
    doe_np = final_phase.cpu().squeeze().numpy()
    doe_uint8 = np.clip(np.round((doe_np + np.pi) / (2.0 * np.pi) * 255.0), 0, 255).astype(np.uint8)

    # 5. Преобразование восстановленных слоев в изображения и расчет метрик по слоям
    rec_np = rec_layers.cpu().squeeze().numpy()
    rec_imgs = []
    layer_metrics_html = "<table style='width:100%; text-align:center; border-collapse:collapse; margin-top:10px; font-size:14px;'>"
    layer_metrics_html += "<tr style='background:#2b2b2b; color:#fff;'><th>Слой</th><th>Расстояние z</th><th>КК (Пирсон)</th><th>ИСС (SSIM)</th></tr>"
    
    total_kk = 0.0
    total_iss = 0.0
    z_mms = [24.1, 25.3, 26.5, 27.7]

    for i in range(4):
        layer_uint8 = np.clip(np.round(rec_np[i] * 255.0), 0, 255).astype(np.uint8)
        rec_imgs.append(layer_uint8)
        
        # Считаем метрики для этого конкретного слоя
        orig_t = target_imgs[0, i:i+1]
        rec_t = rec_layers[0, i:i+1]
        _, kk_i, iss_i = calc_metrics_and_loss(rec_t, orig_t)
        total_kk += kk_i
        total_iss += iss_i

        bg_color = "#1e1e1e" if i % 2 == 0 else "#252525"
        layer_metrics_html += f"<tr style='background:{bg_color};'><td><b>Слой {i+1}</b></td><td>{z_mms[i]} мм</td><td><span style='color:#4ade80; font-weight:bold;'>{kk_i:.4f}</span></td><td><span style='color:#60a5fa; font-weight:bold;'>{iss_i:.4f}</span></td></tr>"

    avg_kk = total_kk / 4.0
    avg_iss = total_iss / 4.0
    layer_metrics_html += f"<tr style='background:#334155; font-weight:bold; color:#fff;'><td><b>ИТОГО (Среднее)</b></td><td>4 плоскости</td><td style='color:#4ade80;'>{avg_kk:.4f}</td><td style='color:#60a5fa;'>{avg_iss:.4f}</td></tr>"
    layer_metrics_html += "</table>"

    summary_card = f"""
    <div style='background:linear-gradient(135deg, #1e293b, #0f172a); padding:16px; border-radius:10px; border:1px solid #334155;'>
        <div style='display:flex; justify-content:space-around; align-items:center;'>
            <div style='text-align:center;'>
                <div style='font-size:12px; color:#94a3b8;'>Средний Пирсон (КК)</div>
                <div style='font-size:24px; font-weight:bold; color:#4ade80;'>{avg_kk:.4f}</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:12px; color:#94a3b8;'>Средний SSIM (ИСС)</div>
                <div style='font-size:24px; font-weight:bold; color:#60a5fa;'>{avg_iss:.4f}</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:12px; color:#94a3b8;'>Время работы</div>
                <div style='font-size:24px; font-weight:bold; color:#f59e0b;'>{calc_time:.2f} с</div>
            </div>
        </div>
    </div>
    """

    return doe_uint8, rec_imgs[0], rec_imgs[1], rec_imgs[2], rec_imgs[3], summary_card, layer_metrics_html


def load_demo_examples():
    """Загрузка тестовых картинок из проекта"""
    demos = ["1.png", "2.png", "3.png", "4.png"]
    found = []
    for d in demos:
        if os.path.exists(d):
            found.append(d)
        else:
            found.append(None)
    return found[0], found[1], found[2], found[3]


# Создание современного UI на Gradio
custom_css = """
.main-header { text-align: center; margin-bottom: 20px; }
.main-header h1 { font-size: 26px; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
"""

with gr.Blocks(title="AI Фотоника | Синтез ДОЭ") as demo:
    gr.HTML("""
    <div class='main-header'>
        <h1>🔬 Нейросетевой синтез дифракционных оптических элементов (ДОЭ)</h1>
    </div>
    """)

    with gr.Accordion("💡 Как это работает? (Физика и нейросеть простыми словами — нажмите, чтобы открыть)", open=False):
        gr.HTML("""
        <div style='background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 15px;'>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px;'>
                
                <div style='background: #0f172a; padding: 18px; border-radius: 10px; border-left: 4px solid #38bdf8;'>
                    <h3 style='margin-top:0; color: #38bdf8; font-size: 16px;'>1. 🔬 Физика: что делает луч?</h3>
                    <p style='color: #cbd5e1; font-size: 13.5px; line-height: 1.55; margin-bottom: 0;'>
                        Представьте прозрачное стеклышко с микрорельефом (ДОЭ). Зеленый лазер (532 нм) проходит сквозь него, лучи преломляются (дифракция Френеля) и интерферируют в пространстве. В воздухе за стеклышком рождается <b>настоящая 3D-сцена</b>: на 4 разных глубинах (24.1, 25.3, 26.5 и 27.7 мм) в воздухе четко фокусируются 4 разные картинки!
                    </p>
                </div>

                <div style='background: #0f172a; padding: 18px; border-radius: 10px; border-left: 4px solid #f59e0b;'>
                    <h3 style='margin-top:0; color: #f59e0b; font-size: 16px;'>2. ⏳ В чём была проблема?</h3>
                    <p style='color: #cbd5e1; font-size: 13.5px; line-height: 1.55; margin-bottom: 0;'>
                        Подобрать рельеф одного плоского стеклышка так, чтобы оно одновременно кодировало сразу 4 разных изображения — сложнейшая математическая задача. Классические итеративные методы (алгоритм Герчберга–Сэкстона) подбирали этот узор <b>минутами через сотни циклов</b>. Показать живое 3D-видео при такой скорости невозможно.
                    </p>
                </div>

                <div style='background: #0f172a; padding: 18px; border-radius: 10px; border-left: 4px solid #4ade80;'>
                    <h3 style='margin-top:0; color: #4ade80; font-size: 16px;'>3. 🧠 Наш AI: Physics-Informed U-Net</h3>
                    <p style='color: #cbd5e1; font-size: 13.5px; line-height: 1.55; margin-bottom: 0;'>
                        Мы заменили долгий перебор обученной нейросетью <b>U-Net</b>. Она принимает 4 картинки и <b>всего за 5 миллисекунд</b> синтезирует готовую фазовую матрицу [-π, +π]. Модель обучалась с встроенным дифференцируемым слоем оптики — градиенты оптимизатора обучали сеть напрямую по законам физики света.
                    </p>
                </div>

                <div style='background: #0f172a; padding: 18px; border-radius: 10px; border-left: 4px solid #a855f7;'>
                    <h3 style='margin-top:0; color: #a855f7; font-size: 16px;'>4. 🎯 Оценка качества и метрики</h3>
                    <p style='color: #cbd5e1; font-size: 13.5px; line-height: 1.55; margin-bottom: 0;'>
                        Восстановленные слои сравниваются с оригиналами по двум строгим метрикам: <b>Коэффициент Пирсона (КК)</b> проверяет корреляцию яркости, а <b>SSIM (ИСС)</b> — структурную схожесть контуров. В режиме доводки система разгоняет метрики выше <b>0.55+</b> за пару секунд.
                    </p>
                </div>

            </div>
        </div>
        """)

    with gr.Row():
        # ЛЕВАЯ КОЛОНКА: Входные данные и управление
        with gr.Column(scale=4):
            gr.Markdown("### 📥 1. Входные срезы 3D-сцены (4 изображения 128×128)")
            with gr.Row():
                in_img1 = gr.Image(label="Слой 1 (z = 24.1 мм)", type="numpy", sources=["upload"])
                in_img2 = gr.Image(label="Слой 2 (z = 25.3 мм)", type="numpy", sources=["upload"])
            with gr.Row():
                in_img3 = gr.Image(label="Слой 3 (z = 26.5 мм)", type="numpy", sources=["upload"])
                in_img4 = gr.Image(label="Слой 4 (z = 27.7 мм)", type="numpy", sources=["upload"])

            btn_demo = gr.Button("📂 Загрузить демонстрационные примеры", variant="secondary")

            gr.Markdown("### ⚙️ 2. Параметры синтеза")
            slider_steps = gr.Slider(
                minimum=0,
                maximum=500,
                value=150,
                step=25,
                label="Шаги оптимизации (0 = мгновенный вывод чистой нейросети ~5 мс, 150 = баланс, 400 = максимум качества)"
            )
            chk_intensity = gr.Checkbox(label="Возвращать интенсивность (|K|²) вместо модуля (|K|)", value=False)

            btn_run = gr.Button("🚀 СИНТЕЗИРОВАТЬ ДОЭ И ВОССТАНОВИТЬ СЛОИ", variant="primary", scale=2)

        # ПРАВАЯ КОЛОНКА: Результаты
        with gr.Column(scale=5):
            gr.Markdown("### 📤 3. Результаты работы системы")
            
            summary_box = gr.HTML("<div style='text-align:center; padding:20px; color:#64748b;'>Загрузите 4 картинки и нажмите «Синтезировать ДОЭ»</div>")

            with gr.Row():
                with gr.Column(scale=1):
                    out_doe = gr.Image(label="Синтезированный ДОЭ (Голограмма 128×128)", format="png")
                with gr.Column(scale=2):
                    metrics_table = gr.HTML()

            gr.Markdown("### 🌈 Восстановленные слои 3D-сцены (дифракция Френеля)")
            with gr.Row():
                out_rec1 = gr.Image(label="Восстановлен: z = 24.1 мм")
                out_rec2 = gr.Image(label="Восстановлен: z = 25.3 мм")
            with gr.Row():
                out_rec3 = gr.Image(label="Восстановлен: z = 26.5 мм")
                out_rec4 = gr.Image(label="Восстановлен: z = 27.7 мм")

    # Логика кнопок
    btn_demo.click(
        fn=load_demo_examples,
        inputs=[],
        outputs=[in_img1, in_img2, in_img3, in_img4]
    )

    btn_run.click(
        fn=process_images,
        inputs=[in_img1, in_img2, in_img3, in_img4, slider_steps, chk_intensity],
        outputs=[out_doe, out_rec1, out_rec2, out_rec3, out_rec4, summary_box, metrics_table]
    )


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
