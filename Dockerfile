FROM python:3.11-slim

WORKDIR /app

# Системные библиотеки для OpenCV и работы с изображениями
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода и обученных весов модели
COPY . .

# По умолчанию запускаем генерацию ДОЭ
ENTRYPOINT ["python", "run.py"]
