#!/bin/bash
echo "=========================================="
echo " Настройка виртуального окружения..."
echo "=========================================="

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

if [ ! -d "models" ] || [ ! -f "models/tf_model_19x256.keras" ]; then
    echo "⬇️  Скачивание моделей и данных..."
    gdown --folder "https://drive.google.com/drive/folders/1eegVg9K5tn4KqDwbuMyUgeh_lyxGjVTl" -O . --quiet
fi

# 2. Создаем .env, если его нет (чтобы код понял, что ENV=local)
if [ ! -f ".env" ]; then
    echo "ENV=local" > .env
fi

echo "=========================================="
echo " Запуск Chess Blunder Detector API..."
echo "=========================================="

python web_api.py