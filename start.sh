#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " Директория проекта: $SCRIPT_DIR"
echo "=========================================="

# 1. Проверяем наличие uv
if ! command -v uv &> /dev/null; then
    echo "📦 Установка uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ uv version: $(uv --version)"

# 2. Создаём venv на Python 3.12 (uv сам скачает его, если нужно)
if [ ! -d "venv" ]; then
    echo "📦 Создание venv на Python 3.12..."
    uv venv venv --python 3.12
else
    echo "✅ venv уже существует"
fi

# 3. Активируем
source venv/bin/activate

# 4. Устанавливаем maia2 БЕЗ зависимостей (быстро через uv)
echo "📥 Установка maia2 (без зависимостей)..."
uv pip install --no-deps git+https://github.com/CSSLab/maia2

# 5. Устанавливаем остальные зависимости
echo "📥 Установка зависимостей из requirements.txt..."
uv pip install -r requirements.txt

# 6. Скачиваем модели, если их нет
if [ ! -d "models" ] || [ ! -f "models/tf_model_19x256.keras" ]; then
    echo "⬇️  Скачивание моделей и данных..."
    uv pip install gdown
    gdown --folder "https://drive.google.com/drive/folders/1eegVg9K5tn4KqDwbuMyUgeh_lyxGjVTl" -O . --quiet
fi

# 7. Создаём .env для локального режима
if [ ! -f ".env" ]; then
    echo "ENV=local" > .env
fi

echo "=========================================="
echo " Python version in venv:"
python --version
echo "=========================================="
echo " Запуск Chess Blunder Detector API..."
echo "=========================================="
python web_api.py