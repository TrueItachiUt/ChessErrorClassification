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

# 2. Создаём .venv на Python 3.12 (стандартное имя для uv и VSCode)
if [ ! -d ".venv" ]; then
    echo "📦 Создание .venv на Python 3.12..."
    uv venv .venv --python 3.12
else
    echo "✅ .venv уже существует"
fi

# 3. Устанавливаем maia2 БЕЗ зависимостей (быстро через uv)
echo "📥 Установка maia2 (без зависимостей)..."
uv pip install --python .venv/bin/python --no-deps git+https://github.com/CSSLab/maia2

# 4. Устанавливаем остальные зависимости
echo "📥 Установка зависимостей из requirements.txt..."
uv pip install --python .venv/bin/python -r requirements.txt

# 5. Скачиваем модели, если их нет
if [ ! -d "models" ] || [ ! -f "models/tf_model_19x256.keras" ]; then
    echo "⬇️  Скачивание моделей и данных..."
    uv pip install --python .venv/bin/python gdown
    # Вызываем gdown напрямую из .venv, чтобы не зависеть от PATH
    .venv/bin/gdown --folder "https://drive.google.com/drive/folders/1eegVg9K5tn4KqDwbuMyUgeh_lyxGjVTl" -O . --quiet
fi

# 6. Создаём .env для локального режима
if [ ! -f ".env" ]; then
    echo "ENV=local" > .env
fi

echo "=========================================="
echo " Python version in .venv:"
.venv/bin/python --version
echo "=========================================="
echo "✅ Настройка окружения успешно завершена!"
echo "=========================================="
echo "🚀 Чтобы запустить проект из терминала:"
echo "   uv run web_api.py"
echo "=========================================="