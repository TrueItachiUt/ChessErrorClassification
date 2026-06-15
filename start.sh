#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " Директория проекта: $SCRIPT_DIR"
echo "=========================================="

USE_UV=false

# --------------------------------------------------
# Поиск Python 3.12
# --------------------------------------------------
if command -v python3.12 &> /dev/null; then
    echo "✅ Найден python3.12"

    if [ ! -d ".venv" ]; then
        echo "📦 Создание виртуального окружения..."
        python3.12 -m venv .venv
    else
        echo "✅ .venv уже существует"
    fi

    source .venv/bin/activate

    echo "📦 Обновление pip..."
    python -m pip install --upgrade pip

    echo "📥 Установка maia2..."
    pip install --no-deps git+https://github.com/CSSLab/maia2

    echo "📥 Установка зависимостей..."
    pip install -r requirements.txt

else
    USE_UV=true

    echo "⚠️ python3.12 не найден, используем uv"

    if ! command -v uv &> /dev/null; then
        echo "📦 Установка uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    echo "✅ uv version: $(uv --version)"

    echo "📦 Установка Python 3.12 через uv..."
    uv python install 3.12

    if [ ! -d ".venv" ]; then
        echo "📦 Создание виртуального окружения..."
        uv venv .venv --python 3.12
    else
        echo "✅ .venv уже существует"
    fi

    echo "📥 Установка maia2..."
    uv pip install --python .venv/bin/python --no-deps \
        git+https://github.com/CSSLab/maia2

    echo "📥 Установка зависимостей..."
    uv pip install --python .venv/bin/python \
        -r requirements.txt
fi

# --------------------------------------------------
# Модели
# --------------------------------------------------
if [ ! -d "models" ] || [ ! -f "models/tf_model_19x256.keras" ]; then
    echo "⬇️ Скачивание моделей..."

    if [ "$USE_UV" = true ]; then
        uv pip install --python .venv/bin/python gdown
    else
        .venv/bin/python -m pip install gdown
    fi

    .venv/bin/gdown \
        --folder "https://drive.google.com/drive/folders/1eegVg9K5tn4KqDwbuMyUgeh_lyxGjVTl" \
        -O .
else
    echo "✅ Модели уже скачаны"
fi

# --------------------------------------------------
# .env
# --------------------------------------------------
if [ ! -f ".env" ]; then
    echo "ENV=local" > .env
    echo "✅ Создан .env"
fi

echo "=========================================="
echo " Python version:"
.venv/bin/python --version
echo "=========================================="
echo "✅ Настройка окружения завершена!"
echo "=========================================="
echo "🚀 Запуск проекта:"
echo "   source .venv/bin/activate"
echo "   python web_api.py"
echo ""
echo "или"
echo ""
echo "   .venv/bin/python web_api.py"
echo "=========================================="