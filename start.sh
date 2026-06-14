#!/bin/bash
echo "=========================================="
echo " Настройка виртуального окружения..."
echo "=========================================="

# 1. Ищем Python 3.12 (нужен для maia2 и совместимости с TensorFlow)
PYTHON=""
if command -v python3.12 &> /dev/null; then
    PYTHON="python3.12"
elif command -v python3.12.13 &> /dev/null; then
    PYTHON="python3.12.13"
else
    echo "❌ Python 3.12 не найден в системе!"
    echo "Установите его, например:"
    echo "  sudo apt install python3.12 python3.12-venv python3.12-dev"
    echo "  # или через pyenv:"
    echo "  pyenv install 3.12.13"
    exit 1
fi

echo "✅ Используем: $($PYTHON --version)"

# 2. Создаём venv только если его нет
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения на $($PYTHON --version)..."
    $PYTHON -m venv venv
fi

# 3. Активируем
source venv/bin/activate

# 4. Обновляем pip
pip install --upgrade pip setuptools wheel -q

# 5. Устанавливаем maia2 БЕЗ зависимостей (они конфликтуют с TF)
echo "📥 Установка maia2 (без зависимостей)..."
pip install --no-deps git+https://github.com/CSSLab/maia2 -q

# 6. Устанавливаем остальные зависимости
echo "📥 Установка зависимостей из requirements.txt..."
pip install -q -r requirements.txt

# 7. Скачиваем модели, если их нет
if [ ! -d "models" ] || [ ! -f "models/tf_model_19x256.keras" ]; then
    echo "⬇️  Скачивание моделей и данных..."
    gdown --folder "https://drive.google.com/drive/folders/1eegVg9K5tn4KqDwbuMyUgeh_lyxGjVTl" -O . --quiet
fi

# 8. Создаём .env для локального режима
if [ ! -f ".env" ]; then
    echo "ENV=local" > .env
fi

echo "=========================================="
echo " Запуск Chess Blunder Detector API..."
echo "=========================================="
python web_api.py