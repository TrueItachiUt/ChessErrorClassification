# Chess Blunder Detector

Веб-приложение для анализа шахматных партий и поиска тактических ошибок с использованием Stockfish, Maia2 и обученных моделей классификации тактик.

## Требования

Перед установкой убедитесь, что у вас установлены:

* Python 3.12 или ниже (Python 3.13 не поддерживается)
* Git
* Git LFS
* Bash (Linux/macOS или Git Bash для Windows)

Проверить версии:

```bash
python3.12 --version
git --version
git lfs version
```

Если Git LFS установлен впервые:

```bash
git lfs install
```

---

## Клонирование репозитория

```bash
git clone <repository_url>
cd <repository_name>
git lfs pull
```

---

## Быстрый запуск

Запустите скрипт настройки:

```bash
chmod +x start.sh
./start.sh
```

Скрипт автоматически:

1. Создаст виртуальное окружение `.venv`
2. Установит зависимости
3. Установит Maia2
4. Скачает модели
5. Создаст файл `.env`

После завершения:

```bash
source .venv/bin/activate
python web_api.py
```

Сервис будет доступен по адресу:

```text
http://localhost:8000
```

---

## Ручная установка

### Создание виртуального окружения

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Установка Maia2

```bash
pip install --no-deps git+https://github.com/CSSLab/maia2
```

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Скачивание моделей

```bash
pip install gdown

gdown --folder \
"https://drive.google.com/drive/folders/1eegVg9K5tn4KqDwbuMyUgeh_lyxGjVTl" \
-O .
```

### Создание .env

```bash
echo "ENV=local" > .env
```

### Запуск

```bash
python web_api.py
```

---

## Структура проекта

```text
.
├── web_api.py          # FastAPI сервер
├── backend_tools.py    # Основная логика анализа
├── Dataset.py
├── requirements.txt
├── start.sh
├── models/
├── static/
│   └── index.html
└── .env
```

---

## Возможные проблемы

### Python 3.13

Проект не поддерживает Python 3.13.

Используйте Python 3.12 или более раннюю версию:

```bash
python3.12 --version
```

### Отсутствуют модели

Удалите папку `models` и снова запустите:

```bash
./start.sh
```

### Не найден Git LFS

Установите Git LFS:

Ubuntu/Debian:

```bash
sudo apt install git-lfs
git lfs install
```

macOS:

```bash
brew install git-lfs
git lfs install
```
