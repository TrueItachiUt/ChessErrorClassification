@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo Project directory: %CD%
echo ==========================================

REM --------------------------------------------------
REM Поиск Python
REM --------------------------------------------------

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.12 not found.
    echo Install Python 3.12 and try again.
    pause
    exit /b 1
)

echo [OK] Python 3.12 found

if not exist ".venv" (
    echo Creating virtual environment...
    py -3.12 -m venv .venv
) else (
    echo [OK] .venv already exists
)

call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing maia2...
pip install --no-deps git+https://github.com/CSSLab/maia2

echo Installing dependencies...
pip install -r requirements.txt

REM --------------------------------------------------
REM Models
REM --------------------------------------------------

if not exist "models\tf_model_19x256.keras" (

    echo Downloading models...

    pip install gdown

    gdown ^
      --folder "https://drive.google.com/drive/folders/1eegVg9K5tn4KqDwbuMyUgeh_lyxGjVTl" ^
      -O .

) else (
    echo [OK] Models already downloaded
)

REM --------------------------------------------------
REM .env
REM --------------------------------------------------

if not exist ".env" (
    echo ENV=local>.env
    echo [OK] .env created
)

echo ==========================================
python --version
echo ==========================================
echo Environment ready
echo ==========================================
echo Run:
echo.
echo     .venv\Scripts\activate
echo     python web_api.py
echo.
echo or
echo.
echo     .venv\Scripts\python.exe web_api.py
echo ==========================================

pause