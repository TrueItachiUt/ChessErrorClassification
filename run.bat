@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Environment is not initialized.
    echo Run activate.bat first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe web_api.py
pause