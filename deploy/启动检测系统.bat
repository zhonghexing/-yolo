@echo off
chcp 65001 >nul
title 钢材缺陷检测系统

echo ========================================
echo   钢材缺陷检测系统 - 环境安装 & 启动
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 安装依赖
echo [1/2] 检查并安装依赖...
pip install ultralytics PyQt5 opencv-python numpy Pillow matplotlib seaborn scikit-learn pandas tqdm pyttsx3 flask flask-socketio --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

echo.
echo [2/2] 启动检测系统...
python app.py
pause
