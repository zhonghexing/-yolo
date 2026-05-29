@echo off
chcp 65001 >nul
title 环境检查

echo ========================================
echo   钢材缺陷检测系统 - 环境检查
echo ========================================
echo.

:: Python 版本
echo [1/5] Python 版本:
python --version
echo.

:: PyTorch & CUDA
echo [2/5] PyTorch:
python -c "import torch; print(f'  版本: {torch.__version__}'); print(f'  CUDA 可用: {torch.cuda.is_available()}')" 2>nul || echo   [未安装]
echo.

:: Ultralytics
echo [3/5] Ultralytics (YOLOv8):
python -c "import ultralytics; print(f'  版本: {ultralytics.__version__}')" 2>nul || echo   [未安装]
echo.

:: PyQt5
echo [4/5] PyQt5:
python -c "from PyQt5.QtCore import QT_VERSION_STR; print(f'  Qt 版本: {QT_VERSION_STR}')" 2>nul || echo   [未安装]
echo.

:: OpenCV
echo [5/5] OpenCV:
python -c "import cv2; print(f'  版本: {cv2.__version__}')" 2>nul || echo   [未安装]
echo.

:: 模型文件
echo [模型] best.pt:
if exist best.pt (
    for %%A in (best.pt) do echo   大小: %%~zA bytes
) else (
    echo   [未找到] 请确保 best.pt 在当前目录
)
echo.

echo ========================================
echo   检查完成
echo ========================================
pause
