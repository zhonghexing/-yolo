@echo off
chcp 65001 >nul
title 命令行检测

echo ========================================
echo   钢材缺陷检测 - 命令行模式
echo ========================================
echo.

:: 检查参数
if "%~1"=="" (
    echo 用法: 命令行检测.bat 图片路径
    echo 示例: 命令行检测.bat test_images\crazing_271.jpg
    echo.
    echo 未指定图片，使用 test_images 目录演示...
    python inference.py --dir test_images
) else (
    python inference.py --image "%~1"
)
pause
