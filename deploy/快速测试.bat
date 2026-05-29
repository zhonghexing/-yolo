@echo off
chcp 65001 >nul
title 快速测试

echo ========================================
echo   钢材缺陷检测系统 - 快速测试
echo ========================================
echo.

:: 检查依赖
python -c "import ultralytics, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo [提示] 首次运行请先双击 启动检测系统.bat 安装依赖
    pause
    exit /b 1
)

:: 检查模型
if not exist best.pt (
    echo [错误] 未找到 best.pt 模型文件
    pause
    exit /b 1
)

:: 运行检测测试
echo 使用 test_images 目录下的 4 张图片进行测试...
echo.
python inference.py --dir test_images
echo.
echo ========================================
echo   测试完成！查看上方检测结果
echo ========================================
pause
