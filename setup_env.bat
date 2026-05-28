@echo off
echo ============================================================
echo 配置 YOLO 项目环境变量
echo ============================================================
echo.

REM 获取当前用户 PATH
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"

REM 要添加的路径
set "NEW_PATHS=D:\Anaconda3\envs\yolo_screw;D:\Anaconda3\envs\yolo_screw\Scripts;D:\Anaconda3\envs\yolo_screw\Library\bin"

REM 检查是否已包含
echo %USER_PATH% | find /i "yolo_screw" >nul
if %errorlevel% equ 0 (
    echo [信息] 环境变量已配置，无需重复添加
) else (
    echo [操作] 添加 yolo_screw 环境到 PATH...
    reg add "HKCU\Environment" /v Path /t REG_EXPAND_SZ /d "%USER_PATH%;%NEW_PATHS%" /f >nul
    if %errorlevel% equ 0 (
        echo [成功] 环境变量已添加！
    ) else (
        echo [错误] 添加失败，请手动添加
    )
)

echo.
echo ============================================================
echo 需要添加的路径：
echo ============================================================
echo   D:\Anaconda3\envs\yolo_screw
echo   D:\Anaconda3\envs\yolo_screw\Scripts
echo   D:\Anaconda3\envs\yolo_screw\Library\bin
echo.
echo ============================================================
echo 验证配置
echo ============================================================
echo.
echo [测试] 检查 Python...
D:\Anaconda3\envs\yolo_screw\python.exe --version
echo.
echo [测试] 检查 PyTorch...
D:\Anaconda3\envs\yolo_screw\python.exe -c "import torch; print(f'PyTorch {torch.__version__}')"
echo.
echo [测试] 检查 ultralytics...
D:\Anaconda3\envs\yolo_screw\python.exe -c "import ultralytics; print(f'ultralytics {ultralytics.__version__}')"
echo.
echo ============================================================
echo 配置完成！请重新打开命令行窗口使环境变量生效
echo ============================================================
echo.
echo 使用方法：
echo   1. 打开新的 CMD 或 PowerShell
echo   2. 激活环境: conda activate yolo_screw
echo   3. 或直接使用: python train.py
echo.
pause
