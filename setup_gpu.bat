@echo off
REM ============================================================
REM YOLOv8 钢材表面缺陷检测 - GPU 环境配置脚本
REM 适用于 NVIDIA RTX 5070 (Blackwell/sm_120) + CUDA 12.4
REM ============================================================

echo ============================================================
echo YOLOv8 钢材表面缺陷检测 - GPU 环境配置
echo ============================================================
echo.

REM 检查 conda 是否可用
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 conda，请先安装 Anaconda3
    pause
    exit /b 1
)

REM 创建 conda 环境
echo [1/4] 创建 conda 环境: yolo_screw
conda create -n yolo_screw python=3.10 -y
if %errorlevel% neq 0 (
    echo [错误] 创建环境失败
    pause
    exit /b 1
)

REM 激活环境
echo [2/4] 激活环境...
call conda activate yolo_screw

REM 安装 PyTorch nightly (RTX 5070 Blackwell 需要 CUDA 13.2)
echo [3/4] 安装 PyTorch nightly (CUDA 13.2)...
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
if %errorlevel% neq 0 (
    echo [警告] PyTorch nightly 安装失败，重试中...
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
)

REM 安装其他依赖
echo [4/4] 安装项目依赖...
pip install -r requirements.txt

echo.
echo ============================================================
echo 环境配置完成！
echo ============================================================
echo.
echo 使用方法：
echo   1. 激活环境: conda activate yolo_screw
echo   2. 开始训练: python train.py
echo   3. 模型评估: python evaluate.py
echo.
echo GPU 信息:
python -c "import torch; print(f'  CUDA 可用: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'  PyTorch: {torch.__version__}')"
echo.
pause
