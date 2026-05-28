@echo off
echo ============================================================
echo YOLOv8 螺丝缺陷检测 - GPU 训练启动
echo ============================================================
echo.

REM 激活conda环境
call conda activate yolo_screw

REM 检查GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

REM 开始训练
echo.
echo 开始训练...
python train.py --epochs 100 --batch 16

pause
