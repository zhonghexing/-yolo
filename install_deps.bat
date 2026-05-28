@echo off
echo ============================================================
echo 安装项目依赖（不包含 PyTorch）
echo ============================================================
echo.

call conda activate yolo_screw

echo [信息] PyTorch 已安装，跳过
python -c "import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo.

echo [1/2] 安装 ultralytics...
pip install ultralytics>=8.0.0 --no-deps
pip install matplotlib numpy opencv-python pillow pyyaml requests scipy tqdm

echo.
echo [2/2] 安装其他依赖...
pip install pandas scikit-learn seaborn tensorboard

echo.
echo ============================================================
echo 安装完成！
echo ============================================================
python -c "import torch; import ultralytics; print(f'torch {torch.__version__}'); print(f'ultralytics {ultralytics.__version__}')"
pause
