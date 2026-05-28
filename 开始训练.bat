@echo off
title YOLOv8 训练
cd /d D:\yolo
call conda activate yolo_screw
python train.py --epochs 100 --batch 16
pause
