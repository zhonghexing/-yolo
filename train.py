"""
YOLOv8 钢材表面缺陷检测训练脚本
Steel Surface Defect Detection Training Script

基于 v1 (screw_defect-11) 最佳参数配置，mAP@0.5 = 0.761

使用方法：
    python train.py                          # 使用默认参数训练（yolov8s, 150 epochs）
    python train.py --epochs 50 --batch 16   # 自定义参数
    python train.py --model yolov8n.pt       # 使用更小的模型快速调试
    python train.py --resume path/to/last.pt # 恢复训练
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.absolute()


def get_dataset_config():
    """获取数据集配置文件路径，并验证其存在"""
    root = get_project_root()
    data_yaml = root / "datasets" / "neu_det" / "data.yaml"

    if not data_yaml.exists():
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        sys.exit(1)

    # 检查图片目录
    train_images = root / "datasets" / "neu_det" / "train" / "images"
    if not train_images.exists() or len(list(train_images.glob("*.jpg"))) == 0:
        print(f"[错误] 训练图片目录为空或不存在: {train_images}")
        sys.exit(1)

    print(f"[信息] 数据集配置: {data_yaml}")
    print(f"[信息] 训练图片数量: {len(list(train_images.glob('*.jpg')))}")

    return str(data_yaml)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 钢材表面缺陷检测训练")

    # 模型参数
    parser.add_argument(
        "--model", type=str, default="yolov8s.pt",
        help="预训练模型路径 (默认: yolov8s.pt)"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=150, help="训练轮数 (默认: 150)")
    parser.add_argument("--batch", type=int, default=32, help="批量大小 (默认: 32)")
    parser.add_argument("--imgsz", type=int, default=800, help="输入图片尺寸 (默认: 800)")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率 (默认: 0.001)")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率比例 (默认: 0.01)")

    # 数据增强
    parser.add_argument("--augment", action="store_true", default=True, help="启用数据增强")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic增强概率 (默认: 1.0)")
    parser.add_argument("--mixup", type=float, default=0.1, help="MixUp增强概率 (默认: 0.1)")

    # 训练控制
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值 (默认: 50)")
    parser.add_argument("--workers", type=int, default=2, help="数据加载线程数 (默认: 2)")
    parser.add_argument("--device", type=str, default="", help="训练设备 (默认: 自动选择)")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的检查点路径")

    # 输出控制
    parser.add_argument("--project", type=str, default="runs/train", help="项目保存目录")
    parser.add_argument("--name", type=str, default="screw_defect", help="实验名称")
    parser.add_argument("--exist-ok", action="store_true", help="覆盖已有实验目录")

    return parser.parse_args()


def train(args):
    """执行训练"""
    root = get_project_root()
    data_yaml = get_dataset_config()

    # 设置设备
    if args.device:
        device = args.device
    else:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
        if device == "0":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[信息] 使用 GPU: {gpu_name}")
        else:
            print("[信息] 未检测到 GPU，使用 CPU 训练（速度较慢）")

    # 加载模型
    print(f"\n[信息] 加载预训练模型: {args.model}")
    if args.resume:
        print(f"[信息] 从检查点恢复训练: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(args.model)

    # 训练参数配置（基于 v1 最佳参数，mAP@0.5=0.761）
    train_args = {
        # 数据集
        "data": data_yaml,

        # 训练参数
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,

        # 数据增强
        "augment": args.augment,
        "mosaic": args.mosaic,
        "mixup": args.mixup,

        # 训练控制
        "patience": args.patience,
        "workers": args.workers,
        "device": device,

        # 输出
        "project": str(root / args.project),
        "name": args.name,
        "exist_ok": args.exist_ok,

        # 优化器配置
        "optimizer": "AdamW",
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # 损失函数权重
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,

        # 学习率调度
        "cos_lr": True,

        # 数据增强
        "close_mosaic": 10,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "erasing": 0.4,

        # 训练控制
        "amp": True,
        "cache": "disk",
        "save": True,
        "save_period": -1,
        "plots": True,
        "verbose": True,
        "seed": 42,
        "deterministic": True,
    }

    # 打印训练配置
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # 开始训练
    start_time = datetime.now()
    print(f"[信息] 训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = model.train(**train_args)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n[信息] 训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[信息] 总训练时长: {duration}")

    # 输出最佳模型路径
    save_dir = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  最佳模型: {best_model}")
    print(f"  最后模型: {last_model}")
    print(f"  训练日志: {save_dir}")
    print("=" * 60)

    return results


def main():
    """主函数"""
    print("=" * 60)
    print("YOLOv8 钢材表面缺陷检测训练")
    print("=" * 60)

    args = parse_args()

    try:
        results = train(args)
        print("\n[成功] 训练完成!")
        return 0
    except KeyboardInterrupt:
        print("\n[警告] 训练被用户中断")
        return 1
    except Exception as e:
        print(f"\n[错误] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
