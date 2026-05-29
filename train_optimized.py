"""
优化版训练脚本 - 使用更大模型和优化参数
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
    """获取数据集配置文件路径"""
    root = get_project_root()
    data_yaml = root / "datasets" / "neu_det" / "data.yaml"

    if not data_yaml.exists():
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        sys.exit(1)

    return str(data_yaml)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="优化版 YOLOv8 训练")

    # 模型选择
    parser.add_argument(
        "--model", type=str, default="yolov8l.pt",
        help="预训练模型 (默认: yolov8l.pt，可选: yolov8m.pt, yolov8x.pt)"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数 (默认: 300)")
    parser.add_argument("--batch", type=int, default=8, help="批量大小 (默认: 8)")
    parser.add_argument("--imgsz", type=int, default=1024, help="输入图片尺寸 (默认: 1024)")
    parser.add_argument("--lr0", type=float, default=0.0005, help="初始学习率 (默认: 0.0005)")
    parser.add_argument("--lrf", type=float, default=0.005, help="最终学习率比例 (默认: 0.005)")

    # 训练控制
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值 (默认: 50)")
    parser.add_argument("--workers", type=int, default=2, help="数据加载线程数 (默认: 2)")
    parser.add_argument("--device", type=str, default="", help="训练设备")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的检查点路径")

    # 输出控制
    parser.add_argument("--project", type=str, default="runs/train", help="项目保存目录")
    parser.add_argument("--name", type=str, default="screw_defect_v3_optimized", help="实验名称")
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
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[信息] 使用 GPU: {gpu_name} ({gpu_mem:.1f}GB)")

            # 根据显存调整 batch size
            if gpu_mem < 8:
                args.batch = max(4, args.batch // 2)
                print(f"[信息] 显存较小，调整 batch_size 为 {args.batch}")
            elif gpu_mem >= 16:
                args.batch = min(16, args.batch * 2)
                print(f"[信息] 显存充足，调整 batch_size 为 {args.batch}")
        else:
            print("[信息] 未检测到 GPU，使用 CPU 训练")

    # 加载模型
    print(f"\n[信息] 加载预训练模型: {args.model}")
    if args.resume:
        print(f"[信息] 从检查点恢复训练: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(args.model)

    # 优化后的训练参数配置
    train_args = {
        # 数据集
        "data": data_yaml,

        # 训练参数
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,

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
        "weight_decay": 0.001,          # 增加权重衰减
        "warmup_epochs": 5.0,           # 增加预热轮数
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # 损失函数权重
        "box": 10.0,                    # 增加边界框损失权重
        "cls": 0.5,
        "dfl": 1.5,

        # 学习率调度
        "cos_lr": True,
        "close_mosaic": 15,             # 最后15轮关闭Mosaic

        # 数据增强 - 更激进的增强策略
        "augment": True,
        "mosaic": 1.0,
        "mixup": 0.15,                  # 增加 MixUp
        "copy_paste": 0.1,              # 添加复制粘贴增强
        "degrees": 15.0,                # 增加旋转角度
        "translate": 0.15,              # 增加平移
        "scale": 0.6,                   # 增加缩放范围
        "shear": 5.0,                   # 添加剪切变换
        "perspective": 0.001,           # 添加透视变换
        "flipud": 0.3,                  # 添加上下翻转
        "fliplr": 0.5,
        "hsv_h": 0.02,                  # 增加色调增强
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "erasing": 0.3,                 # 添加随机擦除

        # 训练控制
        "amp": True,
        "cache": "disk",
        "save": True,
        "save_period": -1,
        "plots": True,
        "verbose": True,
        "seed": 42,

        # 标签平滑
        "label_smoothing": 0.01,        # 添加标签平滑
    }

    # 打印训练配置
    print("\n" + "=" * 60)
    print("优化版训练配置")
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
    print("优化版 YOLOv8 训练脚本")
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
