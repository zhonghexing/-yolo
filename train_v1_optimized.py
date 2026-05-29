"""
基于 v1 的优化训练脚本
v1 配置: yolov8s, imgsz=800, batch=32, mAP50=0.761

优化策略：
1. 降低学习率 (0.001 → 0.0005) - 更稳定收敛
2. 增加训练轮数 (150 → 200) - 配合早停
3. 优化数据增强 - 针对弱类别 (crazing, rolled-in_scale)
4. 更早关闭 Mosaic (10 → 15) - 让模型学习更多真实特征

使用方法：
    python train_v1_optimized.py
"""

import os
import sys
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

    train_images = root / "datasets" / "neu_det" / "train" / "images"
    if not train_images.exists() or len(list(train_images.glob("*.jpg"))) == 0:
        print(f"[错误] 训练图片目录为空或不存在: {train_images}")
        sys.exit(1)

    print(f"[信息] 数据集配置: {data_yaml}")
    print(f"[信息] 训练图片数量: {len(list(train_images.glob('*.jpg')))}")

    return str(data_yaml)


def train():
    """执行训练"""
    root = get_project_root()
    data_yaml = get_dataset_config()

    # 设备选择
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[信息] 使用 GPU: {gpu_name}")
    else:
        print("[信息] 未检测到 GPU，使用 CPU 训练")

    # ============================================================
    # 基于 v1 的优化配置
    # ============================================================
    print("\n[信息] 加载预训练模型: yolov8s.pt")
    model = YOLO("yolov8s.pt")

    # 训练参数
    train_args = {
        # 数据集
        "data": data_yaml,

        # 模型配置 - 与 v1 保持一致
        "model": "yolov8s.pt",
        "imgsz": 800,              # 与 v1 一致
        "batch": 32,               # 与 v1 一致

        # 训练轮数 - 增加
        "epochs": 200,             # 从 150 增加到 200
        "patience": 50,            # 早停耐心值

        # 学习率 - 降低
        "lr0": 0.0005,             # 从 0.001 降低到 0.0005
        "lrf": 0.01,               # 最终学习率比例
        "cos_lr": True,            # 余弦退火

        # 优化器
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

        # ============================================================
        # 数据增强优化 - 针对弱类别
        # ============================================================
        "augment": True,
        "mosaic": 1.0,             # Mosaic 增强
        "close_mosaic": 15,        # 最后 15 轮关闭 Mosaic（比 v1 的 10 更多）
        "mixup": 0.15,             # 从 0.1 增加到 0.15
        "copy_paste": 0.1,         # 新增：复制粘贴增强（帮助小缺陷）

        # 几何变换
        "degrees": 10.0,           # 旋转角度
        "translate": 0.1,          # 平移
        "scale": 0.5,              # 缩放
        "shear": 2.0,              # 新增：剪切变换
        "flipud": 0.3,             # 新增：上下翻转（帮助 crazing）
        "fliplr": 0.5,             # 左右翻转

        # 颜色增强
        "hsv_h": 0.015,            # 色调
        "hsv_s": 0.7,              # 饱和度
        "hsv_v": 0.4,              # 亮度
        "erasing": 0.4,            # 随机擦除

        # 训练控制
        "amp": True,               # 混合精度
        "cache": "disk",           # 磁盘缓存
        "workers": 2,              # 数据加载线程
        "device": device,

        # 输出
        "project": str(root / "runs" / "train"),
        "name": "screw_defect_v1_optimized",
        "exist_ok": True,
        "save": True,
        "save_period": -1,
        "plots": True,
        "verbose": True,
        "seed": 42,
        "deterministic": True,
    }

    # 打印训练配置
    print("\n" + "=" * 60)
    print("基于 v1 的优化训练配置")
    print("=" * 60)
    print("\n[与 v1 对比]")
    print("  模型:     yolov8s (不变)")
    print("  imgsz:    800 (不变)")
    print("  batch:    32 (不变)")
    print("  epochs:   150 → 200")
    print("  lr0:      0.001 → 0.0005")
    print("  mixup:    0.1 → 0.15")
    print("  新增:     copy_paste=0.1, shear=2.0, flipud=0.3")
    print("  close_mosaic: 10 → 15")
    print("\n" + "=" * 60)
    print("\n完整配置:")
    for key, value in train_args.items():
        if key != "data":
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

    # 输出结果
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
    print("基于 v1 的优化训练")
    print("=" * 60)
    print("\n目标: 在 v1 基础上进一步提升 mAP50")
    print("v1 基准: mAP50=0.761, mAP50-95=0.421\n")

    try:
        results = train()
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
