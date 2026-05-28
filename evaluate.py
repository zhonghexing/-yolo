"""
YOLOv8 螺丝缺陷检测模型评估脚本
Screw Defect Detection Model Evaluation Script

使用方法：
    python evaluate.py                                    # 评估默认模型
    python evaluate.py --model runs/train/screw_defect/weights/best.pt  # 指定模型
    python evaluate.py --split test                       # 在测试集上评估
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from ultralytics import YOLO

from constants import CLASS_NAMES, CLASS_NAMES_CN


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.absolute()


def find_best_model():
    """自动查找最佳模型"""
    root = get_project_root()
    runs_dir = root / "runs" / "train"

    if not runs_dir.exists():
        return None

    # 查找所有实验目录中的best.pt
    best_models = list(runs_dir.glob("*/weights/best.pt"))
    if best_models:
        # 按修改时间排序，返回最新的
        best_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return best_models[0]

    return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 模型评估")

    parser.add_argument(
        "--model", type=str, default="",
        help="模型路径 (默认: 自动查找最新的best.pt)"
    )
    parser.add_argument(
        "--data", type=str, default="",
        help="数据集配置文件路径 (默认: datasets/neu_det/data.yaml)"
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["val", "test"],
        help="评估数据集分割 (默认: val)"
    )
    parser.add_argument("--batch", type=int, default=8, help="批量大小 (默认: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU阈值 (默认: 0.6)")
    parser.add_argument("--device", type=str, default="", help="推理设备")
    parser.add_argument(
        "--output", type=str, default="runs/eval",
        help="评估结果保存目录"
    )

    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=CLASS_NAMES):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 绝对数值混淆矩阵
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Count)', fontsize=14)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # 归一化混淆矩阵
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[信息] 混淆矩阵已保存: {save_path}")


def generate_report(metrics, output_dir, split="val"):
    """生成评估报告"""
    report = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": split,
        "overall_metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "per_class_metrics": {}
    }

    # 每个类别的指标
    for i, name in enumerate(CLASS_NAMES):
        if i < len(metrics.box.ap50):
            report["per_class_metrics"][name] = {
                "AP50": float(metrics.box.ap50[i]),
                "AP50-95": float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0,
            }

    # 保存JSON报告
    json_path = output_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[信息] 评估报告已保存: {json_path}")

    # 生成文本报告
    txt_path = output_dir / "evaluation_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("螺丝缺陷检测模型评估报告\n")
        f.write("Screw Defect Detection Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"评估时间: {report['evaluation_time']}\n")
        f.write(f"评估数据集: {split}\n\n")

        f.write("-" * 40 + "\n")
        f.write("整体指标 (Overall Metrics)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  mAP@0.5:      {report['overall_metrics']['mAP50']:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {report['overall_metrics']['mAP50-95']:.4f}\n")
        f.write(f"  Precision:    {report['overall_metrics']['precision']:.4f}\n")
        f.write(f"  Recall:       {report['overall_metrics']['recall']:.4f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("各类别指标 (Per-Class Metrics)\n")
        f.write("-" * 40 + "\n")
        for name, m in report["per_class_metrics"].items():
            f.write(f"\n  {name}:\n")
            f.write(f"    AP@0.5:      {m['AP50']:.4f}\n")
            f.write(f"    AP@0.5:0.95: {m['AP50-95']:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"[信息] 评估报告已保存: {txt_path}")

    return report


def print_metrics_summary(report):
    """打印指标摘要"""
    print("\n" + "=" * 50)
    print("评估结果摘要")
    print("=" * 50)

    om = report["overall_metrics"]
    print(f"  mAP@0.5:      {om['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {om['mAP50-95']:.4f}")
    print(f"  Precision:    {om['precision']:.4f}")
    print(f"  Recall:       {om['recall']:.4f}")

    print("\n各类别 AP@0.5:")
    for name, m in report["per_class_metrics"].items():
        print(f"  {name:20s}: {m['AP50']:.4f}")

    print("=" * 50)


def evaluate(args):
    """执行评估"""
    root = get_project_root()

    # 查找模型
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_best_model()
        if model_path is None:
            print("[错误] 未找到训练好的模型，请使用 --model 参数指定模型路径")
            print("       或先运行 train.py 进行训练")
            sys.exit(1)

    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        sys.exit(1)

    print(f"[信息] 加载模型: {model_path}")

    # 数据集配置
    if args.data:
        data_yaml = args.data
    else:
        data_yaml = str(root / "datasets" / "neu_det" / "data.yaml")

    if not Path(data_yaml).exists():
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        sys.exit(1)

    print(f"[信息] 数据集配置: {data_yaml}")
    print(f"[信息] 评估分割: {args.split}")

    # 创建输出目录
    output_dir = root / args.output / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = YOLO(str(model_path))

    # 设置设备
    if args.device:
        device = args.device
    else:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"[信息] 使用设备: {device}")

    # 执行评估
    print("\n[信息] 开始评估...")
    metrics = model.val(
        data=data_yaml,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        plots=True,
        verbose=True,
    )

    # 生成报告
    report = generate_report(metrics, output_dir, args.split)
    print_metrics_summary(report)

    # 尝试生成混淆矩阵（需要预测结果）
    try:
        print("\n[信息] 生成混淆矩阵...")
        # 获取验证集图片路径
        data_root = root / "datasets" / "neu_det"
        split_dir = data_root / args.split / "images"

        if split_dir.exists():
            image_files = list(split_dir.glob("*.jpg"))
            if image_files:
                y_true = []
                y_pred = []

                for img_path in image_files:
                    # 从文件名推断真实类别
                    filename = img_path.stem
                    for i, cls_name in enumerate(CLASS_NAMES):
                        if cls_name in filename:
                            y_true.append(i)
                            break

                    # 模型预测
                    results = model.predict(
                        str(img_path),
                        conf=args.conf,
                        verbose=False
                    )
                    if results[0].boxes.cls.numel() > 0:
                        # 取置信度最高的预测
                        best_cls = int(results[0].boxes.cls[results[0].boxes.conf.argmax()])
                        y_pred.append(best_cls)
                    else:
                        y_pred.append(-1)  # 无检测结果

                # 过滤有效预测
                valid_idx = [i for i in range(len(y_pred)) if y_pred[i] != -1]
                if valid_idx:
                    y_true_valid = [y_true[i] for i in valid_idx]
                    y_pred_valid = [y_pred[i] for i in valid_idx]

                    cm_path = output_dir / "confusion_matrix.png"
                    plot_confusion_matrix(y_true_valid, y_pred_valid, cm_path)
    except Exception as e:
        print(f"[警告] 生成混淆矩阵时出错: {e}")

    print(f"\n[成功] 评估完成! 结果保存在: {output_dir}")
    return report


def main():
    """主函数"""
    print("=" * 60)
    print("YOLOv8 螺丝缺陷检测模型评估")
    print("=" * 60)

    args = parse_args()

    try:
        report = evaluate(args)
        return 0
    except KeyboardInterrupt:
        print("\n[警告] 评估被用户中断")
        return 1
    except Exception as e:
        print(f"\n[错误] 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
