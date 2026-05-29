"""
模型对比评估脚本
对比 v1 和优化版的 mAP 指标

使用方法：
    python compare_models.py
    python compare_models.py --model runs/train/screw_defect_v1_optimized/weights/best.pt
"""

import sys
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(model_path, split="val"):
    """评估单个模型"""
    print(f"\n[信息] 评估模型: {model_path}")

    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return None

    model = YOLO(model_path)
    results = model.val(split=split, verbose=False)

    return {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "per_class": {
            name: {
                "AP50": results.box.ap50[i],
                "AP50-95": results.box.ap[i],
            }
            for i, name in enumerate(results.names.values())
        },
    }


def print_comparison(v1_results, optimized_results):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("模型对比结果")
    print("=" * 70)

    # 整体指标
    print("\n[整体指标]")
    print(f"{'指标':<15} {'v1':<15} {'优化版':<15} {'提升':<15}")
    print("-" * 60)

    for metric in ["mAP50", "mAP50-95", "precision", "recall"]:
        v1_val = v1_results[metric]
        opt_val = optimized_results[metric]
        diff = opt_val - v1_val
        sign = "+" if diff >= 0 else ""
        print(f"{metric:<15} {v1_val:.4f}{'':<9} {opt_val:.4f}{'':<9} {sign}{diff:.4f}")

    # 各类别 AP50
    print("\n[各类别 AP@0.5]")
    print(f"{'类别':<20} {'v1':<15} {'优化版':<15} {'提升':<15}")
    print("-" * 70)

    for class_name in v1_results["per_class"]:
        v1_ap = v1_results["per_class"][class_name]["AP50"]
        opt_ap = optimized_results["per_class"][class_name]["AP50"]
        diff = opt_ap - v1_ap
        sign = "+" if diff >= 0 else ""
        print(f"{class_name:<20} {v1_ap:.4f}{'':<9} {opt_ap:.4f}{'':<9} {sign}{diff:.4f}")

    print("\n" + "=" * 70)


def main():
    """主函数"""
    # v1 模型路径
    v1_model = "runs/train/screw_defect-11/weights/best.pt"

    # 优化版模型路径
    if len(sys.argv) > 2 and sys.argv[1] == "--model":
        optimized_model = sys.argv[2]
    else:
        optimized_model = "runs/train/screw_defect_v1_optimized/weights/best.pt"

    print("=" * 70)
    print("模型对比评估")
    print("=" * 70)

    # 评估 v1
    v1_results = evaluate_model(v1_model)
    if v1_results is None:
        print("[错误] 无法评估 v1 模型")
        return 1

    # 评估优化版
    opt_results = evaluate_model(optimized_model)
    if opt_results is None:
        print("[错误] 无法评估优化版模型")
        return 1

    # 打印对比
    print_comparison(v1_results, opt_results)

    # 判断是否提升
    if opt_results["mAP50"] > v1_results["mAP50"]:
        print("\n[结论] 优化版 mAP50 提升!")
    else:
        print("\n[结论] 优化版 mAP50 未提升，需要进一步调整")

    return 0


if __name__ == "__main__":
    sys.exit(main())
