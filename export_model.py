"""
YOLOv8 螺丝缺陷检测模型导出脚本
Screw Defect Detection Model Export Script

使用方法：
    python export_model.py                              # 导出ONNX格式
    python export_model.py --format onnx                # 指定ONNX格式
    python export_model.py --format torchscript         # 导出TorchScript格式
    python export_model.py --all                        # 导出所有支持的格式
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


def find_best_model():
    """自动查找最佳模型"""
    root = get_project_root()
    runs_dir = root / "runs" / "train"

    if not runs_dir.exists():
        return None

    best_models = list(runs_dir.glob("*/weights/best.pt"))
    if best_models:
        best_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return best_models[0]

    return None


# 支持的导出格式
EXPORT_FORMATS = {
    "onnx": {
        "suffix": ".onnx",
        "description": "ONNX (通用格式，支持多种推理引擎)",
        "dynamic": True,
    },
    "torchscript": {
        "suffix": ".torchscript",
        "description": "TorchScript (PyTorch原生格式)",
        "dynamic": False,
    },
    "openvino": {
        "suffix": "_openvino_model/",
        "description": "OpenVINO (Intel硬件优化)",
        "dynamic": False,
    },
    "engine": {
        "suffix": ".engine",
        "description": "TensorRT (NVIDIA GPU优化)",
        "dynamic": True,
    },
    "coreml": {
        "suffix": ".mlpackage",
        "description": "CoreML (Apple设备)",
        "dynamic": False,
    },
    "saved_model": {
        "suffix": "_saved_model/",
        "description": "TensorFlow SavedModel",
        "dynamic": False,
    },
    "pb": {
        "suffix": ".pb",
        "description": "TensorFlow GraphDef",
        "dynamic": False,
    },
    "tflite": {
        "suffix": ".tflite",
        "description": "TensorFlow Lite (移动端)",
        "dynamic": False,
    },
    "edgetpu": {
        "suffix": "_edgetpu.tflite",
        "description": "TF Lite Edge TPU (Coral设备)",
        "dynamic": False,
    },
    "tfjs": {
        "suffix": "_web_model/",
        "description": "TensorFlow.js (Web端)",
        "dynamic": False,
    },
    "paddle": {
        "suffix": "_paddle_model/",
        "description": "PaddlePaddle",
        "dynamic": False,
    },
    "ncnn": {
        "suffix": "_ncnn_model/",
        "description": "NCNN (移动端轻量级)",
        "dynamic": False,
    },
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 模型导出")

    parser.add_argument(
        "--model", type=str, default="",
        help="模型路径 (默认: 自动查找最新的best.pt)"
    )
    parser.add_argument(
        "--format", type=str, default="onnx",
        choices=list(EXPORT_FORMATS.keys()),
        help="导出格式 (默认: onnx)"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸 (默认: 640)")
    parser.add_argument("--batch", type=int, default=1, help="批量大小 (默认: 1)")
    parser.add_argument("--half", action="store_true", help="FP16半精度导出")
    parser.add_argument("--dynamic", action="store_true", help="动态输入尺寸")
    parser.add_argument("--simplify", action="store_true", default=True, help="简化ONNX模型")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset版本 (默认: 17)")
    parser.add_argument(
        "--output", type=str, default="runs/export",
        help="导出结果保存目录"
    )
    parser.add_argument("--all", action="store_true", help="导出所有支持的格式")

    return parser.parse_args()


def export_model(model_path, fmt, args, output_dir):
    """导出单个格式"""
    print(f"\n[信息] 加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 导出参数
    export_args = {
        "format": fmt,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "half": args.half,
        "dynamic": args.dynamic and EXPORT_FORMATS[fmt]["dynamic"],
        "opset": args.opset if fmt == "onnx" else None,
        "simplify": args.simplify and fmt == "onnx",
    }

    # 过滤None值
    export_args = {k: v for k, v in export_args.items() if v is not None}

    print(f"[信息] 导出格式: {fmt}")
    print(f"[信息] 格式说明: {EXPORT_FORMATS[fmt]['description']}")
    print(f"[信息] 导出参数: {export_args}")

    # 执行导出
    export_path = model.export(**export_args)

    print(f"[成功] 模型已导出: {export_path}")

    # 获取文件大小
    export_path = Path(export_path)
    if export_path.is_file():
        size_mb = export_path.stat().st_size / (1024 * 1024)
        print(f"[信息] 文件大小: {size_mb:.2f} MB")
    elif export_path.is_dir():
        total_size = sum(f.stat().st_size for f in export_path.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"[信息] 目录大小: {size_mb:.2f} MB")

    return export_path


def main():
    """主函数"""
    print("=" * 60)
    print("YOLOv8 螺丝缺陷检测模型导出")
    print("=" * 60)

    args = parse_args()
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

    # 创建输出目录
    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[信息] 源模型: {model_path}")
    print(f"[信息] 输出目录: {output_dir}")

    # 记录开始时间
    start_time = datetime.now()

    try:
        if args.all:
            # 导出所有格式
            print("\n[信息] 导出所有支持的格式...")
            results = {}
            for fmt in EXPORT_FORMATS:
                try:
                    export_path = export_model(model_path, fmt, args, output_dir)
                    results[fmt] = {"status": "success", "path": str(export_path)}
                except Exception as e:
                    print(f"[警告] 导出 {fmt} 格式失败: {e}")
                    results[fmt] = {"status": "failed", "error": str(e)}

            # 打印汇总
            print("\n" + "=" * 60)
            print("导出结果汇总")
            print("=" * 60)
            for fmt, result in results.items():
                status = "✓" if result["status"] == "success" else "✗"
                print(f"  {status} {fmt}: {result['status']}")
            print("=" * 60)
        else:
            # 导出指定格式
            export_model(model_path, args.format, args, output_dir)

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n[信息] 导出总耗时: {duration}")
        print("[成功] 模型导出完成!")

        return 0

    except KeyboardInterrupt:
        print("\n[警告] 导出被用户中断")
        return 1
    except Exception as e:
        print(f"\n[错误] 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
