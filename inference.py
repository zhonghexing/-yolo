"""
螺丝缺陷检测推理引擎核心模块
Screw Defect Detection Inference Engine

功能：
    - 加载 YOLOv8 模型（支持 GPU/CPU 自动切换）
    - 单张图片检测
    - 批量图片检测
    - 结果可视化
    - 检测结果数据结构化输出

使用方法：
    # 单张检测
    python inference.py --image path/to/image.jpg

    # 批量检测
    python inference.py --dir path/to/images/

    # 指定模型
    python inference.py --model runs/train/screw_defect/weights/best.pt --image test.jpg
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from constants import CLASS_NAMES, CLASS_NAMES_CN, CLASS_COLORS_BGR as CLASS_COLORS


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class SingleDetection:
    """单个检测框的结果"""
    class_id: int                       # 类别 ID
    class_name: str                     # 类别英文名
    class_name_cn: str                  # 类别中文名
    confidence: float                   # 置信度 [0, 1]
    bbox: tuple                         # 边界框 (x1, y1, x2, y2) 像素坐标
    is_defect: bool                     # 是否为缺陷

    def __str__(self):
        status = "缺陷" if self.is_defect else "合格"
        return (
            f"[{status}] {self.class_name_cn}({self.class_name}) "
            f"置信度={self.confidence:.2%} "
            f"bbox=({self.bbox[0]:.0f},{self.bbox[1]:.0f},{self.bbox[2]:.0f},{self.bbox[3]:.0f})"
        )


@dataclass
class ImageDetectionResult:
    """单张图片的完整检测结果"""
    image_path: str                     # 图片路径
    detections: List[SingleDetection]   # 所有检测结果列表
    inference_time_ms: float            # 推理耗时（毫秒）
    image_shape: tuple                  # 图片尺寸 (H, W, C)
    overall_verdict: str = ""           # 整体判定结果

    def __post_init__(self):
        """初始化后自动计算整体判定（NEU-DET 所有类别均为缺陷）"""
        if not self.overall_verdict:
            if len(self.detections) == 0:
                self.overall_verdict = "合格 - 未检出缺陷"
            else:
                defect_types = set(
                    d.class_name_cn for d in self.detections
                )
                self.overall_verdict = f"不合格 - 检出缺陷: {', '.join(defect_types)}"

    @property
    def has_defect(self) -> bool:
        """是否存在缺陷"""
        return any(d.is_defect for d in self.detections)

    @property
    def defect_count(self) -> int:
        """缺陷数量"""
        return sum(1 for d in self.detections if d.is_defect)

    @property
    def max_confidence(self) -> float:
        """最高置信度"""
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)


# ============================================================
# 推理引擎核心类
# ============================================================

class ScrewDefectDetector:
    """
    螺丝缺陷检测器

    基于 YOLOv8 模型的推理引擎，支持 GPU/CPU 自动切换，
    提供单张和批量检测能力。

    参数：
        model_path: str - 模型权重文件路径 (.pt)
        conf_threshold: float - 置信度阈值，默认 0.25
        iou_threshold: float - NMS IoU 阈值，默认 0.45
        device: str - 推理设备，'' 表示自动选择
        img_size: int - 输入图片尺寸，默认 640
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "",
        img_size: int = 800,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        # 自动选择设备
        self.device = self._resolve_device(device)
        self.use_fp16 = self.device != "cpu"  # GPU 时启用 FP16
        print(f"[推理引擎] 使用设备: {self.device}, FP16: {self.use_fp16}")

        # 加载模型
        self.model = self._load_model()

        # 预热推理（首次推理通常较慢，提前完成）
        self._warmup()

    def _resolve_device(self, device: str) -> str:
        """
        解析并选择推理设备

        优先级：用户指定 > GPU (CUDA) > CPU
        """
        if device:
            return device

        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_cap = torch.cuda.get_device_capability(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

                print(f"[推理引擎] 检测到 GPU: {gpu_name} (CC {gpu_cap[0]}.{gpu_cap[1]}, {gpu_mem:.1f} GB)")

                # RTX 5070/5090 (sm_120) 需要 PyTorch nightly
                if gpu_cap[0] >= 12:
                    print(f"[推理引擎] 注意: 该 GPU 架构需要 PyTorch nightly 版本才能正常工作")

                return "0"
            except Exception as e:
                print(f"[推理引擎] GPU 检测异常: {e}，使用 CPU")
                return "cpu"
        else:
            print("[推理引擎] 未检测到 CUDA GPU，使用 CPU 推理")
            return "cpu"

    def _load_model(self) -> YOLO:
        """加载 YOLOv8 模型"""
        if not self.model_path.exists():
            # 如果指定路径不存在，尝试在 runs/train 目录下查找
            fallback = self._find_best_model()
            if fallback:
                print(f"[推理引擎] 指定模型不存在，使用自动查找到的模型: {fallback}")
                self.model_path = fallback
            else:
                print(f"[推理引擎] 警告: 模型文件不存在: {self.model_path}")
                print("[推理引擎] 将使用 YOLOv8n 预训练模型（未针对螺丝缺陷微调）")
                self.model_path = Path("yolov8n.pt")

        print(f"[推理引擎] 加载模型: {self.model_path}")
        model = YOLO(str(self.model_path))

        # YOLOv8 的 predict 方法会自动处理 device 参数，不需要手动 to()
        return model

    def _find_best_model(self) -> Optional[Path]:
        """在项目目录中查找最佳训练模型"""
        root = Path(__file__).parent.absolute()
        runs_dir = root / "runs" / "train"

        if not runs_dir.exists():
            return None

        best_models = list(runs_dir.glob("*/weights/best.pt"))
        if best_models:
            # 按修改时间排序，返回最新的
            best_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return best_models[0]

        return None

    def _warmup(self):
        """预热推理引擎，确保首次推理不会异常缓慢"""
        print("[推理引擎] 预热中...")
        dummy_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        try:
            _ = self.model.predict(
                dummy_img,
                conf=0.1,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                half=self.use_fp16,
                verbose=False,
            )
            print("[推理引擎] 预热完成")
        except Exception as e:
            # FP16 失败时回退到 FP32
            if self.use_fp16:
                print(f"[推理引擎] FP16 预热失败，回退到 FP32: {e}")
                self.use_fp16 = False
                try:
                    _ = self.model.predict(
                        dummy_img, conf=0.1, iou=self.iou_threshold,
                        imgsz=self.img_size, device=self.device, verbose=False,
                    )
                    print("[推理引擎] FP32 预热完成")
                except Exception as e2:
                    print(f"[推理引擎] 预热警告: {e2}")
            else:
                print(f"[推理引擎] 预热警告（不影响后续推理）: {e}")

    def _parse_results(self, result, image_path: str, image_shape: tuple) -> ImageDetectionResult:
        """
        将 YOLOv8 单张原始输出解析为结构化检测结果

        参数：
            result: ultralytics 单张推理结果对象 (Results)
            image_path: 原始图片路径
            image_shape: 图片尺寸 (H, W, C)

        返回：
            ImageDetectionResult 对象
        """
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()    # (N, 4) 像素坐标
            confs = result.boxes.conf.cpu().numpy()    # (N,) 置信度
            clses = result.boxes.cls.cpu().numpy().astype(int)  # (N,) 类别ID

            for i in range(len(boxes)):
                cls_id = clses[i]
                # 防止类别越界
                if 0 <= cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                else:
                    cls_name = f"unknown_{cls_id}"

                cls_name_cn = CLASS_NAMES_CN.get(cls_name, cls_name)
                is_defect = True  # NEU-DET 所有类别均为缺陷

                det = SingleDetection(
                    class_id=cls_id,
                    class_name=cls_name,
                    class_name_cn=cls_name_cn,
                    confidence=float(confs[i]),
                    bbox=tuple(boxes[i].tolist()),
                    is_defect=is_defect,
                )
                detections.append(det)

        # 按置信度降序排列
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return ImageDetectionResult(
            image_path=image_path,
            detections=detections,
            inference_time_ms=0.0,  # 由调用方填充
            image_shape=image_shape,
        )

    def detect_single(self, image_input: Union[str, Path, np.ndarray]) -> ImageDetectionResult:
        """
        单张图片检测

        参数：
            image_input: 图片路径(str/Path) 或 numpy 数组 (BGR格式)

        返回：
            ImageDetectionResult 检测结果对象
        """
        # 读取图片
        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image_path = "<numpy_array>"
            img = image_input
        else:
            raise TypeError(f"不支持的输入类型: {type(image_input)}")

        image_shape = img.shape

        # 执行推理并计时
        t_start = time.perf_counter()
        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            half=self.use_fp16,
            verbose=False,
        )
        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000

        # 解析结果
        result = self._parse_results(results[0], str(image_input), image_shape)
        result.inference_time_ms = inference_time_ms

        return result

    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8,
    ) -> List[ImageDetectionResult]:
        """
        批量图片检测

        利用 YOLOv8 的批量推理能力提升吞吐量，
        满足比赛要求（20个样本 180 秒内完成）。

        参数：
            image_paths: 图片路径列表
            batch_size: 批量大小，默认 8

        返回：
            List[ImageDetectionResult] 检测结果列表
        """
        results_list = []
        total_images = len(image_paths)

        print(f"[推理引擎] 开始批量检测: {total_images} 张图片, batch_size={batch_size}")

        t_total_start = time.perf_counter()

        # 分批处理
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_paths = image_paths[batch_start:batch_end]

            # 读取本批次图片
            batch_images = []
            valid_paths = []
            for p in batch_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    batch_images.append(img)
                    valid_paths.append(str(p))
                else:
                    print(f"[推理引擎] 警告: 无法读取图片 {p}, 跳过")

            if not batch_images:
                continue

            # 批量推理
            t_batch_start = time.perf_counter()
            raw_results = self.model.predict(
                batch_images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                half=self.use_fp16,
                verbose=False,
                stream=True,  # 流式输出节省内存
            )
            t_batch_end = time.perf_counter()
            batch_time_ms = (t_batch_end - t_batch_start) * 1000
            per_image_ms = batch_time_ms / len(batch_images)

            # 解析每张图片的结果
            for idx, raw_result in enumerate(raw_results):
                det_result = self._parse_results(
                    raw_result, valid_paths[idx], batch_images[idx].shape
                )
                det_result.inference_time_ms = per_image_ms
                results_list.append(det_result)

            progress = min(batch_end, total_images)
            print(
                f"[推理引擎] 进度: {progress}/{total_images} "
                f"| 本批次耗时: {batch_time_ms:.1f}ms "
                f"| 单张: {per_image_ms:.1f}ms"
            )

        t_total_end = time.perf_counter()
        total_time_s = t_total_end - t_total_start
        avg_ms = (total_time_s * 1000) / total_images if total_images > 0 else 0

        print(
            f"[推理引擎] 批量检测完成: {len(results_list)} 张, "
            f"总耗时 {total_time_s:.2f}s, 平均 {avg_ms:.1f}ms/张"
        )

        return results_list

    def detect_image_dir(
        self,
        image_dir: Union[str, Path],
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
        batch_size: int = 8,
    ) -> List[ImageDetectionResult]:
        """
        检测目录下的所有图片

        参数：
            image_dir: 图片目录路径
            extensions: 支持的图片扩展名
            batch_size: 批量大小

        返回：
            List[ImageDetectionResult] 检测结果列表
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"目录不存在: {image_dir}")

        # 收集所有图片文件
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))

        # 去重并排序
        image_paths = sorted(set(image_paths))

        if not image_paths:
            print(f"[推理引擎] 目录中未找到图片: {image_dir}")
            return []

        print(f"[推理引擎] 找到 {len(image_paths)} 张图片")

        return self.detect_batch(image_paths, batch_size=batch_size)

    def visualize_result(
        self,
        image_input: Union[str, Path, np.ndarray],
        result: ImageDetectionResult,
        show: bool = False,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        在图片上绘制检测结果

        绘制规则：
            - 绿色框: normal (合格)
            - 红色框: 各类缺陷
            - 标签显示: 类别中文名 + 置信度

        参数：
            image_input: 原始图片路径或 numpy 数组
            result: 检测结果对象
            show: 是否弹窗显示
            save_path: 保存路径，None 表示不保存

        返回：
            绘制后的图片 (numpy 数组)
        """
        # 读取图片
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {image_input}")
        else:
            img = image_input.copy()

        # 绘制每个检测框
        for det in result.detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            color = CLASS_COLORS.get(det.class_name, (0, 255, 255))

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 构造标签文本
            label = f"{det.class_name_cn} {det.confidence:.0%}"

            # 计算标签背景大小
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 绘制标签背景
            cv2.rectangle(
                img,
                (x1, y1 - label_h - baseline - 4),
                (x1 + label_w, y1),
                color,
                -1,  # 填充
            )

            # 绘制标签文字（白色）
            cv2.putText(
                img, label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 绘制整体判定信息
        info_text = f"Result: {result.overall_verdict}"
        time_text = f"Inference: {result.inference_time_ms:.1f}ms"

        # 顶部信息栏背景
        cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)

        # 整体判定
        verdict_color = (0, 200, 0) if not result.has_defect else (0, 0, 255)
        cv2.putText(
            img, info_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, verdict_color, 2, cv2.LINE_AA,
        )

        # 推理耗时
        cv2.putText(
            img, time_text,
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # 保存图片
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"[可视化] 结果已保存: {save_path}")

        # 弹窗显示
        if show:
            cv2.imshow("Screw Defect Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img

    def get_device_info(self) -> dict:
        """获取当前推理设备信息"""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                )
            except:
                info["gpu_name"] = "Unknown"
                info["gpu_memory_gb"] = 0
        return info


# ============================================================
# 辅助函数
# ============================================================

def find_best_model(project_root: Optional[str] = None) -> Optional[Path]:
    """
    在项目目录中查找最佳训练模型

    参数：
        project_root: 项目根目录，默认为本文件所在目录

    返回：
        最佳模型路径，未找到返回 None
    """
    if project_root is None:
        root = Path(__file__).parent.absolute()
    else:
        root = Path(project_root)

    runs_dir = root / "runs" / "train"
    if not runs_dir.exists():
        return None

    best_models = list(runs_dir.glob("*/weights/best.pt"))
    if best_models:
        best_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return best_models[0]

    return None


def get_model_path(model_arg: str = "") -> str:
    """
    获取模型路径：优先用户指定，其次自动查找，最后用预训练模型

    参数：
        model_arg: 命令行传入的模型路径

    返回：
        模型路径字符串
    """
    if model_arg:
        p = Path(model_arg)
        if p.exists():
            return str(p)
        print(f"[警告] 指定模型不存在: {model_arg}")

    best = find_best_model()
    if best:
        print(f"[信息] 自动使用训练好的模型: {best}")
        return str(best)

    print("[信息] 未找到训练模型，使用 yolov8n.pt 预训练权重")
    return "yolov8n.pt"


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="螺丝缺陷检测推理引擎")

    parser.add_argument(
        "--model", type=str, default="",
        help="模型权重路径 (默认: 自动查找 best.pt)"
    )
    parser.add_argument(
        "--image", type=str, default="",
        help="单张图片路径"
    )
    parser.add_argument(
        "--dir", type=str, default="",
        help="图片目录路径（批量检测）"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="置信度阈值 (默认: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU 阈值 (默认: 0.45)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=800,
        help="输入图片尺寸 (默认: 800)"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="推理设备 (默认: 自动选择)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="runs/detect",
        help="检测结果保存目录 (默认: runs/detect)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="显示检测结果窗口"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="批量检测时的 batch size (默认: 8)"
    )

    return parser.parse_args()


def main():
    """命令行主函数"""
    args = parse_args()

    # 获取模型路径
    model_path = get_model_path(args.model)

    # 创建检测器
    detector = ScrewDefectDetector(
        model_path=model_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        img_size=args.imgsz,
    )

    # 创建输出目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        # ---- 单张图片检测 ----
        print(f"\n{'='*50}")
        print(f"检测图片: {args.image}")
        print(f"{'='*50}")

        result = detector.detect_single(args.image)

        # 打印结果
        print(f"\n整体判定: {result.overall_verdict}")
        print(f"推理耗时: {result.inference_time_ms:.1f}ms")
        print(f"检测数量: {len(result.detections)}")
        print(f"缺陷数量: {result.defect_count}")

        for det in result.detections:
            print(f"  {det}")

        # 保存可视化结果
        save_path = str(save_dir / f"result_{Path(args.image).name}")
        detector.visualize_result(args.image, result, show=args.show, save_path=save_path)

    elif args.dir:
        # ---- 批量目录检测 ----
        print(f"\n{'='*50}")
        print(f"批量检测目录: {args.dir}")
        print(f"{'='*50}")

        results = detector.detect_image_dir(args.dir, batch_size=args.batch_size)

        # 统计汇总
        total = len(results)
        defect_count = sum(1 for r in results if r.has_defect)
        pass_count = total - defect_count

        print(f"\n{'='*50}")
        print(f"批量检测汇总")
        print(f"{'='*50}")
        print(f"  总计: {total} 张")
        print(f"  合格: {pass_count} 张")
        print(f"  不合格: {defect_count} 张")
        print(f"  合格率: {pass_count/total*100:.1f}%" if total > 0 else "  合格率: N/A")
        print(f"{'='*50}")

        # 保存每张结果图
        for r in results:
            fname = Path(r.image_path).name
            save_path = str(save_dir / f"result_{fname}")
            try:
                detector.visualize_result(r.image_path, r, save_path=save_path)
            except Exception as e:
                print(f"[警告] 可视化失败 {fname}: {e}")

    else:
        print("[提示] 请指定 --image 或 --dir 参数")
        print("示例:")
        print("  python inference.py --image test.jpg")
        print("  python inference.py --dir ./datasets/screws/images/val")


if __name__ == "__main__":
    main()
