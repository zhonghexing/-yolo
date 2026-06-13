"""
钢材表面缺陷检测演示脚本
Steel Surface Defect Detection Demo

模拟比赛检测流程：
    - 20 个样本，180 秒内完成检测
    - 从测试集随机抽取样本
    - 计时功能
    - 输出完整检测报告

使用方法：
    python demo.py                                      # 从测试集随机抽 20 张
    python demo.py --dir datasets/neu_det/test/images    # 指定图片目录
    python demo.py --samples 30 --limit 300             # 自定义样本数和时间限制
    python demo.py --model best.pt                      # 指定模型
    python demo.py --voice                              # 启用语音播报
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# 导入项目模块
from inference import (
    ScrewDefectDetector,
    ImageDetectionResult,
    SingleDetection,
    CLASS_NAMES,
    CLASS_NAMES_CN,
    get_model_path,
)
from feedback import FeedbackManager
from visualization import ReportGenerator


# ============================================================
# 演示主流程
# ============================================================

class DemoRunner:
    """
    演示运行器

    完整模拟比赛检测流程：
    1. 初始化检测器和反馈系统
    2. 从测试集加载样本
    3. 执行批量检测（计时）
    4. 生成报告和可视化
    """

    def __init__(
        self,
        model_path: str = "",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "",
        time_limit: float = 180.0,
        enable_voice: bool = False,
    ):
        self.time_limit = time_limit

        # 获取模型路径
        resolved_model = get_model_path(model_path)

        # 初始化检测器
        print("\n" + "=" * 60)
        print("初始化推理引擎")
        print("=" * 60)

        self.detector = ScrewDefectDetector(
            model_path=resolved_model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )

        # 初始化反馈系统
        print("\n" + "=" * 60)
        print("初始化反馈系统")
        print("=" * 60)

        self.feedback = FeedbackManager(
            enable_voice=enable_voice,
            enable_log=True,
            log_dir="runs/demo/logs",
            save_visual=True,
            visual_dir="runs/demo/visual",
        )

        # 初始化可视化报告生成器
        self.report_gen = ReportGenerator(output_dir="runs/demo/visualization")

    def run_demo(
        self,
        num_samples: int = 20,
        image_dir: str = "",
        normal_dir: str = "",
        normal_count: int = 3,
    ):
        """
        运行演示

        参数：
            num_samples: 样本数量（从测试集中随机抽取）
            image_dir: 图片目录，为空则使用默认测试集
            normal_dir: 正常样本目录，为空则使用默认目录
            normal_count: 混入的正常样本数量
        """
        print("\n" + "=" * 60)
        print(f"钢材表面缺陷检测演示")
        print(f"时间限制: {self.time_limit} 秒")
        print("=" * 60)

        # ---- 准备数据 ----
        if not image_dir:
            # 默认使用测试集
            image_dir = str(Path(__file__).parent / "datasets" / "neu_det" / "test" / "images")

        if not Path(image_dir).exists():
            print(f"[错误] 图片目录不存在: {image_dir}")
            return

        # 加载所有图片
        all_images = sorted(
            list(Path(image_dir).glob("*.jpg"))
            + list(Path(image_dir).glob("*.png"))
            + list(Path(image_dir).glob("*.jpeg"))
        )

        if not all_images:
            print(f"[错误] 目录中未找到图片: {image_dir}")
            return

        # 按类别分组
        class_images = {}
        for p in all_images:
            stem = p.stem
            cls = stem.rsplit("_", 1)[0] if "_" in stem else "unknown"
            class_images.setdefault(cls, []).append(p)

        # 分层抽样：保证每种缺陷至少 1 张，剩余名额随机补
        defect_samples_to_use = num_samples - normal_count
        selected = []
        remaining_pool = []

        for cls, imgs in class_images.items():
            pick = random.choice(imgs)
            selected.append(pick)
            remaining_pool.extend([x for x in imgs if x != pick])

        # 补齐剩余名额
        extra_needed = defect_samples_to_use - len(selected)
        if extra_needed > 0 and remaining_pool:
            selected.extend(random.sample(remaining_pool, min(extra_needed, len(remaining_pool))))

        image_paths = selected
        defect_samples_to_use = len(image_paths)
        print(f"\n[演示] 分层抽样 {defect_samples_to_use} 张缺陷图片，覆盖 {len(class_images)} 种缺陷类型")

        # 从文件名解析真实类别（如 crazing_271.jpg → crazing）
        samples = []
        for p in image_paths:
            # 文件名格式：类别_编号.jpg
            stem = p.stem  # 如 crazing_271
            true_class = stem.rsplit("_", 1)[0] if "_" in stem else "unknown"
            true_cn = CLASS_NAMES_CN.get(true_class, "未知")
            samples.append((str(p), true_class, true_cn))

        # 加载正常样本
        if not normal_dir:
            normal_dir = str(Path(__file__).parent / "datasets" / "neu_det" / "test" / "images_normal")

        if Path(normal_dir).exists() and normal_count > 0:
            normal_images = sorted(
                list(Path(normal_dir).glob("*.jpg"))
                + list(Path(normal_dir).glob("*.png"))
                + list(Path(normal_dir).glob("*.jpeg"))
            )

            if normal_images:
                # 随机抽取正常样本
                normal_to_use = min(normal_count, len(normal_images))
                normal_paths = random.sample(normal_images, normal_to_use)

                for p in normal_paths:
                    samples.append((str(p), "normal", "正常"))

                print(f"[演示] 混入 {normal_to_use} 张正常样本")
            else:
                print(f"[警告] 正常样本目录为空: {normal_dir}")
        else:
            if normal_count > 0:
                print(f"[警告] 正常样本目录不存在: {normal_dir}")

        # 打乱样本顺序
        random.shuffle(samples)

        # 统计各类别数量
        class_counts = {}
        for _, cls_name, _ in samples:
            cn = CLASS_NAMES_CN.get(cls_name, "未知")
            class_counts[cn] = class_counts.get(cn, 0) + 1

        print(f"\n[演示] 样本类别分布:")
        for cn_name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {cn_name}: {cnt} 张")

        # ---- 开始计时检测 ----
        print("\n" + "=" * 60)
        print("开始检测 (计时中...)")
        print("=" * 60)

        t_start = time.perf_counter()
        results = []

        for i, (img_path, true_class, true_cn) in enumerate(samples):
            # 检查是否超时
            elapsed = time.perf_counter() - t_start
            if elapsed >= self.time_limit:
                print(f"\n[警告] 已达到时间限制 {self.time_limit}s，已完成 {i}/{num_samples}")
                break

            # 执行检测
            try:
                result = self.detector.detect_single(img_path)
                results.append(result)

                # 判断检测是否正确
                detected_classes = [d.class_name for d in result.detections if d.is_defect]
                if true_class == "normal":
                    # 正常样本：没有检出缺陷才算正确
                    match = "✓ 正确" if not result.has_defect else "✗ 误检"
                elif not result.has_defect:
                    # 缺陷样本但模型认为无缺陷
                    match = "✗ 漏检"
                elif true_class in detected_classes:
                    match = "✓ 正确"
                else:
                    # 检测到了缺陷，但类型不完全匹配
                    match = f"→ {CLASS_NAMES_CN.get(detected_classes[0], '?')}"

                time_str = f"{result.inference_time_ms:.1f}ms"
                print(
                    f"  [{i+1:2d}/{num_samples}] {match:8s} | "
                    f"真实:{true_cn:6s} | "
                    f"检测:{result.overall_verdict:20s} | "
                    f"{time_str:>8s} | "
                    f"{Path(img_path).name}"
                )

                # 处理反馈（日志记录）
                self.feedback.process_result(result)

            except Exception as e:
                print(f"  [{i+1:2d}/{num_samples}] ERROR | {e}")

        t_end = time.perf_counter()
        total_time = t_end - t_start

        # ---- 输出结果汇总 ----
        self._print_summary(results, total_time, num_samples, samples)

        # ---- 生成可视化报告 ----
        print("\n" + "=" * 60)
        print("生成可视化报告")
        print("=" * 60)

        # 生成文本报告
        report_text = self.report_gen.generate_text_report(results)
        print("\n" + report_text)

        # 生成图表（传入真实标签用于对比）
        # 将字符串类别名转换为整数 ID（normal 用 -1 表示无缺陷）
        cls_name_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
        cls_name_to_id["normal"] = -1  # 正常样本用 -1 表示
        ground_truths = [cls_name_to_id.get(s[1], -1) for s in samples[:len(results)]]
        self.report_gen.generate_full_report(
            results,
            ground_truths=ground_truths,
        )

        # ---- 记录日志汇总 ----
        if self.feedback.logger:
            self.feedback.logger.log_summary(results)

        print(f"\n[演示] 所有结果保存在: runs/demo/")
        print("=" * 60)

    def _print_summary(
        self,
        results: list,
        total_time: float,
        total_samples: int,
        samples: list,
    ):
        """打印检测结果汇总"""
        n = len(results)
        if n == 0:
            print("[警告] 没有检测结果")
            return

        pass_count = sum(1 for r in results if not r.has_defect)
        defect_count = n - pass_count
        avg_time_ms = (total_time * 1000) / n
        within_limit = total_time <= self.time_limit

        print("\n")
        print("=" * 60)
        print("检测结果汇总")
        print("=" * 60)
        print(f"  总样本数:     {total_samples}")
        print(f"  已检测:       {n}")
        print(f"  检出缺陷:     {defect_count}")
        print(f"  未检出:       {pass_count}")
        print()
        print(f"  总耗时:       {total_time:.2f} 秒")
        print(f"  时间限制:     {self.time_limit:.0f} 秒")
        print(f"  是否达标:     {'✓ 是' if within_limit else '✗ 否'}")
        print(f"  平均单张耗时: {avg_time_ms:.1f} ms")
        print()

        # 统计推理设备信息
        device_info = self.detector.get_device_info()
        print(f"  推理设备:     {device_info['device']}")
        if device_info.get('gpu_name'):
            print(f"  GPU 型号:     {device_info['gpu_name']}")
        print("=" * 60)

        # 缺陷类型统计
        defect_types = {}
        for r in results:
            for det in r.detections:
                if det.is_defect:
                    name = det.class_name_cn
                    defect_types[name] = defect_types.get(name, 0) + 1

        if defect_types:
            print("\n检出缺陷类型分布:")
            print("-" * 40)
            for dtype, cnt in sorted(defect_types.items(), key=lambda x: -x[1]):
                bar = "█" * min(cnt, 20)
                print(f"  {dtype:10s}: {cnt:3d} {bar}")
            print("-" * 40)

        # 各类别检测准确率统计
        print("\n各类别检测准确率:")
        print("-" * 40)
        class_correct = {}
        class_total = {}
        for i, (img_path, true_class, true_cn) in enumerate(samples[:n]):
            class_total[true_cn] = class_total.get(true_cn, 0) + 1
            result = results[i]
            detected_classes = [d.class_name for d in result.detections if d.is_defect]

            if true_class == "normal":
                # 正确样本：没有检出缺陷才算正确
                if not result.has_defect:
                    class_correct[true_cn] = class_correct.get(true_cn, 0) + 1
            else:
                # 缺陷样本：检出对应缺陷才算正确
                if result.has_defect and true_class in detected_classes:
                    class_correct[true_cn] = class_correct.get(true_cn, 0) + 1

        for cn_name in sorted(class_total.keys()):
            total = class_total[cn_name]
            correct = class_correct.get(cn_name, 0)
            rate = correct / total * 100 if total > 0 else 0
            bar = "█" * int(rate / 5)
            print(f"  {cn_name:10s}: {correct}/{total} ({rate:.0f}%) {bar}")
        print("-" * 40)

        # 整体准确率
        total_correct = sum(class_correct.values())
        total_all = sum(class_total.values())
        overall_rate = total_correct / total_all * 100 if total_all > 0 else 0
        print(f"\n  整体准确率: {total_correct}/{total_all} = {overall_rate:.1f}%")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="钢材表面缺陷检测演示脚本")

    parser.add_argument(
        "--model", type=str, default="",
        help="模型权重路径 (默认: 自动查找)"
    )
    parser.add_argument(
        "--samples", type=int, default=20,
        help="样本数量 (默认: 20)"
    )
    parser.add_argument(
        "--limit", type=float, default=180.0,
        help="时间限制/秒 (默认: 180)"
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
        "--device", type=str, default="",
        help="推理设备 (默认: 自动选择)"
    )
    parser.add_argument(
        "--dir", type=str, default="",
        help="图片目录 (默认: datasets/neu_det/test/images)"
    )
    parser.add_argument(
        "--normal-dir", type=str, default="",
        help="正常样本目录 (默认: datasets/neu_det/test/images_normal)"
    )
    parser.add_argument(
        "--normal-count", type=int, default=3,
        help="混入的正常样本数量 (默认: 3)"
    )
    parser.add_argument(
        "--voice", action="store_true",
        help="启用语音播报"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("  钢材表面缺陷检测演示系统")
    print("  Steel Surface Defect Detection Demo")
    print("=" * 60)
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  样本数量: {args.samples}")
    print(f"  时间限制: {args.limit} 秒")
    print(f"  模型路径: {args.model or '自动查找'}")
    print("=" * 60)

    # 创建演示运行器
    runner = DemoRunner(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        time_limit=args.limit,
        enable_voice=args.voice,
    )

    # 运行演示
    try:
        runner.run_demo(
            num_samples=args.samples,
            image_dir=args.dir,
            normal_dir=args.normal_dir,
            normal_count=args.normal_count,
        )
        print("\n[成功] 演示完成!")
    except KeyboardInterrupt:
        print("\n[提示] 演示被用户中断")
    except Exception as e:
        print(f"\n[错误] 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
