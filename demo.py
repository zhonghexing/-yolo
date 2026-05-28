"""
钢材表面缺陷检测演示脚本
Steel Surface Defect Detection Demo Script

模拟比赛检测流程：
    - 20 个样本，180 秒内完成检测
    - 模拟生成测试样本
    - 计时功能
    - 输出完整检测报告

使用方法：
    python demo.py                          # 使用默认配置运行演示
    python demo.py --model best.pt          # 指定模型
    python demo.py --samples 20 --limit 180 # 自定义样本数和时间限制
    python demo.py --real --dir test_images # 使用真实图片目录
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
# 模拟数据生成器
# ============================================================

class SampleGenerator:
    """
    测试样本生成器

    生成模拟的螺丝图片用于演示。
    每张图片包含不同数量的"螺丝"区域，
    并带有模拟的缺陷特征。
    """

    # 模拟类别分布（更贴近实际数据分布）
    CLASS_WEIGHTS = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]

    def __init__(self, image_size: tuple = (640, 640)):
        self.image_size = image_size

    def generate_sample(self, sample_id: int) -> tuple:
        """
        生成单个模拟样本

        参数：
            sample_id: 样本编号

        返回：
            (image: np.ndarray, true_label: int, true_name: str)
        """
        h, w = self.image_size

        # 创建背景（灰色金属质感）
        bg_color = random.randint(140, 180)
        img = np.full((h, w, 3), bg_color, dtype=np.uint8)

        # 添加噪声模拟金属纹理
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 随机选择真实类别
        true_label = random.choices(range(6), weights=self.CLASS_WEIGHTS, k=1)[0]
        true_name = CLASS_NAMES[true_label]

        # 在图片上绘制模拟螺丝区域
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        screw_w = random.randint(100, 200)
        screw_h = random.randint(150, 300)

        # 绘制螺丝主体（深灰色椭圆）
        cv2.ellipse(
            img, (cx, cy), (screw_w // 2, screw_h // 2),
            0, 0, 360, (80, 80, 80), -1
        )
        cv2.ellipse(
            img, (cx, cy), (screw_w // 2, screw_h // 2),
            0, 0, 360, (50, 50, 50), 2
        )

        # 绘制螺丝头部（圆形）
        head_r = screw_w // 3
        cv2.circle(img, (cx, cy - screw_h // 4), head_r, (100, 100, 100), -1)
        cv2.circle(img, (cx, cy - screw_h // 4), head_r, (60, 60, 60), 2)

        # 绘制螺纹线
        for i in range(5):
            y = cy - screw_h // 3 + i * (screw_h // 6)
            cv2.line(
                img,
                (cx - screw_w // 2 + 10, y),
                (cx + screw_w // 2 - 10, y),
                (60, 60, 60), 1
            )

        # 根据类别添加缺陷特征
        if true_label == 1:  # 轻微划痕
            for _ in range(random.randint(1, 2)):
                x1 = cx + random.randint(-screw_w // 3, screw_w // 3)
                y1 = cy + random.randint(-screw_h // 3, screw_h // 3)
                x2 = x1 + random.randint(20, 50)
                y2 = y1 + random.randint(-10, 10)
                cv2.line(img, (x1, y1), (x2, y2), (120, 120, 120), 1)

        elif true_label == 2:  # 严重划痕
            for _ in range(random.randint(2, 4)):
                x1 = cx + random.randint(-screw_w // 3, screw_w // 3)
                y1 = cy + random.randint(-screw_h // 3, screw_h // 3)
                x2 = x1 + random.randint(30, 80)
                y2 = y1 + random.randint(-20, 20)
                cv2.line(img, (x1, y1), (x2, y2), (150, 150, 150), 2)

        elif true_label == 3:  # 缺角
            corner_x = cx + screw_w // 2
            corner_y = cy - screw_h // 2
            pts = np.array([
                [corner_x, corner_y],
                [corner_x - 30, corner_y],
                [corner_x, corner_y + 30],
            ], np.int32)
            cv2.fillPoly(img, [pts], (bg_color, bg_color, bg_color))

        elif true_label == 4:  # 变形
            # 绘制扭曲的轮廓
            pts = []
            for angle in range(0, 360, 30):
                r = screw_w // 2 + random.randint(-15, 15)
                x = int(cx + r * np.cos(np.radians(angle)))
                y = int(cy + r * np.sin(np.radians(angle)))
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.polylines(img, [pts], True, (60, 60, 60), 2)

        elif true_label == 5:  # 混料
            # 绘制颜色异常区域
            anomaly_x = cx + random.randint(-screw_w // 4, screw_w // 4)
            anomaly_y = cy + random.randint(-screw_h // 4, screw_h // 4)
            cv2.circle(img, (anomaly_x, anomaly_y), 20, (40, 80, 150), -1)

        return img, true_label, true_name

    def generate_batch(self, count: int, output_dir: str = "runs/demo/samples") -> list:
        """
        生成一批测试样本

        参数：
            count: 样本数量
            output_dir: 样本保存目录

        返回：
            list of (image_path, true_label, true_name)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for i in range(count):
            img, true_label, true_name = self.generate_sample(i + 1)

            # 保存图片
            filename = f"sample_{i+1:03d}_{true_name}.jpg"
            img_path = str(output_dir / filename)
            cv2.imwrite(img_path, img)

            samples.append((img_path, true_label, true_name))

        print(f"[演示] 已生成 {count} 个模拟样本到: {output_dir}")
        return samples


# ============================================================
# 演示主流程
# ============================================================

class DemoRunner:
    """
    演示运行器

    完整模拟比赛检测流程：
    1. 初始化检测器和反馈系统
    2. 生成或加载测试样本
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
        """
        参数：
            model_path: 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            device: 推理设备
            time_limit: 时间限制（秒）
            enable_voice: 启用语音播报
        """
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
    ):
        """
        运行演示

        参数：
            num_samples: 样本数量（仅在使用模拟数据时有效）
            image_dir: 真实图片目录，为空则使用模拟数据
        """
        print("\n" + "=" * 60)
        print(f"钢材表面缺陷检测演示")
        print(f"时间限制: {self.time_limit} 秒")
        print("=" * 60)

        # ---- 准备数据 ----
        if image_dir and Path(image_dir).exists():
            # 使用真实图片
            print(f"\n[演示] 使用真实图片目录: {image_dir}")
            image_paths = sorted(
                list(Path(image_dir).glob("*.jpg"))
                + list(Path(image_dir).glob("*.png"))
                + list(Path(image_dir).glob("*.jpeg"))
            )
            image_paths = [str(p) for p in image_paths]

            if not image_paths:
                print("[错误] 目录中未找到图片")
                return

            num_samples = len(image_paths)
            samples = [(p, -1, "unknown") for p in image_paths]
        else:
            # 生成模拟样本
            print(f"\n[演示] 生成 {num_samples} 个模拟测试样本...")
            generator = SampleGenerator()
            samples = generator.generate_batch(num_samples)

        print(f"\n[演示] 共 {num_samples} 个待检测样本")

        # ---- 开始计时检测 ----
        print("\n" + "=" * 60)
        print("开始检测 (计时中...)")
        print("=" * 60)

        t_start = time.perf_counter()
        results = []

        for i, (img_path, true_label, true_name) in enumerate(samples):
            # 检查是否超时
            elapsed = time.perf_counter() - t_start
            if elapsed >= self.time_limit:
                print(f"\n[警告] 已达到时间限制 {self.time_limit}s，已完成 {i}/{num_samples}")
                break

            # 执行检测
            try:
                result = self.detector.detect_single(img_path)
                results.append(result)

                # 实时输出
                status = "PASS" if not result.has_defect else "FAIL"
                time_str = f"{result.inference_time_ms:.1f}ms"
                print(
                    f"  [{i+1:2d}/{num_samples}] {status:4s} | "
                    f"{result.overall_verdict:30s} | "
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
        self._print_summary(results, total_time, num_samples)

        # ---- 生成可视化报告 ----
        print("\n" + "=" * 60)
        print("生成可视化报告")
        print("=" * 60)

        # 生成文本报告
        report_text = self.report_gen.generate_text_report(results)
        print("\n" + report_text)

        # 生成图表
        ground_truths = [s[1] for s in samples[:len(results)]]
        # 只有在真实标签有效时才传递
        gt_valid = all(g >= 0 for g in ground_truths)
        self.report_gen.generate_full_report(
            results,
            ground_truths=ground_truths if gt_valid else None,
        )

        # ---- 记录日志汇总 ----
        if self.feedback.logger:
            self.feedback.logger.log_summary(results)

        print(f"\n[演示] 所有结果保存在: runs/demo/")
        print("=" * 60)

    def _print_summary(self, results: list, total_time: float, total_samples: int):
        """打印检测结果汇总"""
        n = len(results)
        pass_count = sum(1 for r in results if not r.has_defect)
        defect_count = n - pass_count
        avg_time_ms = (total_time * 1000) / n if n > 0 else 0

        # 判定是否在时间限制内完成
        within_limit = total_time <= self.time_limit

        print("\n")
        print("=" * 60)
        print("检测结果汇总")
        print("=" * 60)
        print(f"  总样本数:     {total_samples}")
        print(f"  已检测:       {n}")
        print(f"  合格:         {pass_count}")
        print(f"  不合格:       {defect_count}")
        print(f"  合格率:       {pass_count/n*100:.1f}%" if n > 0 else "  合格率:       N/A")
        print()
        print(f"  总耗时:       {total_time:.2f} 秒")
        print(f"  时间限制:     {self.time_limit:.0f} 秒")
        print(f"  是否达标:     {'是' if within_limit else '否'}")
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
            print("\n缺陷类型分布:")
            print("-" * 40)
            for dtype, cnt in sorted(defect_types.items(), key=lambda x: -x[1]):
                bar = "#" * min(cnt, 20)
                print(f"  {dtype:10s}: {cnt:3d} {bar}")
            print("-" * 40)


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
        help="模拟样本数量 (默认: 20)"
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
        help="使用真实图片目录（非模拟数据）"
    )
    parser.add_argument(
        "--real", action="store_true",
        help="使用真实图片（需配合 --dir 使用）"
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
    print("  Screw Defect Detection Demo")
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
            image_dir=args.dir if args.real else "",
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
