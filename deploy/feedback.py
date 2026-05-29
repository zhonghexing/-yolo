"""
钢材表面缺陷检测反馈机制模块
Screw Defect Detection Feedback Module

功能：
    - 视觉反馈：绘制检测框（绿色=合格，红色=缺陷）
    - 语音播报：使用 pyttsx3 进行语音提示
    - 日志记录：检测结果写入 CSV 文件

使用方法：
    # 从 inference 模块获取结果后使用
    from feedback import FeedbackManager

    fb = FeedbackManager()
    fb.process_result(result)                # 处理单张结果（视觉+语音+日志）
    fb.process_batch_results(results)        # 处理批量结果
"""

import os
import csv
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from inference import _put_cn_text


# 导入推理模块的数据结构
# 使用延迟导入避免循环依赖
def _import_inference():
    """延迟导入 inference 模块"""
    from inference import (
        ImageDetectionResult,
        SingleDetection,
        CLASS_COLORS,
        CLASS_NAMES_CN,
    )
    return ImageDetectionResult, SingleDetection, CLASS_COLORS, CLASS_NAMES_CN


# ============================================================
# 视觉反馈绘制器
# ============================================================

class VisualFeedback:
    """
    视觉反馈绘制器

    提供丰富的检测结果可视化功能，
    包括检测框、标签、信息面板等。

    颜色规则：
        - 所有检出框均为缺陷，无检出 = 合格
        - 红色框: 各类缺陷
        - 黄色框: 低置信度（< 0.5）
    """

    # 默认颜色方案 (BGR)
    COLOR_PASS = (0, 200, 0)        # 绿色 - 合格
    COLOR_DEFECT = (0, 0, 255)      # 红色 - 缺陷
    COLOR_LOW_CONF = (0, 255, 255)  # 黄色 - 低置信度
    COLOR_BG = (30, 30, 30)         # 深灰色 - 背景
    COLOR_TEXT = (255, 255, 255)    # 白色 - 文字

    def __init__(self, font_scale: float = 0.6, thickness: int = 2):
        """
        参数：
            font_scale: 字体缩放比例
            thickness: 线条和文字粗细
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_detection(
        self,
        img: np.ndarray,
        detection,
        class_colors: Optional[dict] = None,
    ) -> np.ndarray:
        """
        在图片上绘制单个检测结果

        参数：
            img: 输入图片 (BGR)
            detection: SingleDetection 对象
            class_colors: 自定义颜色映射

        返回：
            绘制后的图片
        """
        if class_colors is None:
            from inference import CLASS_COLORS
            class_colors = CLASS_COLORS

        x1, y1, x2, y2 = [int(c) for c in detection.bbox]

        # 选择颜色
        if detection.confidence < 0.5:
            color = self.COLOR_LOW_CONF
        elif detection.is_defect:
            color = class_colors.get(detection.class_name, self.COLOR_DEFECT)
        else:
            color = self.COLOR_PASS

        # 绘制边框（带圆角效果：绘制两层）
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thickness)
        cv2.rectangle(img, (x1-1, y1-1), (x2+1, y2+1), color, 1)

        # 标签文本（中文）
        label = f"{detection.class_name_cn} {detection.confidence:.0%}"
        img = _put_cn_text(
            img, label,
            pos=(x1 + 3, y1 - 24),
            font_size=18,
            color=(255, 255, 255),
            bg_color=color,
        )

        return img

    def draw_info_panel(
        self,
        img: np.ndarray,
        result,
    ) -> np.ndarray:
        """
        在图片左上角绘制紧凑信息

        显示：整体判定、推理耗时
        """
        # 判定结果
        verdict = result.overall_verdict
        is_pass = not result.has_defect
        verdict_color = self.COLOR_PASS if is_pass else self.COLOR_DEFECT

        img = _put_cn_text(
            img, verdict,
            pos=(4, 4),
            font_size=14,
            color=verdict_color,
            bg_color=(0, 0, 0),
        )

        # 推理耗时和统计
        stats = (
            f"{result.inference_time_ms:.1f}ms | "
            f"Det: {len(result.detections)} | "
            f"Defect: {result.defect_count}"
        )
        cv2.putText(
            img, stats,
            (4, 24),
            self.font, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
        )

        return img

    def render(
        self,
        img: np.ndarray,
        result,
        class_colors: Optional[dict] = None,
    ) -> np.ndarray:
        """
        完整渲染一张图片的检测结果

        参数：
            img: 原始图片 (BGR)
            result: ImageDetectionResult 对象
            class_colors: 自定义颜色映射

        返回：
            渲染后的图片
        """
        output = img.copy()

        # 绘制所有检测框
        for det in result.detections:
            output = self.draw_detection(output, det, class_colors)

        # 绘制信息面板
        output = self.draw_info_panel(output, result)

        return output


# ============================================================
# 语音播报器
# ============================================================

class VoiceFeedback:
    """
    语音反馈播报器

    使用 pyttsx3 进行离线语音播报，
    在后台线程执行避免阻塞推理流程。
    """

    def __init__(self, enabled: bool = True, rate: int = 150, volume: float = 0.8):
        """
        参数：
            enabled: 是否启用语音播报
            rate: 语速（单词/分钟）
            volume: 音量 [0, 1]
        """
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self._engine = None
        self._lock = threading.Lock()

        if enabled:
            self._init_engine()

    def _init_engine(self):
        """初始化 pyttsx3 引擎"""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)

            # 尝试设置中文语音
            voices = self._engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self._engine.setProperty('voice', voice.id)
                    break

            print("[语音] 引擎初始化成功")
        except ImportError:
            print("[语音] 警告: pyttsx3 未安装，语音播报已禁用")
            print("       安装命令: pip install pyttsx3")
            self.enabled = False
        except Exception as e:
            print(f"[语音] 警告: 语音引擎初始化失败: {e}")
            self.enabled = False

    def speak_async(self, text: str):
        """
        异步语音播报（不阻塞主线程）

        参数：
            text: 要播报的文本
        """
        if not self.enabled or self._engine is None:
            return

        def _speak():
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    # 语音播报失败不应影响主流程
                    pass

        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def speak_result(self, result):
        """
        根据检测结果生成语音播报

        参数：
            result: ImageDetectionResult 对象
        """
        if not result.detections:
            self.speak_async("未检测到目标")
            return

        if not result.has_defect:
            self.speak_async("检测合格")
            return

        # 统计缺陷类型
        defect_summary = {}
        for det in result.detections:
            if det.is_defect:
                name = det.class_name_cn
                defect_summary[name] = defect_summary.get(name, 0) + 1

        # 构造播报文本
        parts = ["检测不合格"]
        for name, count in defect_summary.items():
            parts.append(f"{count}处{name}")

        text = "，".join(parts)
        self.speak_async(text)

    def speak_batch_summary(self, total: int, pass_count: int, defect_count: int):
        """播报批量检测汇总"""
        rate = pass_count / total * 100 if total > 0 else 0
        text = (
            f"批量检测完成，共{total}个样本，"
            f"合格{pass_count}个，不合格{defect_count}个，"
            f"合格率{rate:.0f}%"
        )
        self.speak_async(text)


# ============================================================
# 日志记录器
# ============================================================

class DetectionLogger:
    """
    检测结果日志记录器

    将检测结果记录到 CSV 文件，方便后续分析和统计。

    CSV 字段：
        timestamp, image_path, overall_verdict, has_defect,
        defect_count, inference_time_ms,
        class_name, class_name_cn, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2
    """

    def __init__(
        self,
        log_dir: str = "runs/logs",
        log_filename: Optional[str] = None,
    ):
        """
        参数：
            log_dir: 日志保存目录
            log_filename: 日志文件名，默认使用时间戳命名
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"detection_log_{timestamp}.csv"

        self.log_path = self.log_dir / log_filename
        self._write_header()

        print(f"[日志] 记录到: {self.log_path}")

    def _write_header(self):
        """写入 CSV 文件头"""
        headers = [
            "timestamp",
            "image_path",
            "overall_verdict",
            "has_defect",
            "defect_count",
            "inference_time_ms",
            "detection_index",
            "class_id",
            "class_name",
            "class_name_cn",
            "confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ]

        with open(self.log_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_result(self, result):
        """
        记录单张图片的检测结果

        如果图片有多个检测框，每个检测框写一行。
        如果图片无检测结果，写一行（detection_index=-1）。
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows = []
        if result.detections:
            for idx, det in enumerate(result.detections):
                row = [
                    timestamp,
                    result.image_path,
                    result.overall_verdict,
                    result.has_defect,
                    result.defect_count,
                    f"{result.inference_time_ms:.1f}",
                    idx,
                    det.class_id,
                    det.class_name,
                    det.class_name_cn,
                    f"{det.confidence:.4f}",
                    f"{det.bbox[0]:.1f}",
                    f"{det.bbox[1]:.1f}",
                    f"{det.bbox[2]:.1f}",
                    f"{det.bbox[3]:.1f}",
                ]
                rows.append(row)
        else:
            # 无检测结果的图片也记录一行
            row = [
                timestamp,
                result.image_path,
                result.overall_verdict,
                False,
                0,
                f"{result.inference_time_ms:.1f}",
                -1, -1, "none", "无", "0.0",
                "0", "0", "0", "0",
            ]
            rows.append(row)

        with open(self.log_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def log_batch_results(self, results: list):
        """
        记录批量检测结果

        参数：
            results: List[ImageDetectionResult]
        """
        for result in results:
            self.log_result(result)

        print(f"[日志] 已记录 {len(results)} 条检测结果")

    def log_summary(self, results: list):
        """
        记录批量检测汇总统计

        参数：
            results: List[ImageDetectionResult]
        """
        total = len(results)
        pass_count = sum(1 for r in results if not r.has_defect)
        defect_count = total - pass_count
        avg_time = (
            sum(r.inference_time_ms for r in results) / total if total > 0 else 0
        )

        # 统计各缺陷类型数量
        defect_type_counts = {}
        for r in results:
            for det in r.detections:
                if det.is_defect:
                    name = det.class_name_cn
                    defect_type_counts[name] = defect_type_counts.get(name, 0) + 1

        # 写入汇总行（追加在 CSV 末尾，用空行分隔）
        summary_path = self.log_path.parent / (self.log_path.stem + "_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("批量检测汇总报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总样本数:     {total}\n")
            f.write(f"合格数:       {pass_count}\n")
            f.write(f"不合格数:     {defect_count}\n")
            f.write(f"合格率:       {pass_count/total*100:.1f}%\n" if total > 0 else "合格率: N/A\n")
            f.write(f"平均推理耗时: {avg_time:.1f} ms\n\n")

            if defect_type_counts:
                f.write("缺陷类型统计:\n")
                for dtype, cnt in sorted(defect_type_counts.items(), key=lambda x: -x[1]):
                    f.write(f"  {dtype}: {cnt} 处\n")

            f.write("\n" + "=" * 50 + "\n")

        print(f"[日志] 汇总报告已保存: {summary_path}")


# ============================================================
# 反馈管理器（整合三类反馈）
# ============================================================

class FeedbackManager:
    """
    反馈管理器

    统一管理视觉、语音、日志三类反馈机制，
    提供简洁的接口供上层调用。
    """

    def __init__(
        self,
        enable_voice: bool = True,
        enable_log: bool = True,
        log_dir: str = "runs/logs",
        save_visual: bool = True,
        visual_dir: str = "runs/feedback",
    ):
        """
        参数：
            enable_voice: 启用语音播报
            enable_log: 启用日志记录
            log_dir: 日志保存目录
            save_visual: 保存可视化结果图片
            visual_dir: 可视化图片保存目录
        """
        # 视觉反馈
        self.visual = VisualFeedback()
        self.save_visual = save_visual
        self.visual_dir = Path(visual_dir)
        if save_visual:
            self.visual_dir.mkdir(parents=True, exist_ok=True)

        # 语音反馈
        self.voice = VoiceFeedback(enabled=enable_voice)

        # 日志记录
        self.logger = DetectionLogger(log_dir=log_dir) if enable_log else None

    def process_result(
        self,
        result,
        image_input=None,
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        处理单张检测结果（触发所有反馈）

        参数：
            result: ImageDetectionResult 对象
            image_input: 原始图片路径或 numpy 数组（用于可视化）
            save_path: 可视化结果保存路径

        返回：
            绘制后的图片（如果提供了 image_input），否则 None
        """
        # 1. 日志记录
        if self.logger:
            self.logger.log_result(result)

        # 2. 语音播报
        self.voice.speak_result(result)

        # 3. 视觉反馈
        output_img = None
        if image_input is not None:
            if isinstance(image_input, (str, Path)):
                img = cv2.imread(str(image_input))
            else:
                img = image_input

            if img is not None:
                output_img = self.visual.render(img, result)

                # 保存可视化结果
                if self.save_visual and save_path is None:
                    fname = Path(result.image_path).name
                    save_path = str(self.visual_dir / f"feedback_{fname}")

                if save_path:
                    cv2.imwrite(save_path, output_img)

        return output_img

    def process_batch_results(
        self,
        results: list,
        image_inputs: Optional[list] = None,
    ):
        """
        处理批量检测结果

        参数：
            results: List[ImageDetectionResult]
            image_inputs: 原始图片列表（可选，用于可视化）
        """
        total = len(results)

        for i, result in enumerate(results):
            img_input = image_inputs[i] if image_inputs else None
            self.process_result(result, image_input=img_input)

        # 记录汇总
        if self.logger:
            self.logger.log_summary(results)

        # 语音播报汇总
        pass_count = sum(1 for r in results if not r.has_defect)
        defect_count = total - pass_count
        self.voice.speak_batch_summary(total, pass_count, defect_count)

        print(f"\n[反馈] 批量处理完成: {total} 张, 合格 {pass_count}, 不合格 {defect_count}")


# ============================================================
# 测试代码
# ============================================================

def test_visual_feedback():
    """测试视觉反馈（不依赖模型，使用模拟数据）"""
    from inference import ImageDetectionResult, SingleDetection

    # 创建模拟检测结果
    detections = [
        SingleDetection(
            class_id=1,
            class_name="inclusion",
            class_name_cn="夹杂",
            confidence=0.87,
            bbox=(100, 100, 250, 200),
            is_defect=True,
        ),
        SingleDetection(
            class_id=5,
            class_name="scratches",
            class_name_cn="划痕",
            confidence=0.95,
            bbox=(300, 150, 450, 300),
            is_defect=True,
        ),
    ]

    result = ImageDetectionResult(
        image_path="test.jpg",
        detections=detections,
        inference_time_ms=23.5,
        image_shape=(480, 640, 3),
    )

    # 创建模拟图片
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    # 绘制结果
    vf = VisualFeedback()
    output = vf.render(img, result)

    save_path = "runs/feedback/test_visual.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, output)
    print(f"[测试] 视觉反馈测试图已保存: {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("反馈模块测试")
    print("=" * 50)

    test_visual_feedback()

    # 测试日志
    from inference import ImageDetectionResult, SingleDetection

    detections = [
        SingleDetection(
            class_id=2,
            class_name="patches",
            class_name_cn="斑块",
            confidence=0.92,
            bbox=(50, 50, 200, 180),
            is_defect=True,
        ),
    ]

    result = ImageDetectionResult(
        image_path="sample_001.jpg",
        detections=detections,
        inference_time_ms=15.3,
        image_shape=(640, 640, 3),
    )

    logger = DetectionLogger(log_dir="runs/logs")
    logger.log_result(result)
    logger.log_summary([result])

    print("\n[测试] 所有反馈模块测试完成")
