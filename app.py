"""
螺丝缺陷检测 PyQt5 桌面应用
Screw Defect Detection Desktop Application

功能：
    - 图片/文件夹拖拽检测
    - 实时检测结果展示
    - 统计分析面板
    - 180秒计时演示模式（比赛用）
    - 检测报告导出
    - 视频流实时检测（摄像头/视频文件）

使用方法：
    python app.py
    python run_app.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from constants import CLASS_NAMES_CN, CLASS_COLORS_RGB, CLASS_NAMES, CLASS_COLORS_BGR

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMenuBar, QMenu, QAction, QToolBar, QStatusBar, QFileDialog,
    QSplitter, QGroupBox, QGridLayout, QProgressBar, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QDialog, QFormLayout,
    QDialogButtonBox, QTextEdit, QFrame, QSizePolicy, QStyle,
    QCheckBox, QTabWidget, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QTimer, QSize, QThread, pyqtSignal, QMimeData, QUrl
)
from PyQt5.QtGui import (
    QPixmap, QImage, QIcon, QFont, QColor, QPalette,
    QPainter, QPen, QBrush, QDragEnterEvent, QDropEvent
)


# ============================================================
# 常量定义
# ============================================================

APP_TITLE = "Steel Defect Inspector"
APP_TITLE_CN = "钢材缺陷检测系统"
APP_VERSION = "2.2.0"
WINDOW_MIN_SIZE = (1360, 860)

# ── 工业风配色方案 ──────────────────────────────────────────
# 主色调: 深蓝灰 + 钢铁蓝  |  语义色: 绿=合格  红=缺陷  琥珀=警告
COLOR_PRIMARY    = "#3b82f6"   # 钢铁蓝
COLOR_PRIMARY_DK = "#2563eb"
COLOR_SUCCESS    = "#10b981"   # 合格绿
COLOR_DANGER     = "#ef4444"   # 缺陷红
COLOR_WARNING    = "#f59e0b"   # 警告琥珀
COLOR_INFO       = "#06b6d4"   # 信息青

# ── 深色主题 ────────────────────────────────────────────────
DARK_THEME = """
/* ── 全局 ── */
QMainWindow, QWidget#centralWidget {
    background-color: #0f1117;
    color: #e2e8f0;
}

/* ── 菜单栏 ── */
QMenuBar {
    background-color: #161822;
    color: #94a3b8;
    padding: 2px 4px;
    font-size: 13px;
    border-bottom: 1px solid #1e2030;
}
QMenuBar::item { padding: 6px 14px; border-radius: 4px; }
QMenuBar::item:selected { background-color: #1e293b; color: #e2e8f0; }

QMenu {
    background-color: #161822;
    color: #e2e8f0;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item { padding: 8px 24px; border-radius: 4px; }
QMenu::item:selected { background-color: #1e293b; }
QMenu::separator { height: 1px; background: #2a2d3a; margin: 4px 8px; }

/* ── 工具栏 ── */
QToolBar {
    background-color: #161822;
    border-bottom: 1px solid #1e2030;
    spacing: 4px;
    padding: 6px 8px;
}

/* ── 按钮 ── */
QPushButton {
    background-color: #3b82f6;
    color: #ffffff;
    border: none;
    padding: 8px 18px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover { background-color: #2563eb; }
QPushButton:pressed { background-color: #1d4ed8; }
QPushButton:disabled { background-color: #2a2d3a; color: #4a5568; }

QPushButton#startBtn {
    background-color: #10b981;
    color: #ffffff;
}
QPushButton#startBtn:hover { background-color: #059669; }

QPushButton#clearBtn {
    background-color: #ef4444;
}
QPushButton#clearBtn:hover { background-color: #dc2626; }

QPushButton#timerBtn {
    background-color: #f59e0b;
    color: #1a1a2e;
}
QPushButton#timerBtn:hover { background-color: #d97706; }

QPushButton#cameraBtn {
    background-color: #8b5cf6;
}
QPushButton#cameraBtn:hover { background-color: #7c3aed; }

QPushButton#videoBtn {
    background-color: #6366f1;
}
QPushButton#videoBtn:hover { background-color: #4f46e5; }

QPushButton#stopVideoBtn {
    background-color: #ef4444;
}
QPushButton#stopVideoBtn:hover { background-color: #dc2626; }

QPushButton#themeBtn {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #2a2d3a;
    padding: 6px 12px;
    font-size: 14px;
}
QPushButton#themeBtn:hover { background-color: #334155; color: #e2e8f0; }

/* ── 分组框 ── */
QGroupBox {
    font-weight: 600;
    border: 1px solid #1e2030;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 20px;
    background-color: #161822;
    font-size: 13px;
    color: #94a3b8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #3b82f6;
}

/* ── 标签页 ── */
QTabWidget::pane {
    border: 1px solid #1e2030;
    border-radius: 8px;
    background-color: #161822;
    top: -1px;
}
QTabBar::tab {
    background-color: #0f1117;
    color: #64748b;
    padding: 10px 22px;
    border: 1px solid #1e2030;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #161822;
    color: #3b82f6;
    border-bottom: 2px solid #3b82f6;
}

/* ── 表格 ── */
QTableWidget {
    background-color: #161822;
    alternate-background-color: #1a1d28;
    border: 1px solid #1e2030;
    border-radius: 8px;
    gridline-color: #1e2030;
    font-size: 13px;
    color: #e2e8f0;
}
QTableWidget::item { padding: 6px; }
QTableWidget::item:selected {
    background-color: #1e40af;
    color: #ffffff;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #1e2030;
    color: #94a3b8;
    padding: 10px;
    border: none;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── 状态栏 ── */
QStatusBar {
    background-color: #0f1117;
    color: #64748b;
    font-size: 12px;
    border-top: 1px solid #1e2030;
}

/* ── 标签 ── */
QLabel#titleLabel {
    font-size: 20px;
    font-weight: 700;
    color: #e2e8f0;
    padding: 4px 0;
}
QLabel#versionLabel {
    font-size: 11px;
    color: #475569;
    background-color: #1e2030;
    padding: 2px 8px;
    border-radius: 4px;
}
QLabel#statsLabel {
    font-size: 13px;
    color: #94a3b8;
    padding: 4px;
}
QLabel#fpsLabel {
    font-size: 13px;
    font-weight: 700;
    color: #10b981;
    padding: 4px 12px;
    background-color: #0f2922;
    border: 1px solid #10b981;
    border-radius: 6px;
}
QLabel#dropLabel {
    color: #475569;
    font-size: 15px;
}

/* ── 进度条 ── */
QProgressBar {
    border: 1px solid #1e2030;
    border-radius: 5px;
    text-align: center;
    height: 20px;
    background-color: #1e2030;
    color: #e2e8f0;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 4px;
}

/* ── 图片区域 ── */
QFrame#imageFrame {
    background-color: #0f1117;
    border: 2px dashed #2a2d3a;
    border-radius: 12px;
}

/* ── 分割器 ── */
QSplitter::handle {
    background-color: #1e2030;
    width: 2px;
}

/* ── 滚动条 ── */
QScrollBar:vertical {
    background-color: #0f1117;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background-color: #2a2d3a;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background-color: #3b4252; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { height: 0px; }

/* ── 输入框 ── */
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1e2030;
    color: #e2e8f0;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #3b82f6;
}

/* ── 对话框 ── */
QDialog {
    background-color: #0f1117;
    color: #e2e8f0;
}
QTextEdit {
    background-color: #161822;
    color: #e2e8f0;
    border: 1px solid #1e2030;
    border-radius: 8px;
}
"""

# ── 浅色主题 ────────────────────────────────────────────────
LIGHT_THEME = """
QMainWindow, QWidget#centralWidget {
    background-color: #f1f5f9;
    color: #1e293b;
}
QMenuBar {
    background-color: #ffffff;
    color: #475569;
    padding: 2px 4px;
    font-size: 13px;
    border-bottom: 1px solid #e2e8f0;
}
QMenuBar::item { padding: 6px 14px; border-radius: 4px; }
QMenuBar::item:selected { background-color: #f1f5f9; color: #1e293b; }
QMenu {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item { padding: 8px 24px; border-radius: 4px; }
QMenu::item:selected { background-color: #f1f5f9; }
QMenu::separator { height: 1px; background: #e2e8f0; margin: 4px 8px; }
QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    spacing: 4px;
    padding: 6px 8px;
}
QPushButton {
    background-color: #3b82f6;
    color: #ffffff;
    border: none;
    padding: 8px 18px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover { background-color: #2563eb; }
QPushButton:pressed { background-color: #1d4ed8; }
QPushButton:disabled { background-color: #e2e8f0; color: #94a3b8; }
QPushButton#startBtn { background-color: #10b981; color: #ffffff; }
QPushButton#startBtn:hover { background-color: #059669; }
QPushButton#clearBtn { background-color: #ef4444; }
QPushButton#clearBtn:hover { background-color: #dc2626; }
QPushButton#timerBtn { background-color: #f59e0b; color: #1a1a2e; }
QPushButton#timerBtn:hover { background-color: #d97706; }
QPushButton#cameraBtn { background-color: #8b5cf6; }
QPushButton#cameraBtn:hover { background-color: #7c3aed; }
QPushButton#videoBtn { background-color: #6366f1; }
QPushButton#videoBtn:hover { background-color: #4f46e5; }
QPushButton#stopVideoBtn { background-color: #ef4444; }
QPushButton#stopVideoBtn:hover { background-color: #dc2626; }
QPushButton#themeBtn {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
    padding: 6px 12px;
    font-size: 14px;
}
QPushButton#themeBtn:hover { background-color: #e2e8f0; color: #1e293b; }
QGroupBox {
    font-weight: 600;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 20px;
    background-color: #ffffff;
    font-size: 13px;
    color: #475569;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #3b82f6;
}
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background-color: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background-color: #f1f5f9;
    color: #64748b;
    padding: 10px 22px;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #3b82f6;
    border-bottom: 2px solid #3b82f6;
}
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    gridline-color: #f1f5f9;
    font-size: 13px;
    color: #1e293b;
}
QTableWidget::item { padding: 6px; }
QTableWidget::item:selected {
    background-color: #3b82f6;
    color: #ffffff;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #1e293b;
    color: #ffffff;
    padding: 10px;
    border: none;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QStatusBar {
    background-color: #1e293b;
    color: #94a3b8;
    font-size: 12px;
}
QLabel#titleLabel {
    font-size: 20px;
    font-weight: 700;
    color: #1e293b;
    padding: 4px 0;
}
QLabel#versionLabel {
    font-size: 11px;
    color: #64748b;
    background-color: #e2e8f0;
    padding: 2px 8px;
    border-radius: 4px;
}
QLabel#statsLabel { font-size: 13px; color: #475569; padding: 4px; }
QLabel#fpsLabel {
    font-size: 13px;
    font-weight: 700;
    color: #059669;
    padding: 4px 12px;
    background-color: #ecfdf5;
    border: 1px solid #10b981;
    border-radius: 6px;
}
QLabel#dropLabel { color: #94a3b8; font-size: 15px; }
QProgressBar {
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    text-align: center;
    height: 20px;
    background-color: #f1f5f9;
    color: #1e293b;
    font-size: 11px;
}
QProgressBar::chunk { background-color: #3b82f6; border-radius: 4px; }
QFrame#imageFrame {
    background-color: #0f172a;
    border: 2px dashed #334155;
    border-radius: 12px;
}
QSplitter::handle { background-color: #e2e8f0; width: 2px; }
QScrollBar:vertical { background-color: #f1f5f9; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background-color: #cbd5e1; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background-color: #94a3b8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { height: 0px; }
QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #3b82f6; }
QDialog { background-color: #f1f5f9; color: #1e293b; }
QTextEdit {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}
"""

STYLE_SHEET = DARK_THEME  # 默认深色主题


# ============================================================
# 视频流检测工作线程
# ============================================================

class VideoStreamThread(QThread):
    """视频流实时检测线程，支持摄像头和视频文件"""

    frame_ready = pyqtSignal(object, object)   # QPixmap, ImageDetectionResult
    fps_update = pyqtSignal(float)              # 当前 FPS
    error_occurred = pyqtSignal(str)            # 错误信息
    stream_finished = pyqtSignal()              # 视频流结束

    def __init__(self, detector, source=0, parent=None):
        """
        参数:
            detector: ScrewDefectDetector 实例
            source: 摄像头索引(int) 或视频文件路径(str)
            parent: 父对象
        """
        super().__init__(parent)
        self.detector = detector
        self.source = source
        self._is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_occurred.emit(f"无法打开视频源: {self.source}")
            return

        fps_counter = 0
        fps_start_time = time.perf_counter()
        fps_display_interval = 0.5  # 每0.5秒更新一次FPS显示

        while self._is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 视频文件播放完毕
                if isinstance(self.source, str):
                    break
                # 摄像头读取失败，重试
                continue

            # 跳帧机制：检测耗时较长时，不排队堆积
            # 如果队列中有待处理的帧，跳过当前帧
            # 这里用一个简单的标志来实现
            frame_for_detection = frame.copy()

            try:
                # 使用 YOLO 模型直接推理（不走 detect_single 避免重复读文件）
                t_start = time.perf_counter()
                results = self.detector.model.predict(
                    frame_for_detection,
                    conf=self.detector.conf_threshold,
                    iou=self.detector.iou_threshold,
                    imgsz=self.detector.img_size,
                    device=self.detector.device,
                    half=self.detector.use_fp16,
                    verbose=False,
                )
                t_end = time.perf_counter()
                inference_time_ms = (t_end - t_start) * 1000

                # 解析检测结果
                result = self.detector._parse_results(
                    results[0], "<video_frame>", frame.shape
                )
                result.inference_time_ms = inference_time_ms

                # 在帧上绘制检测框（与 ImageDisplayWidget 相同样式）
                annotated_frame = self._draw_detections(frame, result.detections)

                # 在帧上绘制 FPS 和推理信息
                self._draw_info_overlay(annotated_frame, inference_time_ms)

                # 转换为 QPixmap
                pixmap = self._cv_frame_to_qpixmap(annotated_frame)

                self.frame_ready.emit(pixmap, result)

            except Exception as e:
                # 推理失败时发送原始帧
                pixmap = self._cv_frame_to_qpixmap(frame)
                self.frame_ready.emit(pixmap, None)

            # 计算 FPS
            fps_counter += 1
            elapsed = time.perf_counter() - fps_start_time
            if elapsed >= fps_display_interval:
                current_fps = fps_counter / elapsed
                self.fps_update.emit(current_fps)
                fps_counter = 0
                fps_start_time = time.perf_counter()

        cap.release()
        self.stream_finished.emit()

    def _draw_detections(self, frame, detections):
        """在帧上绘制检测框（与 ImageDisplayWidget.set_image_with_detections 相同样式）"""
        img = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]

            # 颜色：缺陷红色，合格绿色
            if det.is_defect:
                color_bgr = (0, 0, 255)
            else:
                color_bgr = (0, 200, 0)

            # 绘制边框
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

            # 标签
            label = f"{det.class_name_cn} {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 标签背景
            cv2.rectangle(
                img,
                (x1, y1 - th - baseline - 8),
                (x1 + tw + 4, y1),
                color_bgr, -1
            )

            # 标签文字
            cv2.putText(
                img, label,
                (x1 + 2, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        return img

    def _draw_info_overlay(self, frame, inference_time_ms):
        """在帧上绘制 FPS 和推理时间信息栏"""
        h, w = frame.shape[:2]

        # 顶部信息栏背景
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)

        # 推理时间
        info_text = f"Inference: {inference_time_ms:.1f}ms"
        cv2.putText(
            frame, info_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

        # 分辨率信息
        res_text = f"{w}x{h}"
        cv2.putText(
            frame, res_text,
            (w - 120, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (150, 150, 150), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _cv_frame_to_qpixmap(frame):
        """将 OpenCV BGR 帧转换为 QPixmap"""
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def stop(self):
        """安全停止线程"""
        self._is_running = False


# ============================================================
# 检测工作线程
# ============================================================

class DetectionThread(QThread):
    """后台检测线程，避免阻塞UI"""

    progress = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(object, object)  # image_path, result
    finished_all = pyqtSignal(list)  # all results
    error_occurred = pyqtSignal(str)

    def __init__(self, detector, image_paths, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.image_paths = image_paths
        self._is_running = True

    def run(self):
        results = []
        total = len(self.image_paths)

        for i, img_path in enumerate(self.image_paths):
            if not self._is_running:
                break

            try:
                result = self.detector.detect_single(str(img_path))
                results.append(result)
                self.result_ready.emit(str(img_path), result)
                self.progress.emit(i + 1, total)
            except Exception as e:
                self.error_occurred.emit(f"检测失败 {img_path}: {str(e)}")

        if self._is_running:
            self.finished_all.emit(results)

    def stop(self):
        self._is_running = False


# ============================================================
# 图片显示组件（支持绘制检测框）
# ============================================================

class ImageDisplayWidget(QLabel):
    """支持拖拽和检测框绘制的图片显示组件"""

    image_dropped = pyqtSignal(list)  # 拖入的文件路径列表

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setObjectName("imageFrame")
        self.setAcceptDrops(True)

        self._original_pixmap = None
        self._detections = []
        self._image_path = ""

        # 默认显示
        self._show_placeholder()

    def _show_placeholder(self):
        """显示占位提示"""
        self.setText("拖拽图片到此处\n或点击工具栏打开图片")
        self.setObjectName("dropLabel")
        self.setStyleSheet("")

    def set_image(self, image_path: str):
        """设置要显示的图片"""
        self._image_path = image_path
        self._original_pixmap = QPixmap(image_path)

        if self._original_pixmap.isNull():
            self.setText(f"无法加载图片: {image_path}")
            return

        self._update_display()

    def set_image_with_detections(self, image_path: str, result):
        """设置图片并绘制检测结果"""
        self._image_path = image_path
        self._detections = result.detections if result else []

        # 读取原图并绘制检测框
        img = cv2.imread(image_path)
        if img is None:
            self.setText(f"无法加载图片: {image_path}")
            return

        # 绘制检测框
        for det in self._detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]

            # 选择颜色 (BGR)
            if det.is_defect:
                color_bgr = (0, 0, 255)  # 红色
            else:
                color_bgr = (0, 200, 0)  # 绿色

            # 绘制边框
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

            # 标签
            label = f"{det.class_name_cn} {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 标签背景
            cv2.rectangle(
                img,
                (x1, y1 - th - baseline - 8),
                (x1 + tw + 4, y1),
                color_bgr, -1
            )

            # 标签文字
            cv2.putText(
                img, label,
                (x1 + 2, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        # 转换为 QPixmap
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._original_pixmap = QPixmap.fromImage(q_img)

        self._update_display()

    def set_frame(self, pixmap: QPixmap):
        """直接设置 QPixmap（用于视频流帧，跳过文件读取）"""
        self._original_pixmap = pixmap
        self._detections = []
        self._image_path = ""
        self._update_display()

    def _update_display(self):
        """更新显示（自适应缩放）"""
        if self._original_pixmap is None:
            return

        # 获取可用大小
        available_size = self.size() - QSize(20, 20)

        # 缩放图片
        scaled = self._original_pixmap.scaled(
            available_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled)

    def resizeEvent(self, event):
        """窗口大小改变时重新缩放"""
        super().resizeEvent(event)
        self._update_display()

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """放下事件"""
        urls = event.mimeData().urls()
        file_paths = []

        for url in urls:
            path = url.toLocalFile()
            if os.path.isfile(path):
                # 检查是否是图片文件
                ext = Path(path).suffix.lower()
                if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
                    file_paths.append(path)
            elif os.path.isdir(path):
                file_paths.append(path)

        if file_paths:
            self.image_dropped.emit(file_paths)


# ============================================================
# 设置对话框
# ============================================================

class SettingsDialog(QDialog):
    """设置对话框"""

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.settings = settings.copy()
        self.setWindowTitle("设置")
        self.setMinimumWidth(450)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout()

        self.model_path_edit = QLabel(self.settings.get("model_path", "自动查找"))
        model_btn = QPushButton("选择模型")
        model_btn.clicked.connect(self._select_model)

        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_btn)

        model_layout.addRow("模型路径:", model_path_layout)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.settings.get("conf_threshold", 0.25))
        model_layout.addRow("置信度阈值:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(self.settings.get("iou_threshold", 0.45))
        model_layout.addRow("IoU阈值:", self.iou_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(self.settings.get("img_size", 640))
        model_layout.addRow("输入尺寸:", self.imgsz_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 检测设置
        detect_group = QGroupBox("检测设置")
        detect_layout = QFormLayout()

        self.voice_check = QCheckBox("启用语音播报")
        self.voice_check.setChecked(self.settings.get("enable_voice", False))
        detect_layout.addRow(self.voice_check)

        self.log_check = QCheckBox("启用检测日志")
        self.log_check.setChecked(self.settings.get("enable_log", True))
        detect_layout.addRow(self.log_check)

        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)

        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _select_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.onnx);;所有文件 (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _on_accept(self):
        self.settings["model_path"] = self.model_path_edit.text()
        self.settings["conf_threshold"] = self.conf_spin.value()
        self.settings["iou_threshold"] = self.iou_spin.value()
        self.settings["img_size"] = self.imgsz_spin.value()
        self.settings["enable_voice"] = self.voice_check.isChecked()
        self.settings["enable_log"] = self.log_check.isChecked()
        self.accept()

    def get_settings(self):
        return self.settings


# ============================================================
# 帮助对话框
# ============================================================

class HelpDialog(QDialog):
    """帮助对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("帮助")
        self.setMinimumSize(500, 400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>螺丝缺陷检测系统 - 使用说明</h2>

        <h3>基本操作</h3>
        <ul>
            <li><b>打开图片</b>: 点击工具栏"打开图片"按钮或使用菜单 文件 > 打开图片</li>
            <li><b>批量检测</b>: 点击"打开文件夹"或拖拽文件夹到窗口</li>
            <li><b>拖拽图片</b>: 直接拖拽图片文件到检测区域</li>
        </ul>

        <h3>视频流检测</h3>
        <ul>
            <li><b>摄像头检测</b>: 点击工具栏"打开摄像头"或使用 Ctrl+Shift+C</li>
            <li><b>视频文件检测</b>: 点击工具栏"打开视频"或使用 Ctrl+Shift+V</li>
            <li><b>停止视频</b>: 点击"停止视频"按钮或按 Escape 键</li>
        </ul>

        <h3>检测结果</h3>
        <ul>
            <li>所有检出框均为缺陷，无检出 = 合格</li>
            <li><font color="red">红色框</font>: 缺陷品</li>
        </ul>

        <h3>缺陷类型</h3>
        <ul>
            <li><b>正常</b>: 无缺陷</li>
            <li><b>轻微划痕</b>: 表面轻微划痕</li>
            <li><b>严重划痕</b>: 表面严重划痕</li>
            <li><b>缺角</b>: 螺丝头部缺角</li>
            <li><b>变形</b>: 螺丝变形</li>
            <li><b>混料</b>: 混入其他材料</li>
        </ul>

        <h3>演示模式</h3>
        <p>点击工具栏"180秒计时"按钮启动比赛演示模式，系统会在180秒倒计时内完成检测。</p>

        <h3>快捷键</h3>
        <ul>
            <li><b>Ctrl+O</b>: 打开图片</li>
            <li><b>Ctrl+Shift+O</b>: 打开文件夹</li>
            <li><b>Ctrl+Shift+C</b>: 打开摄像头</li>
            <li><b>Ctrl+Shift+V</b>: 打开视频文件</li>
            <li><b>Escape</b>: 停止视频流</li>
            <li><b>F5</b>: 开始检测</li>
            <li><b>Ctrl+E</b>: 导出报告</li>
            <li><b>Delete</b>: 清除结果</li>
        </ul>

        <h3>关于</h3>
        <p>螺丝缺陷检测系统 v2.1.0<br>
        基于 YOLOv8 深度学习模型</p>
        """)

        layout.addWidget(help_text)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


# ============================================================
# 主窗口
# ============================================================

class ScrewDefectApp(QMainWindow):
    """螺丝缺陷检测主应用窗口"""

    def __init__(self):
        super().__init__()

        # 检测器（延迟初始化）
        self.detector = None
        self.detection_thread = None

        # 视频流状态
        self.video_thread = None
        self.is_video_streaming = False

        # 当前状态
        self.current_image_path = ""
        self.current_result = None
        self.all_results = []
        self.is_detecting = False

        # 设置
        self.settings = {
            "model_path": "",
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "img_size": 640,
            "enable_voice": False,
            "enable_log": True,
        }

        # 计时器模式
        self.timer_mode = False
        self.timer_seconds = 180
        self.timer_remaining = 180
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._update_timer)

        # 初始化UI
        self._init_ui()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()
        self._connect_signals()

        # 初始化检测器
        self._init_detector()

    def _init_ui(self):
        """初始化主界面"""
        self.setWindowTitle(f"{APP_TITLE} - {APP_TITLE_CN}")
        self.setMinimumSize(*WINDOW_MIN_SIZE)
        self.setStyleSheet(STYLE_SHEET)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 顶部标题
        title_layout = QHBoxLayout()
        title_label = QLabel(APP_TITLE)
        title_label.setObjectName("titleLabel")
        title_layout.addWidget(title_label)

        # 版本徽章
        version_badge = QLabel(f"v{APP_VERSION}")
        version_badge.setObjectName("versionLabel")
        title_layout.addWidget(version_badge)
        title_layout.addStretch()

        # 计时器显示
        self.timer_label = QLabel("")
        self.timer_label.setStyleSheet("""
            QLabel {
                font-size: 26px;
                font-weight: 700;
                color: #ef476f;
                padding: 6px 18px;
                background-color: #fff0f3;
                border: 2px solid #ef476f;
                border-radius: 8px;
            }
        """)
        self.timer_label.hide()
        title_layout.addWidget(self.timer_label)

        main_layout.addLayout(title_layout)

        # 主内容区域（左右分割）
        content_splitter = QSplitter(Qt.Horizontal)

        # ---- 左侧：图片显示区域 ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)

        # 图片显示
        self.image_display = ImageDisplayWidget()
        left_layout.addWidget(self.image_display)

        # 图片信息
        self.image_info_label = QLabel("未加载图片")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setStyleSheet("color: #888; padding: 6px; font-size: 13px;")
        left_layout.addWidget(self.image_info_label)

        content_splitter.addWidget(left_widget)

        # ---- 右侧：结果面板 ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)

        # 标签页
        tab_widget = QTabWidget()

        # 检测结果标签页
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels([
            "序号", "类别", "置信度", "状态", "位置"
        ])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        result_layout.addWidget(self.result_table)

        tab_widget.addTab(result_tab, "检测结果")

        # 统计信息标签页
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        # 统计卡片
        stats_grid = QGridLayout()

        # 合格数量
        pass_card = self._create_stat_card("合格数量", "0", "#27ae60")
        stats_grid.addWidget(pass_card, 0, 0)

        # 不合格数量
        fail_card = self._create_stat_card("不合格数量", "0", "#e74c3c")
        stats_grid.addWidget(fail_card, 0, 1)

        # 检测总数
        total_card = self._create_stat_card("检测总数", "0", "#3498db")
        stats_grid.addWidget(total_card, 1, 0)

        # 合格率
        rate_card = self._create_stat_card("合格率", "0%", "#f39c12")
        stats_grid.addWidget(rate_card, 1, 1)

        stats_layout.addLayout(stats_grid)

        # 缺陷分布
        defect_group = QGroupBox("缺陷类型分布")
        defect_layout = QVBoxLayout()
        self.defect_bars_layout = QVBoxLayout()
        defect_layout.addLayout(self.defect_bars_layout)
        defect_group.setLayout(defect_layout)
        stats_layout.addWidget(defect_group)

        stats_layout.addStretch()

        tab_widget.addTab(stats_tab, "统计信息")

        right_layout.addWidget(tab_widget)

        # 操作按钮区域
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)

        self.clear_btn = QPushButton("清除结果")
        self.clear_btn.setObjectName("clearBtn")
        btn_layout.addWidget(self.clear_btn)

        right_layout.addLayout(btn_layout)

        content_splitter.addWidget(right_widget)

        # 设置分割比例
        content_splitter.setSizes([700, 400])
        main_layout.addWidget(content_splitter)

        # 保存统计卡片引用
        self._pass_count_label = pass_card.findChild(QLabel, "valueLabel")
        self._fail_count_label = fail_card.findChild(QLabel, "valueLabel")
        self._total_count_label = total_card.findChild(QLabel, "valueLabel")
        self._rate_label = rate_card.findChild(QLabel, "valueLabel")

    def _create_stat_card(self, title: str, value: str, color: str) -> QFrame:
        """创建统计卡片"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #ffffff;
                border: 1px solid #e8e8e8;
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 12px;
                margin: 4px;
            }}
            QFrame:hover {{
                border: 1px solid {color};
                border-left: 4px solid {color};
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888; font-size: 12px; font-weight: 500;")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName("valueLabel")
        value_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: 700;")
        layout.addWidget(value_label)

        return card

    def _init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        open_image_action = QAction("打开图片(&O)", self)
        open_image_action.setShortcut("Ctrl+O")
        open_image_action.triggered.connect(self._open_image)
        file_menu.addAction(open_image_action)

        open_folder_action = QAction("打开文件夹(&D)", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        # 视频检测子菜单
        video_menu = file_menu.addMenu("视频检测(&V)")

        self.camera_menu_action = QAction("打开摄像头(&C)", self)
        self.camera_menu_action.setShortcut("Ctrl+Shift+C")
        self.camera_menu_action.triggered.connect(self._start_camera)
        video_menu.addAction(self.camera_menu_action)

        self.video_file_menu_action = QAction("打开视频文件(&F)", self)
        self.video_file_menu_action.setShortcut("Ctrl+Shift+V")
        self.video_file_menu_action.triggered.connect(self._open_video_file)
        video_menu.addAction(self.video_file_menu_action)

        video_menu.addSeparator()

        self.stop_video_menu_action = QAction("停止视频(&S)", self)
        self.stop_video_menu_action.setShortcut("Escape")
        self.stop_video_menu_action.triggered.connect(self._stop_video)
        self.stop_video_menu_action.setEnabled(False)
        video_menu.addAction(self.stop_video_menu_action)

        file_menu.addSeparator()

        export_action = QAction("导出报告(&E)", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_report)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 设置菜单
        settings_menu = menubar.addMenu("设置(&S)")

        settings_action = QAction("参数设置(&P)", self)
        settings_action.triggered.connect(self._show_settings)
        settings_menu.addAction(settings_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")

        help_action = QAction("使用说明(&H)", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self._show_help)
        help_menu.addAction(help_action)

        help_menu.addSeparator()

        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_toolbar(self):
        """初始化工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # ---- 图片检测按钮组 ----
        open_btn = QPushButton("  打开图片")
        open_btn.clicked.connect(self._open_image)
        toolbar.addWidget(open_btn)

        folder_btn = QPushButton("  打开文件夹")
        folder_btn.clicked.connect(self._open_folder)
        toolbar.addWidget(folder_btn)

        toolbar.addSeparator()

        # 开始检测
        self.detect_btn = QPushButton("  开始检测")
        self.detect_btn.setObjectName("startBtn")
        self.detect_btn.clicked.connect(self._start_detection)
        toolbar.addWidget(self.detect_btn)

        # 清除结果
        clear_btn = QPushButton("  清除")
        clear_btn.setObjectName("clearBtn")
        clear_btn.clicked.connect(self._clear_results)
        toolbar.addWidget(clear_btn)

        toolbar.addSeparator()

        # 导出报告
        export_btn = QPushButton("  导出报告")
        export_btn.clicked.connect(self._export_report)
        toolbar.addWidget(export_btn)

        toolbar.addSeparator()

        # ---- 视频检测按钮组 ----
        self.camera_btn = QPushButton("  摄像头")
        self.camera_btn.setObjectName("cameraBtn")
        self.camera_btn.clicked.connect(self._start_camera)
        toolbar.addWidget(self.camera_btn)

        self.video_file_btn = QPushButton("  视频文件")
        self.video_file_btn.setObjectName("videoBtn")
        self.video_file_btn.clicked.connect(self._open_video_file)
        toolbar.addWidget(self.video_file_btn)

        self.stop_video_btn = QPushButton("  停止视频")
        self.stop_video_btn.setObjectName("stopVideoBtn")
        self.stop_video_btn.clicked.connect(self._stop_video)
        self.stop_video_btn.hide()
        toolbar.addWidget(self.stop_video_btn)

        # FPS 显示标签
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fpsLabel")
        self.fps_label.hide()
        toolbar.addWidget(self.fps_label)

        toolbar.addSeparator()

        # 180秒计时模式
        self.timer_btn = QPushButton("  180秒计时")
        self.timer_btn.setObjectName("timerBtn")
        self.timer_btn.clicked.connect(self._toggle_timer_mode)
        toolbar.addWidget(self.timer_btn)

        toolbar.addSeparator()

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        toolbar.addWidget(self.progress_bar)

        # ---- 右侧：主题切换 ----
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        self.theme_btn = QPushButton("  ")
        self.theme_btn.setObjectName("themeBtn")
        self.theme_btn.setToolTip("切换深色/浅色主题")
        self.theme_btn.clicked.connect(self._toggle_theme)
        toolbar.addWidget(self.theme_btn)

        self.is_dark_theme = True
        self._update_theme_button()

    def _init_statusbar(self):
        """初始化状态栏"""
        self.statusBar().showMessage("就绪")

        # 状态指示器 - 用圆点 + 文字
        self.model_info_label = QLabel("  模型: 未加载")
        self.model_info_label.setStyleSheet("padding: 0 8px; font-size: 12px;")
        self.statusBar().addPermanentWidget(self.model_info_label)

        self.device_info_label = QLabel("  设备: CPU")
        self.device_info_label.setStyleSheet("padding: 0 8px; font-size: 12px;")
        self.statusBar().addPermanentWidget(self.device_info_label)

        # 分隔线
        sep = QLabel("|")
        sep.setStyleSheet("color: #2a2d3a; padding: 0 4px;")
        self.statusBar().addPermanentWidget(sep)

        # 版本标签
        ver_label = QLabel(f"v{APP_VERSION}")
        ver_label.setStyleSheet("color: #475569; padding: 0 8px; font-size: 11px;")
        self.statusBar().addPermanentWidget(ver_label)

    def _toggle_theme(self):
        """切换深色/浅色主题"""
        self.is_dark_theme = not self.is_dark_theme
        if self.is_dark_theme:
            self.setStyleSheet(DARK_THEME)
        else:
            self.setStyleSheet(LIGHT_THEME)
        self._update_theme_button()

    def _update_theme_button(self):
        """更新主题切换按钮显示"""
        if self.is_dark_theme:
            self.theme_btn.setText("  ")
        else:
            self.theme_btn.setText("  ")

    def _connect_signals(self):
        """连接信号"""
        self.start_btn.clicked.connect(self._start_detection)
        self.clear_btn.clicked.connect(self._clear_results)
        self.image_display.image_dropped.connect(self._handle_dropped_files)

    def _init_detector(self):
        """初始化检测器"""
        try:
            from inference import ScrewDefectDetector, find_best_model

            # 查找模型
            model_path = self.settings["model_path"]
            if not model_path or not Path(model_path).exists():
                best = find_best_model()
                if best:
                    model_path = str(best)
                else:
                    model_path = "yolov8n.pt"

            self.statusBar().showMessage(f"正在加载模型: {model_path}...")

            # 创建检测器
            self.detector = ScrewDefectDetector(
                model_path=model_path,
                conf_threshold=self.settings["conf_threshold"],
                iou_threshold=self.settings["iou_threshold"],
                img_size=self.settings["img_size"],
            )

            # 更新状态栏
            device_info = self.detector.get_device_info()
            self.model_info_label.setText(f"模型: {Path(model_path).name}")
            self.device_info_label.setText(f"设备: {device_info['device'].upper()}")

            self.statusBar().showMessage("模型加载完成", 3000)

        except Exception as e:
            self.statusBar().showMessage(f"模型加载失败: {str(e)}")
            QMessageBox.warning(
                self, "警告",
                f"模型加载失败:\n{str(e)}\n\n请检查模型文件是否存在。"
            )

    def _open_image(self):
        """打开单张图片"""
        if self.is_video_streaming:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
        )

        if file_path:
            self._load_image(file_path)

    def _open_folder(self):
        """打开文件夹"""
        if self.is_video_streaming:
            return

        folder_path = QFileDialog.getExistingDirectory(
            self, "选择图片文件夹"
        )

        if folder_path:
            self._load_folder(folder_path)

    def _load_image(self, image_path: str):
        """加载单张图片"""
        self.current_image_path = image_path
        self.image_display.set_image(image_path)

        # 更新图片信息
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            fname = Path(image_path).name
            self.image_info_label.setText(f"{fname} | {w}x{h}")

        # 启用检测按钮
        self.start_btn.setEnabled(True)
        self.statusBar().showMessage(f"已加载图片: {Path(image_path).name}")

    def _load_folder(self, folder_path: str):
        """加载文件夹中的所有图片"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(Path(folder_path).glob(f"*{ext}"))
            image_paths.extend(Path(folder_path).glob(f"*{ext.upper()}"))

        image_paths = sorted(set(image_paths))

        if not image_paths:
            QMessageBox.information(self, "提示", "文件夹中未找到图片文件")
            return

        # 存储待检测图片列表
        self._pending_images = [str(p) for p in image_paths]
        self.current_image_path = self._pending_images[0]

        # 显示第一张图片
        self.image_display.set_image(self.current_image_path)
        self.image_info_label.setText(
            f"已加载 {len(image_paths)} 张图片 | 当前: {Path(self.current_image_path).name}"
        )

        # 启用检测按钮
        self.start_btn.setEnabled(True)
        self.statusBar().showMessage(f"已加载 {len(image_paths)} 张图片")

    def _handle_dropped_files(self, file_paths: list):
        """处理拖拽的文件"""
        if not file_paths:
            return

        if self.is_video_streaming:
            return

        first_path = file_paths[0]

        if os.path.isdir(first_path):
            self._load_folder(first_path)
        elif os.path.isfile(first_path):
            if len(file_paths) == 1:
                self._load_image(first_path)
            else:
                # 多个文件
                self._pending_images = file_paths
                self.current_image_path = file_paths[0]
                self._load_image(file_paths[0])
                self.image_info_label.setText(
                    f"已加载 {len(file_paths)} 张图片"
                )

    def _start_detection(self):
        """开始检测"""
        if self.is_detecting or self.is_video_streaming:
            return

        if not self.detector:
            QMessageBox.warning(self, "警告", "检测器未初始化，请检查模型配置")
            return

        # 获取待检测图片列表
        if hasattr(self, '_pending_images') and self._pending_images:
            image_paths = self._pending_images
        elif self.current_image_path:
            image_paths = [self.current_image_path]
        else:
            QMessageBox.information(self, "提示", "请先打开图片或文件夹")
            return

        # 开始检测
        self.is_detecting = True
        self.start_btn.setEnabled(False)
        self.progress_bar.setMaximum(len(image_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        # 清空之前的结果
        self.all_results = []
        self._update_stats()

        # 创建检测线程
        self.detection_thread = DetectionThread(
            self.detector, image_paths, self
        )
        self.detection_thread.progress.connect(self._on_detection_progress)
        self.detection_thread.result_ready.connect(self._on_detection_result)
        self.detection_thread.finished_all.connect(self._on_detection_finished)
        self.detection_thread.error_occurred.connect(self._on_detection_error)

        self.statusBar().showMessage("正在检测...")
        self.detection_thread.start()

    def _on_detection_progress(self, current: int, total: int):
        """检测进度更新"""
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"检测进度: {current}/{total}")

    def _on_detection_result(self, image_path: str, result):
        """单张检测完成"""
        self.all_results.append(result)

        # 如果是当前显示的图片，更新显示
        if image_path == self.current_image_path:
            self.current_result = result
            self.image_display.set_image_with_detections(image_path, result)
            self._update_result_table(result)

        # 更新统计
        self._update_stats()

    def _on_detection_finished(self, results: list):
        """所有检测完成"""
        self.is_detecting = False
        self.start_btn.setEnabled(True)
        self.progress_bar.hide()

        # 显示最后一张结果
        if results:
            self.current_result = results[-1]
            self._update_result_table(results[-1])

        self.statusBar().showMessage(
            f"检测完成: 共 {len(results)} 张图片", 5000
        )

        # 计时器模式下显示完成信息
        if self.timer_mode:
            elapsed = self.timer_seconds - self.timer_remaining
            QMessageBox.information(
                self, "检测完成",
                f"在 {elapsed} 秒内完成 {len(results)} 张图片的检测"
            )

    def _on_detection_error(self, error_msg: str):
        """检测错误"""
        print(f"[错误] {error_msg}")

    # ============================================================
    # 视频流检测方法
    # ============================================================

    def _start_camera(self):
        """启动摄像头视频流检测"""
        if self.is_video_streaming:
            return
        if not self.detector:
            QMessageBox.warning(self, "警告", "检测器未初始化，请检查模型配置")
            return
        self._start_video_stream(0)

    def _open_video_file(self):
        """打开视频文件进行检测"""
        if self.is_video_streaming:
            return
        if not self.detector:
            QMessageBox.warning(self, "警告", "检测器未初始化，请检查模型配置")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*)"
        )

        if file_path:
            self._start_video_stream(file_path)

    def _start_video_stream(self, source):
        """
        启动视频流检测线程

        参数:
            source: 摄像头索引(int=0) 或视频文件路径(str)
        """
        if self.is_video_streaming:
            return

        # 清空之前的检测结果
        self.all_results = []
        self._update_stats()
        self.result_table.setRowCount(0)

        # 更新 UI 状态
        self.is_video_streaming = True
        self._set_video_ui_state(True)

        # 确定显示名称
        if isinstance(source, int):
            source_name = f"摄像头 (camera {source})"
        else:
            source_name = Path(source).name

        self.image_info_label.setText(f"视频流: {source_name}")
        self.statusBar().showMessage(f"正在打开视频流: {source_name}...")

        # 创建并启动视频流线程
        self.video_thread = VideoStreamThread(
            self.detector, source, self
        )
        self.video_thread.frame_ready.connect(self._on_video_frame)
        self.video_thread.fps_update.connect(self._on_fps_update)
        self.video_thread.error_occurred.connect(self._on_video_error)
        self.video_thread.stream_finished.connect(self._on_video_stream_finished)
        self.video_thread.start()

    def _on_video_frame(self, pixmap, result):
        """接收到视频帧，更新显示"""
        # 直接设置帧到显示组件
        self.image_display.set_frame(pixmap)

        # 更新检测结果表格
        if result:
            self.current_result = result
            self._update_result_table(result)

            # 更新统计（视频流模式下累计所有帧的结果）
            self.all_results.append(result)
            self._update_stats()

    def _on_fps_update(self, fps):
        """更新 FPS 显示"""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_video_error(self, error_msg):
        """视频流错误"""
        QMessageBox.critical(self, "视频流错误", error_msg)
        self._stop_video()

    def _on_video_stream_finished(self):
        """视频流自然结束（视频文件播放完毕）"""
        self._stop_video()
        self.statusBar().showMessage("视频播放完毕", 5000)

    def _stop_video(self):
        """停止视频流检测"""
        if not self.is_video_streaming:
            return

        # 停止线程
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait(3000)  # 等待最多3秒
            self.video_thread = None

        # 恢复 UI 状态
        self.is_video_streaming = False
        self._set_video_ui_state(False)

        self.fps_label.setText("FPS: --")
        self.image_info_label.setText("视频流已停止")
        self.statusBar().showMessage("视频流检测已停止", 3000)

    def _set_video_ui_state(self, streaming: bool):
        """
        根据视频流状态切换 UI 控件

        参数:
            streaming: True 表示正在流式检测
        """
        # 视频按钮：流式时隐藏摄像头和视频文件按钮，显示停止按钮
        self.camera_btn.setVisible(not streaming)
        self.video_file_btn.setVisible(not streaming)
        self.stop_video_btn.setVisible(streaming)
        self.fps_label.setVisible(streaming)

        # 菜单项
        self.camera_menu_action.setEnabled(not streaming)
        self.video_file_menu_action.setEnabled(not streaming)
        self.stop_video_menu_action.setEnabled(streaming)

        # 图片检测按钮：流式时禁用
        self.start_btn.setEnabled(not streaming and bool(self.current_image_path))
        self.detect_btn.setEnabled(not streaming)
        self.clear_btn.setEnabled(not streaming)

    def _update_result_table(self, result):
        """更新检测结果表格"""
        self.result_table.setRowCount(len(result.detections))

        for i, det in enumerate(result.detections):
            # 序号
            self.result_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))

            # 类别
            class_item = QTableWidgetItem(det.class_name_cn)
            if det.is_defect:
                class_item.setForeground(QColor("#e74c3c"))
            else:
                class_item.setForeground(QColor("#27ae60"))
            self.result_table.setItem(i, 1, class_item)

            # 置信度
            conf_item = QTableWidgetItem(f"{det.confidence:.2%}")
            self.result_table.setItem(i, 2, conf_item)

            # 状态
            status = "缺陷" if det.is_defect else "合格"
            status_item = QTableWidgetItem(status)
            if det.is_defect:
                status_item.setForeground(QColor("#e74c3c"))
            else:
                status_item.setForeground(QColor("#27ae60"))
            self.result_table.setItem(i, 3, status_item)

            # 位置
            bbox_str = f"({det.bbox[0]:.0f},{det.bbox[1]:.0f},{det.bbox[2]:.0f},{det.bbox[3]:.0f})"
            self.result_table.setItem(i, 4, QTableWidgetItem(bbox_str))

    def _update_stats(self):
        """更新统计信息"""
        if not self.all_results:
            return

        total = len(self.all_results)
        pass_count = sum(1 for r in self.all_results if not r.has_defect)
        fail_count = total - pass_count
        rate = (pass_count / total * 100) if total > 0 else 0

        self._pass_count_label.setText(str(pass_count))
        self._fail_count_label.setText(str(fail_count))
        self._total_count_label.setText(str(total))
        self._rate_label.setText(f"{rate:.1f}%")

        # 更新缺陷分布
        self._update_defect_distribution()

    def _update_defect_distribution(self):
        """更新缺陷分布显示"""
        # 清空现有内容
        while self.defect_bars_layout.count():
            item = self.defect_bars_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 统计缺陷类型
        defect_counts = {}
        for result in self.all_results:
            for det in result.detections:
                if det.is_defect:
                    name = det.class_name_cn
                    defect_counts[name] = defect_counts.get(name, 0) + 1

        if not defect_counts:
            no_defect_label = QLabel("暂无缺陷数据")
            no_defect_label.setStyleSheet("color: #95a5a6; padding: 10px;")
            no_defect_label.setAlignment(Qt.AlignCenter)
            self.defect_bars_layout.addWidget(no_defect_label)
            return

        # 计算最大值用于归一化
        max_count = max(defect_counts.values())

        # 创建进度条显示
        for name, count in sorted(defect_counts.items(), key=lambda x: -x[1]):
            bar_layout = QHBoxLayout()

            name_label = QLabel(name)
            name_label.setFixedWidth(80)
            name_label.setStyleSheet("font-size: 12px;")
            bar_layout.addWidget(name_label)

            progress = QProgressBar()
            progress.setMaximum(max_count)
            progress.setValue(count)
            progress.setTextVisible(True)
            progress.setFormat(f"{count}")
            bar_layout.addWidget(progress)

            self.defect_bars_layout.addLayout(bar_layout)

    def _clear_results(self):
        """清除检测结果"""
        if self.is_video_streaming:
            return

        self.all_results = []
        self.current_result = None

        # 清空表格
        self.result_table.setRowCount(0)

        # 重置统计
        self._pass_count_label.setText("0")
        self._fail_count_label.setText("0")
        self._total_count_label.setText("0")
        self._rate_label.setText("0%")

        # 清空缺陷分布
        while self.defect_bars_layout.count():
            item = self.defect_bars_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.statusBar().showMessage("已清除所有结果")

    def _export_report(self):
        """导出检测报告"""
        if not self.all_results:
            QMessageBox.information(self, "提示", "没有检测结果可导出")
            return

        # 选择保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"检测报告_{timestamp}"

        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出报告",
            default_name,
            "文本文件 (*.txt);;CSV文件 (*.csv);;所有文件 (*)"
        )

        if not save_path:
            return

        try:
            if save_path.endswith('.csv'):
                self._export_csv(save_path)
            else:
                self._export_txt(save_path)

            QMessageBox.information(
                self, "导出成功",
                f"报告已保存到:\n{save_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "导出失败",
                f"导出报告失败:\n{str(e)}"
            )

    def _export_txt(self, save_path: str):
        """导出文本报告"""
        total = len(self.all_results)
        pass_count = sum(1 for r in self.all_results if not r.has_defect)
        fail_count = total - pass_count
        rate = (pass_count / total * 100) if total > 0 else 0

        lines = [
            "=" * 60,
            "螺丝缺陷检测报告",
            "=" * 60,
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"总样本数:       {total}",
            f"合格数:         {pass_count}",
            f"不合格数:       {fail_count}",
            f"合格率:         {rate:.1f}%",
            "",
            "-" * 60,
            "详细结果:",
            "-" * 60,
            "",
        ]

        for i, result in enumerate(self.all_results, 1):
            fname = Path(result.image_path).name
            lines.append(f"[{i}] {fname}")
            lines.append(f"    判定: {result.overall_verdict}")
            lines.append(f"    耗时: {result.inference_time_ms:.1f}ms")

            if result.detections:
                for j, det in enumerate(result.detections, 1):
                    status = "缺陷" if det.is_defect else "合格"
                    lines.append(
                        f"    检测{j}: {det.class_name_cn} "
                        f"({det.confidence:.2%}) [{status}]"
                    )
            else:
                lines.append("    未检测到目标")
            lines.append("")

        lines.append("=" * 60)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_csv(self, save_path: str):
        """导出CSV报告"""
        import csv

        with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 表头
            writer.writerow([
                "图片路径", "整体判定", "是否有缺陷", "缺陷数量",
                "推理耗时(ms)", "检测序号", "类别", "类别(中文)",
                "置信度", "状态", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
            ])

            # 数据
            for result in self.all_results:
                if result.detections:
                    for idx, det in enumerate(result.detections):
                        writer.writerow([
                            result.image_path,
                            result.overall_verdict,
                            result.has_defect,
                            result.defect_count,
                            f"{result.inference_time_ms:.1f}",
                            idx + 1,
                            det.class_name,
                            det.class_name_cn,
                            f"{det.confidence:.4f}",
                            "缺陷" if det.is_defect else "合格",
                            f"{det.bbox[0]:.1f}",
                            f"{det.bbox[1]:.1f}",
                            f"{det.bbox[2]:.1f}",
                            f"{det.bbox[3]:.1f}",
                        ])
                else:
                    writer.writerow([
                        result.image_path,
                        result.overall_verdict,
                        False,
                        0,
                        f"{result.inference_time_ms:.1f}",
                        0,
                        "none",
                        "无检测结果",
                        "0",
                        "无",
                        "0", "0", "0", "0"
                    ])

    def _toggle_timer_mode(self):
        """切换计时器模式"""
        if self.timer_mode:
            # 停止计时
            self.timer_mode = False
            self.countdown_timer.stop()
            self.timer_label.hide()
            self.timer_btn.setText("180秒计时")
            self.timer_btn.setStyleSheet("")
            self.statusBar().showMessage("计时模式已关闭")
        else:
            # 启动计时
            self.timer_mode = True
            self.timer_remaining = self.timer_seconds
            self._update_timer_display()
            self.timer_label.show()
            self.countdown_timer.start(1000)  # 每秒更新
            self.timer_btn.setText("停止计时")
            self.timer_btn.setStyleSheet("background-color: #e74c3c;")
            self.statusBar().showMessage("计时模式已启动 - 180秒倒计时")

    def _update_timer(self):
        """更新计时器"""
        self.timer_remaining -= 1
        self._update_timer_display()

        if self.timer_remaining <= 0:
            self.countdown_timer.stop()
            self.timer_mode = False
            self.timer_label.hide()
            self.timer_btn.setText("180秒计时")
            self.timer_btn.setStyleSheet("")
            QMessageBox.warning(self, "时间到", "180秒计时已结束")

    def _update_timer_display(self):
        """更新计时器显示"""
        minutes = self.timer_remaining // 60
        seconds = self.timer_remaining % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

        # 最后30秒变红闪烁效果
        if self.timer_remaining <= 30:
            if self.timer_remaining % 2 == 0:
                self.timer_label.setStyleSheet("""
                    QLabel {
                        font-size: 24px;
                        font-weight: bold;
                        color: white;
                        padding: 5px 15px;
                        background-color: #e74c3c;
                        border: 2px solid #c0392b;
                        border-radius: 5px;
                    }
                """)
            else:
                self.timer_label.setStyleSheet("""
                    QLabel {
                        font-size: 24px;
                        font-weight: bold;
                        color: #e74c3c;
                        padding: 5px 15px;
                        background-color: #fdf2f2;
                        border: 2px solid #e74c3c;
                        border-radius: 5px;
                    }
                """)

    def _show_settings(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec_() == QDialog.Accepted:
            new_settings = dialog.get_settings()

            # 检查是否需要重新加载模型
            need_reload = (
                new_settings["model_path"] != self.settings["model_path"] or
                new_settings["conf_threshold"] != self.settings["conf_threshold"] or
                new_settings["iou_threshold"] != self.settings["iou_threshold"] or
                new_settings["img_size"] != self.settings["img_size"]
            )

            self.settings = new_settings

            if need_reload:
                self._init_detector()

    def _show_help(self):
        """显示帮助"""
        dialog = HelpDialog(self)
        dialog.exec_()

    def _show_about(self):
        """显示关于"""
        QMessageBox.about(
            self, "关于",
            f"<h2 style='color:{COLOR_PRIMARY}'>{APP_TITLE}</h2>"
            f"<h4>{APP_TITLE_CN}</h4>"
            f"<p>版本 {APP_VERSION}</p>"
            "<p>基于 YOLOv8 深度学习模型的钢材表面缺陷检测系统</p>"
            "<p>支持 6 类缺陷：龟裂、夹杂、斑块、麻点、氧化皮、划痕</p>"
            "<hr>"
            "<p><b>功能特点：</b></p>"
            "<ul>"
            "<li>图片/文件夹批量检测</li>"
            "<li>视频流实时检测（摄像头/视频文件）</li>"
            "<li>180秒比赛计时模式</li>"
            "<li>深色/浅色主题切换</li>"
            "<li>检测报告导出</li>"
            "</ul>"
        )

    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key_Escape and self.is_video_streaming:
            self._stop_video()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """关闭事件"""
        # 先停止视频流
        if self.is_video_streaming:
            self._stop_video()

        if self.is_detecting:
            reply = QMessageBox.question(
                self, "确认退出",
                "检测正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                event.ignore()
                return

            # 停止检测线程
            if self.detection_thread:
                self.detection_thread.stop()
                self.detection_thread.wait()

        event.accept()


# ============================================================
# 应用入口
# ============================================================

def main():
    """应用主入口"""
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(APP_VERSION)

    # 设置应用图标（如果存在）
    icon_path = Path(__file__).parent / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # 创建并显示主窗口
    window = ScrewDefectApp()
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
