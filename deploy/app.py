"""
钢材缺陷检测系统 v3.0 - 现代化 UI
Steel Defect Detection System - Modern UI

基于 UI/UX 设计原则重构：
- 深色主题 + 工业风配色
- 卡片式布局
- 现代化按钮和表格
- 流畅动画过渡
- 响应式设计
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import cv2

from constants import CLASS_NAMES_CN, CLASS_NAMES
from inference import _put_cn_text

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAction, QToolBar, QFileDialog,
    QSplitter, QProgressBar, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QDialog, QFormLayout,
    QDialogButtonBox, QFrame, QSizePolicy,
    QScrollArea,
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal,
)
from PyQt5.QtGui import (
    QPixmap, QImage, QIcon, QFont, QColor,
    QDragEnterEvent, QDropEvent,
)


# ============================================================
# 常量定义
# ============================================================

APP_TITLE = "Steel Defect Inspector"
APP_TITLE_CN = "钢材缺陷检测系统"
APP_VERSION = "3.0.0"
WINDOW_MIN_SIZE = (1200, 800)

# ── 极简深色配色 ──
COLOR_BG             = "#111111"   # 统一背景
COLOR_BG_ALT         = "#1a1a1a"   # 次级背景
COLOR_SURFACE        = "#222222"   # 表面/卡片
COLOR_BORDER         = "#2a2a2a"   # 边框

COLOR_TEXT            = "#e0e0e0"   # 主文字
COLOR_TEXT_DIM        = "#888888"   # 次文字
COLOR_TEXT_FAINT      = "#555555"   # 弱文字

COLOR_ACCENT         = "#4a9eff"   # 唯一强调色
COLOR_OK             = "#4ade80"   # 合格
COLOR_ERR            = "#f87171"   # 不合格
COLOR_WARN           = "#fbbf24"   # 警告
COLOR_ERR_BG         = "#3d1a1a"   # 红色背景（计时器警告）
COLOR_WARN_BG         = "#3d2e0a"   # 黄色背景（计时器提示）


# ============================================================
# 样式表 - 极简风格
# ============================================================

MODERN_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
    font-size: 13px;
}}

/* ── 菜单栏 ── */
QMenuBar {{
    background: {COLOR_BG_ALT};
    color: {COLOR_TEXT_DIM};
    padding: 2px 8px;
    border-bottom: 1px solid {COLOR_BORDER};
}}
QMenuBar::item {{
    padding: 6px 12px;
}}
QMenuBar::item:selected {{
    background: {COLOR_SURFACE};
    color: {COLOR_TEXT};
}}

QMenu {{
    background: {COLOR_BG_ALT};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 8px 20px;
}}
QMenu::item:selected {{
    background: {COLOR_SURFACE};
}}
QMenu::separator {{
    height: 1px;
    background: {COLOR_BORDER};
    margin: 4px 8px;
}}

/* ── 工具栏 ── */
QToolBar {{
    background: {COLOR_BG_ALT};
    border-bottom: 1px solid {COLOR_BORDER};
    spacing: 4px;
    padding: 4px 8px;
}}

/* ── 按钮 ── */
QPushButton {{
    background: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    padding: 8px 16px;
    font-size: 13px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {COLOR_BORDER};
    border-color: {COLOR_TEXT_FAINT};
}}
QPushButton:pressed {{
    background: {COLOR_ACCENT};
    color: #fff;
    border-color: {COLOR_ACCENT};
}}
QPushButton:disabled {{
    color: {COLOR_TEXT_FAINT};
    border-color: {COLOR_BORDER};
}}

QPushButton#startBtn {{
    background: {COLOR_OK};
    color: #111;
    border: none;
    font-weight: 600;
    padding: 8px 24px;
}}
QPushButton#startBtn:hover {{
    background: #5cf098;
}}

QPushButton#clearBtn {{
    background: transparent;
    border: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT_DIM};
}}
QPushButton#clearBtn:hover {{
    color: {COLOR_ERR};
    border-color: {COLOR_ERR};
}}

QPushButton#timerBtn {{
    background: {COLOR_WARN};
    color: #111;
    border: none;
    font-weight: 600;
}}

QPushButton#cameraBtn, QPushButton#videoBtn {{
    background: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
}}

QPushButton#stopVideoBtn {{
    background: {COLOR_ERR};
    color: #fff;
    border: none;
}}

QPushButton#snapshotBtn {{
    background: {COLOR_ACCENT};
    color: #fff;
    border: none;
    font-weight: 600;
}}

/* ── 卡片 ── */
QFrame#card {{
    background: {COLOR_BG_ALT};
    border: 1px solid {COLOR_BORDER};
}}

/* ── 统计卡片 ── */
QFrame#statCard {{
    background: {COLOR_BG_ALT};
    border: 1px solid {COLOR_BORDER};
}}
QFrame#statCard[cardType="pass"] {{
    border-left: 2px solid {COLOR_OK};
}}
QFrame#statCard[cardType="fail"] {{
    border-left: 2px solid {COLOR_ERR};
}}
QFrame#statCard[cardType="total"] {{
    border-left: 2px solid {COLOR_ACCENT};
}}
QFrame#statCard[cardType="rate"] {{
    border-left: 2px solid {COLOR_WARN};
}}

/* ── 表格 ── */
QTableWidget {{
    background: {COLOR_BG_ALT};
    alternate-background: {COLOR_BG_ALT};
    border: 1px solid {COLOR_BORDER};
    gridline-color: {COLOR_BORDER};
    selection-background: {COLOR_SURFACE};
    font-size: 12px;
}}
QTableWidget::item {{
    padding: 6px 8px;
    border-bottom: 1px solid {COLOR_BORDER};
    background: {COLOR_BG_ALT};
}}
QHeaderView::section {{
    background: {COLOR_SURFACE};
    color: {COLOR_TEXT_DIM};
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid {COLOR_BORDER};
    font-size: 11px;
}}

/* ── 进度条 ── */
QProgressBar {{
    background: {COLOR_SURFACE};
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {COLOR_ACCENT};
}}

/* ── 状态栏 ── */
QStatusBar {{
    background: {COLOR_BG_ALT};
    color: {COLOR_TEXT_FAINT};
    border-top: 1px solid {COLOR_BORDER};
    padding: 2px 8px;
    font-size: 11px;
}}

/* ── 标签 ── */
QLabel {{
    color: {COLOR_TEXT};
}}
QLabel#timerLabel {{
    font-size: 14px;
    font-weight: 600;
    color: {COLOR_WARN};
}}

/* ── 滚动条 ── */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
}}
QScrollBar::handle:vertical {{
    background: {COLOR_BORDER};
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLOR_TEXT_FAINT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
}}
QScrollBar::handle:horizontal {{
    background: {COLOR_BORDER};
    min-width: 30px;
}}

/* ── 对话框 ── */
QDialog, QMessageBox {{
    background: {COLOR_BG_ALT};
}}

/* ── 输入框 ── */
QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    padding: 4px 8px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

/* ── 文本编辑 ── */
QTextEdit {{
    background: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    padding: 8px;
}}

/* ── 复选框 ── */
QCheckBox {{
    color: {COLOR_TEXT};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLOR_BORDER};
    background: {COLOR_SURFACE};
}}
QCheckBox::indicator:checked {{
    background: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
}}

/* ── 分割器 ── */
QSplitter::handle {{
    background: {COLOR_BORDER};
}}
QSplitter::handle:hover {{
    background: {COLOR_ACCENT};
}}
"""


# ============================================================
# 主应用类
# ============================================================

class SteelDefectApp(QMainWindow):
    """钢材缺陷检测主应用窗口 - 现代化 UI"""

    def __init__(self):
        super().__init__()

        # 检测器（延迟初始化）
        self.detector = None
        self.detection_thread = None

        # 视频流状态
        self.video_thread = None
        self.is_video_streaming = False
        self.current_video_frame = None
        self.current_video_result = None

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
        """初始化主界面 —— 左右分栏布局"""
        self.setWindowTitle(f"{APP_TITLE} - {APP_TITLE_CN}")
        self.setMinimumSize(*WINDOW_MIN_SIZE)

        icon_path = Path(__file__).parent / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 4, 8, 8)
        main_layout.setSpacing(6)

        # ── 顶部标题栏 ──
        header = QHBoxLayout()
        header.setSpacing(6)

        title = QLabel(f"{APP_TITLE_CN}")
        title.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px;")
        title.setFixedHeight(16)
        header.addWidget(title)

        header.addStretch()

        self.timer_label = QLabel("180s")
        self.timer_label.setObjectName("timerLabel")
        self.timer_label.setFixedHeight(16)
        self.timer_label.hide()
        header.addWidget(self.timer_label)

        main_layout.addLayout(header)

        # ── 主内容区：左右分栏 ──
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(3)

        # 左侧：预览区（图片/视频）
        preview_panel = self._create_preview_panel()
        content_splitter.addWidget(preview_panel)

        # 右侧：控制 & 结果面板
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)

        content_splitter.setStretchFactor(0, 3)  # 预览占 75%
        content_splitter.setStretchFactor(1, 1)  # 右侧面板占 25%
        content_splitter.setSizes([900, 310])
        main_layout.addWidget(content_splitter)

    def _create_preview_panel(self):
        """左侧预览面板 — 最大化显示区域"""
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 图片/视频显示（占满可用空间）
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 300)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_display.setStyleSheet(f"""
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 16px;
            color: {COLOR_TEXT_DIM};
            font-size: 14px;
        """)
        self.image_display.setText("拖拽图片到此处\n或通过工具栏上传")
        self.image_display.setAcceptDrops(True)
        layout.addWidget(self.image_display)

        # 底部信息栏（单行）
        info_bar = QHBoxLayout()
        info_bar.setSpacing(8)
        self.image_info_label = QLabel("")
        self.image_info_label.setStyleSheet(f"color: {COLOR_TEXT_FAINT}; font-size: 10px;")
        info_bar.addWidget(self.image_info_label)
        info_bar.addStretch()
        layout.addLayout(info_bar)

        return panel

    def _create_right_panel(self):
        """右侧面板 — 控制 & 结果（可滚动）"""
        outer = QFrame()
        outer.setObjectName("card")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: transparent; border: none; }}")

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # ── 1. 检测模式 ──
        mode_frame = QFrame()
        mode_frame.setObjectName("card")
        mode_layout = QVBoxLayout(mode_frame)
        mode_layout.setContentsMargins(10, 8, 10, 8)
        mode_layout.setSpacing(6)
        mode_title = QLabel("检测模式")
        mode_title.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px; font-weight: 600;")
        mode_layout.addWidget(mode_title)
        self.right_mode_combo = QComboBox()
        self.right_mode_combo.addItems(["图片检测", "摄像头检测", "视频检测"])
        self.right_mode_combo.currentIndexChanged.connect(self._on_right_mode_changed)
        mode_layout.addWidget(self.right_mode_combo)
        # 图片模式操作按钮
        img_actions = QHBoxLayout()
        img_actions.setSpacing(4)
        open_btn2 = QPushButton("打开图片")
        open_btn2.clicked.connect(self._open_image)
        img_actions.addWidget(open_btn2)
        open_dir_btn2 = QPushButton("文件夹")
        open_dir_btn2.clicked.connect(self._open_directory)
        img_actions.addWidget(open_dir_btn2)
        self.start_btn2 = QPushButton("开始检测")
        self.start_btn2.setObjectName("startBtn")
        self.start_btn2.clicked.connect(self._start_detection)
        self.start_btn2.setEnabled(False)
        img_actions.addWidget(self.start_btn2)
        self.clear_btn2 = QPushButton("清除")
        self.clear_btn2.setObjectName("clearBtn")
        self.clear_btn2.clicked.connect(self._clear_results)
        img_actions.addWidget(self.clear_btn2)
        self.right_image_actions = img_actions
        mode_layout.addLayout(img_actions)
        # 摄像头模式操作按钮
        cam_actions = QHBoxLayout()
        cam_actions.setSpacing(4)
        self.right_camera_combo = QComboBox()
        self.right_camera_combo.setFixedWidth(140)
        cam_actions.addWidget(self.right_camera_combo)
        self.right_camera_btn = QPushButton("启动")
        self.right_camera_btn.setObjectName("cameraBtn")
        self.right_camera_btn.clicked.connect(self._start_camera)
        cam_actions.addWidget(self.right_camera_btn)
        self.right_video_btn = QPushButton("选择视频")
        self.right_video_btn.setObjectName("videoBtn")
        self.right_video_btn.clicked.connect(self._open_video_file)
        cam_actions.addWidget(self.right_video_btn)
        self.right_stop_btn = QPushButton("停止")
        self.right_stop_btn.setObjectName("stopVideoBtn")
        self.right_stop_btn.clicked.connect(self._stop_video)
        self.right_stop_btn.hide()
        cam_actions.addWidget(self.right_stop_btn)
        self.right_cam_actions = cam_actions
        mode_layout.addLayout(cam_actions)
        # 扫描摄像头填充下拉
        self._scan_right_cameras()
        layout.addWidget(mode_frame)

        # ── 2. 统计卡片（2×2 网格） ──
        stats_frame = QFrame()
        stats_frame.setObjectName("card")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.setSpacing(4)

        self._pass_count_label = QLabel("0")
        pass_card = self._create_mini_card("● 合格", self._pass_count_label, COLOR_OK)
        stats_layout.addWidget(pass_card, 0, 0)

        self._fail_count_label = QLabel("0")
        fail_card = self._create_mini_card("● 不合格", self._fail_count_label, COLOR_ERR)
        stats_layout.addWidget(fail_card, 0, 1)

        self._total_count_label = QLabel("0")
        total_card = self._create_mini_card("● 总数", self._total_count_label, COLOR_ACCENT)
        stats_layout.addWidget(total_card, 1, 0)

        self._rate_label = QLabel("0%")
        rate_card = self._create_mini_card("● 合格率", self._rate_label, COLOR_WARN)
        stats_layout.addWidget(rate_card, 1, 1)

        layout.addWidget(stats_frame)

        # ── 3. 检测结果表格 ──
        table_frame = QFrame()
        table_frame.setObjectName("card")
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(6, 6, 6, 6)
        table_layout.setSpacing(2)
        table_title = QLabel("检测结果")
        table_title.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px; font-weight: 600;")
        table_layout.addWidget(table_title)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["#", "类别", "置信度", "状态"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setAlternatingRowColors(False)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setMinimumHeight(120)
        table_layout.addWidget(self.result_table)
        layout.addWidget(table_frame)

        # ── 4. 缺陷分布 ──
        defect_frame = QFrame()
        defect_frame.setObjectName("card")
        defect_layout = QVBoxLayout(defect_frame)
        defect_layout.setContentsMargins(6, 6, 6, 6)
        defect_layout.setSpacing(2)
        defect_title = QLabel("缺陷分布")
        defect_title.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px; font-weight: 600;")
        defect_layout.addWidget(defect_title)

        self.defect_bars = {}
        for class_name, class_name_cn in zip(CLASS_NAMES, CLASS_NAMES_CN):
            bar_widget = self._create_defect_bar(class_name_cn)
            defect_layout.addWidget(bar_widget)
            self.defect_bars[class_name] = bar_widget
        layout.addWidget(defect_frame)

        # ── 5. 拍照记录（摄像头模式可见） ──
        self.snapshot_frame = QFrame()
        self.snapshot_frame.setObjectName("card")
        snap_layout = QVBoxLayout(self.snapshot_frame)
        snap_layout.setContentsMargins(6, 6, 6, 6)
        snap_layout.setSpacing(2)
        snap_title = QLabel("拍照记录")
        snap_title.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px; font-weight: 600;")
        snap_layout.addWidget(snap_title)

        self.snapshot_table = QTableWidget()
        self.snapshot_table.setColumnCount(3)
        self.snapshot_table.setHorizontalHeaderLabels(["时间", "判定", "缺陷"])
        self.snapshot_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.snapshot_table.verticalHeader().setVisible(False)
        self.snapshot_table.setAlternatingRowColors(False)
        self.snapshot_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.snapshot_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.snapshot_table.setMinimumHeight(80)
        snap_layout.addWidget(self.snapshot_table)
        layout.addWidget(self.snapshot_frame)

        # 拍照按钮（摄像头模式下显示在面板底部）
        self.right_snapshot_btn = QPushButton("  拍照统计")
        self.right_snapshot_btn.setObjectName("snapshotBtn")
        self.right_snapshot_btn.clicked.connect(self._take_snapshot)
        layout.addWidget(self.right_snapshot_btn)

        # ── 6. 导出 / 设置 ──
        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)
        export_btn2 = QPushButton("导出报告")
        export_btn2.clicked.connect(self._export_report)
        actions_row.addWidget(export_btn2)
        import_model_btn2 = QPushButton("导入模型")
        import_model_btn2.clicked.connect(self._import_model)
        actions_row.addWidget(import_model_btn2)
        layout.addLayout(actions_row)

        # 检测数量标签
        self.detection_count_badge = QLabel("0 项检测")
        self.detection_count_badge.setAlignment(Qt.AlignCenter)
        self.detection_count_badge.setStyleSheet(f"""
            background-color: {COLOR_ACCENT}; color: white;
            padding: 4px; border-radius: 8px;
            font-size: 11px; font-weight: 600;
        """)
        layout.addWidget(self.detection_count_badge)

        layout.addStretch()

        scroll.setWidget(panel)
        outer_layout.addWidget(scroll)

        self.right_image_widgets = [open_btn2, open_dir_btn2, self.start_btn2, self.clear_btn2]
        self.right_cam_widgets = [self.right_camera_combo, self.right_camera_btn, self.right_video_btn, self.right_stop_btn]
        self.right_panel_widgets = {
            "snapshot_frame": self.snapshot_frame,
            "snapshot_btn": self.right_snapshot_btn,
        }

        # 初始状态：图片模式
        self._sync_right_mode_ui(0)

        return outer

    def _create_mini_card(self, title, value_label, color):
        """创建迷你统计卡片"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_BG_ALT};
                border: 1px solid {COLOR_BORDER};
                border-left: 2px solid {color};
                border-radius: 4px;
            }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px;")
        layout.addWidget(title_lbl)
        layout.addStretch()
        value_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: 700;")
        layout.addWidget(value_label)

        return card

    def _create_stat_inline(self, title, value_label, color):
        """创建行内统计（紧凑横向）"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 彩色圆点
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {color}; font-size: 8px;")
        layout.addWidget(dot)

        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 11px;")
        layout.addWidget(title_label)

        # 数值
        value_label.setStyleSheet(f"""
            color: {color};
            font-size: 16px;
            font-weight: 700;
        """)
        layout.addWidget(value_label)

        return widget

    def _create_stat_card(self, title, value_label, color, card_type):
        """创建统计卡片（紧凑版）"""
        card = QFrame()
        card.setObjectName("statCard")
        card.setProperty("cardType", card_type)
        card.setStyleSheet(f"""
            QFrame#statCard {{
                background-color: {COLOR_BG_ALT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 8px;
                border-left: 3px solid {color};
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 4, 6, 4)

        # 图标 + 标题
        header = QHBoxLayout()
        icon_label = QLabel(title.split()[0])
        icon_label.setStyleSheet(f"font-size: 13px;")
        header.addWidget(icon_label)

        title_label = QLabel(title.split()[-1] if len(title.split()) > 1 else title)
        title_label.setStyleSheet(f"""
            color: {COLOR_TEXT_DIM};
            font-size: 11px;
            font-weight: 500;
        """)
        header.addWidget(title_label)
        header.addStretch()
        layout.addLayout(header)

        # 数值
        value_label.setStyleSheet(f"""
            font-size: 22px;
            font-weight: 700;
            color: {COLOR_TEXT};
        """)
        layout.addWidget(value_label)

        return card

    def _create_defect_bar(self, class_name_cn):
        """创建缺陷分布条（紧凑版）"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        # 类别名
        name_label = QLabel(class_name_cn)
        name_label.setFixedWidth(40)
        name_label.setStyleSheet(f"""
            color: {COLOR_TEXT};
            font-size: 11px;
        """)
        layout.addWidget(name_label)

        # 进度条
        progress = QProgressBar()
        progress.setFixedHeight(6)
        progress.setTextVisible(False)
        progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLOR_SURFACE};
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLOR_ACCENT}, stop:1 {COLOR_ACCENT});
                border-radius: 3px;
            }}
        """)
        layout.addWidget(progress)

        # 数量
        count_label = QLabel("0")
        count_label.setFixedWidth(24)
        count_label.setStyleSheet(f"""
            color: {COLOR_TEXT};
            font-weight: 600;
            font-size: 11px;
        """)
        count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(count_label)

        widget.progress = progress
        widget.count_label = count_label

        return widget

    def _init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        open_action = QAction("打开图片(&O)", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)

        open_dir_action = QAction("打开文件夹(&D)", self)
        open_dir_action.setShortcut("Ctrl+Shift+O")
        open_dir_action.triggered.connect(self._open_directory)
        file_menu.addAction(open_dir_action)

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

        # 视频菜单
        video_menu = menubar.addMenu("视频(&V)")

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

        self.snapshot_menu_action = QAction("拍照(&P)", self)
        self.snapshot_menu_action.setShortcut("Space")
        self.snapshot_menu_action.triggered.connect(self._take_snapshot)
        self.snapshot_menu_action.setEnabled(False)
        video_menu.addAction(self.snapshot_menu_action)

        # 设置菜单
        settings_menu = menubar.addMenu("设置(&S)")

        settings_action = QAction("检测参数(&P)", self)
        settings_action.triggered.connect(self._open_settings)
        settings_menu.addAction(settings_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")

        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_toolbar(self):
        """初始化工具栏"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # ── 检测模式选择 ──
        mode_label = QLabel("  检测模式 ")
        mode_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 12px;")
        toolbar.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["图片检测", "摄像头检测", "视频检测"])
        self.mode_combo.setFixedWidth(120)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.mode_combo)

        toolbar.addSeparator()

        # ── 图片模式控件 ──
        self.image_widgets = []
        open_btn = QPushButton("  打开图片")
        open_btn.clicked.connect(self._open_image)
        toolbar.addWidget(open_btn)
        self.image_widgets.append(open_btn)

        open_dir_btn = QPushButton("  打开文件夹")
        open_dir_btn.clicked.connect(self._open_directory)
        toolbar.addWidget(open_dir_btn)
        self.image_widgets.append(open_dir_btn)

        self.start_btn = QPushButton("  开始检测")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start_detection)
        self.start_btn.setEnabled(False)
        self.start_btn2.setEnabled(False)
        toolbar.addWidget(self.start_btn)
        self.image_widgets.append(self.start_btn)

        self.clear_btn = QPushButton("  清除")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self._clear_results)
        toolbar.addWidget(self.clear_btn)
        self.image_widgets.append(self.clear_btn)

        # ── 摄像头/视频模式控件 ──
        self.video_widgets = []

        # 摄像头设备选择
        cam_label = QLabel("  摄像头 ")
        cam_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 12px;")
        toolbar.addWidget(cam_label)
        self.video_widgets.append(cam_label)

        self.camera_combo = QComboBox()
        self.camera_combo.setFixedWidth(180)
        self._scan_cameras()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        toolbar.addWidget(self.camera_combo)
        self.video_widgets.append(self.camera_combo)

        # 启动摄像头按钮
        self.camera_btn = QPushButton("  启动摄像头")
        self.camera_btn.setObjectName("cameraBtn")
        self.camera_btn.clicked.connect(self._start_camera)
        toolbar.addWidget(self.camera_btn)
        self.video_widgets.append(self.camera_btn)

        # 视频文件按钮
        self.video_file_btn = QPushButton("  视频文件")
        self.video_file_btn.setObjectName("videoBtn")
        self.video_file_btn.clicked.connect(self._open_video_file)
        toolbar.addWidget(self.video_file_btn)
        self.video_widgets.append(self.video_file_btn)

        # 停止视频
        self.stop_video_btn = QPushButton("  停止")
        self.stop_video_btn.setObjectName("stopVideoBtn")
        self.stop_video_btn.clicked.connect(self._stop_video)
        self.stop_video_btn.hide()
        toolbar.addWidget(self.stop_video_btn)
        self.video_widgets.append(self.stop_video_btn)

        # 拍照
        self.snapshot_btn = QPushButton("  拍照")
        self.snapshot_btn.setObjectName("snapshotBtn")
        self.snapshot_btn.clicked.connect(self._take_snapshot)
        self.snapshot_btn.hide()
        toolbar.addWidget(self.snapshot_btn)
        self.video_widgets.append(self.snapshot_btn)

        # FPS
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fpsLabel")
        self.fps_label.hide()
        toolbar.addWidget(self.fps_label)
        self.video_widgets.append(self.fps_label)

        toolbar.addSeparator()

        # ── 通用控件 ──
        # 导出报告
        export_btn = QPushButton("  导出报告")
        export_btn.clicked.connect(self._export_report)
        toolbar.addWidget(export_btn)

        # 导入模型
        import_model_btn = QPushButton("  导入模型")
        import_model_btn.clicked.connect(self._import_model)
        toolbar.addWidget(import_model_btn)

        # 置信度阈值
        conf_label = QLabel("  置信度 ")
        conf_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 12px;")
        toolbar.addWidget(conf_label)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.settings["conf_threshold"])
        self.conf_spin.setFixedWidth(70)
        self.conf_spin.valueChanged.connect(self._on_conf_changed)
        toolbar.addWidget(self.conf_spin)

        toolbar.addSeparator()

        # 180秒计时
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

        # 弹簧
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # 初始状态：图片模式
        self._on_mode_changed(0)

    def _init_statusbar(self):
        """初始化状态栏"""
        status_bar = self.statusBar()

        # 左侧状态
        self.status_label = QLabel("就绪")
        status_bar.addWidget(self.status_label)

        # 右侧永久信息
        self.model_info_label = QLabel("模型: 加载中...")
        status_bar.addPermanentWidget(self.model_info_label)

        self.device_info_label = QLabel("设备: --")
        status_bar.addPermanentWidget(self.device_info_label)

        version_info = QLabel(f"v{APP_VERSION}")
        status_bar.addPermanentWidget(version_info)

    def _connect_signals(self):
        """连接信号"""
        pass

    def _scan_cameras(self):
        """扫描可用摄像头设备"""
        self.camera_combo.clear()
        available = []
        for i in range(5):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                try:
                    cap = cv2.VideoCapture(i, backend)
                except Exception:
                    continue
                if cap.isOpened():
                    name = f"摄像头 {i}"
                    try:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if w > 0 and h > 0:
                            name += f" ({w}x{h})"
                    except Exception:
                        pass
                    available.append((i, name))
                    cap.release()
                    break
                cap.release()
        if available:
            for idx, name in available:
                self.camera_combo.addItem(name, idx)
        else:
            self.camera_combo.addItem("未检测到摄像头", -1)

    def _on_camera_changed(self, index):
        """摄像头设备切换（工具栏 + 右侧面板同步）"""
        if index >= 0:
            self.right_camera_combo.setCurrentIndex(index)
        if self.is_video_streaming and self.mode_combo.currentIndex() == 1:
            self._stop_video()

    def _on_mode_changed(self, index):
        """检测模式切换：0=图片, 1=摄像头, 2=视频"""
        if self.is_video_streaming:
            self._stop_video()

        is_image = (index == 0)
        is_camera = (index == 1)

        # 工具栏控件
        for w in self.image_widgets:
            w.setVisible(is_image)
        for w in self.video_widgets:
            w.setVisible(not is_image)
        self.camera_combo.setVisible(is_camera)
        self.camera_btn.setVisible(is_camera)
        self.video_file_btn.setVisible(index == 2)
        self.stop_video_btn.setVisible(False)
        self.snapshot_btn.setVisible(False)
        self.fps_label.setVisible(False)
        self.start_btn.setEnabled(is_image and bool(self.current_image_path))
        self.start_btn2.setEnabled(is_image and bool(self.current_image_path))
        self.camera_menu_action.setEnabled(not is_image)
        self.video_file_menu_action.setEnabled(not is_image)

        # 同步右侧面板
        self.right_mode_combo.blockSignals(True)
        self.right_mode_combo.setCurrentIndex(index)
        self.right_mode_combo.blockSignals(False)
        self._sync_right_mode_ui(index)

    def _scan_right_cameras(self):
        """扫描摄像头填充右侧面板下拉框"""
        self.right_camera_combo.clear()
        available = []
        for i in range(5):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                try:
                    cap = cv2.VideoCapture(i, backend)
                except Exception:
                    continue
                if cap.isOpened():
                    name = f"摄像头 {i}"
                    try:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if w > 0 and h > 0:
                            name += f" ({w}x{h})"
                    except Exception:
                        pass
                    available.append((i, name))
                    cap.release()
                    break
                cap.release()
        if available:
            for idx, name in available:
                self.right_camera_combo.addItem(name, idx)
        else:
            self.right_camera_combo.addItem("未检测到摄像头", -1)

    def _on_right_mode_changed(self, index):
        """右侧面板模式切换 → 同步工具栏"""
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(index)
        self.mode_combo.blockSignals(False)
        self._on_mode_changed(index)

    def _sync_right_mode_ui(self, index):
        """同步右侧面板 UI 到指定模式"""
        is_image = (index == 0)
        is_camera = (index == 1)
        is_video = (index == 2)

        for w in self.right_image_widgets:
            w.setVisible(is_image)
        for w in self.right_cam_widgets:
            w.setVisible(not is_image)
        self.right_camera_combo.setVisible(is_camera)
        self.right_camera_btn.setVisible(is_camera)
        self.right_video_btn.setVisible(is_video)
        self.right_stop_btn.setVisible(False)
        self.right_panel_widgets["snapshot_frame"].setVisible(is_camera)
        self.right_panel_widgets["snapshot_btn"].setVisible(is_camera)

    def _init_detector(self):
        """初始化检测器"""
        try:
            from inference import ScrewDefectDetector, find_best_model

            # 查找模型
            model_path = self.settings["model_path"]
            if not model_path or not Path(model_path).exists():
                # 打包后的路径
                if getattr(sys, 'frozen', False):
                    base_path = Path(sys._MEIPASS)
                    packaged_model = base_path / "best.pt"
                    if packaged_model.exists():
                        model_path = str(packaged_model)
                    else:
                        model_path = "yolov8n.pt"
                else:
                    best = find_best_model()
                    if best:
                        model_path = str(best)
                    else:
                        model_path = "yolov8n.pt"

            self.status_label.setText(f"正在加载模型: {Path(model_path).name}...")

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

            self.status_label.setText("模型加载完成")

        except Exception as e:
            self.status_label.setText(f"模型加载失败: {str(e)}")
            QMessageBox.warning(
                self, "警告",
                f"模型加载失败:\n{str(e)}\n\n请检查模型文件是否存在。"
            )

    def _on_conf_changed(self, value):
        """置信度阈值变更"""
        self.settings["conf_threshold"] = value
        if self.detector:
            self.detector.conf_threshold = value

    def _import_model(self):
        """导入自定义模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "模型文件 (*.pt *.onnx);;所有文件 (*)"
        )
        if not file_path:
            return

        if not Path(file_path).exists():
            QMessageBox.warning(self, "错误", "模型文件不存在。")
            return

        try:
            from inference import ScrewDefectDetector

            self.status_label.setText(f"正在加载模型: {Path(file_path).name}...")

            # 创建新的检测器
            new_detector = ScrewDefectDetector(
                model_path=file_path,
                conf_threshold=self.settings["conf_threshold"],
                iou_threshold=self.settings["iou_threshold"],
                img_size=self.settings["img_size"],
            )

            # 替换旧检测器
            self.detector = new_detector
            self.settings["model_path"] = file_path

            # 更新界面信息
            device_info = self.detector.get_device_info()
            self.model_info_label.setText(f"模型: {Path(file_path).name}")
            self.device_info_label.setText(f"设备: {device_info['device'].upper()}")
            self.status_label.setText(f"模型加载成功: {Path(file_path).name}")

        except Exception as e:
            QMessageBox.warning(
                self, "加载失败",
                f"模型加载失败:\n{str(e)}"
            )
            self.status_label.setText("模型加载失败")

    # ── 以下为功能方法，与原版相同 ──

    def _open_image(self):
        """打开单张图片"""
        if self.is_video_streaming:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
        )

        if file_path:
            self.current_image_path = file_path
            self._display_image(file_path)
            self.start_btn.setEnabled(True)
            self.start_btn2.setEnabled(True)
            self.image_info_label.setText(f"文件: {Path(file_path).name}")
            self.status_label.setText(f"已加载: {Path(file_path).name}")

    def _open_directory(self):
        """打开文件夹"""
        if self.is_video_streaming:
            return

        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if dir_path:
            self.current_image_path = dir_path
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = set()
            for ext in extensions:
                image_files.update(Path(dir_path).glob(ext))
                image_files.update(Path(dir_path).glob(ext.upper()))

            if image_files:
                self.start_btn.setEnabled(True)
                self.start_btn2.setEnabled(True)
                sorted_files = sorted(image_files)
                first_image = str(sorted_files[0])
                self._display_image(first_image)
                self.image_info_label.setText(f"文件夹: {Path(dir_path).name} ({len(sorted_files)} 张图片)")
                self.status_label.setText(f"已加载 {len(sorted_files)} 张图片")
            else:
                QMessageBox.information(self, "提示", "该文件夹中没有找到图片文件")

    def _fit_image(self, pixmap):
        """将图片缩放适配显示区（完整显示，不裁剪）"""
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_display.setPixmap(scaled)

    def _display_image(self, image_path):
        """显示图片"""
        pixmap = QPixmap(image_path)
        self._fit_image(pixmap)

    def _start_detection(self):
        """开始检测"""
        if not self.current_image_path or not self.detector:
            return

        if self.is_video_streaming:
            return

        self.is_detecting = True
        self.start_btn.setEnabled(False)
        self.start_btn2.setEnabled(False)
        self.status_label.setText("正在检测...")

        path = Path(self.current_image_path)
        if path.is_dir():
            # 批量检测
            self._start_batch_detection(str(path))
        else:
            # 单张检测
            self._detect_single_image(str(path))

    def _detect_single_image(self, image_path):
        """检测单张图片"""
        try:
            result = self.detector.detect_single(image_path)
            annotated = self.detector.visualize_result(image_path, result)

            # 显示结果
            h, w, ch = annotated.shape
            bytes_per_line = ch * w
            rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(q_img)
            self._fit_image(pixmap)
            self.all_results.append(result)
            self._update_result_table(result)
            self._update_stats()
            self._update_defect_distribution()

            self.image_info_label.setText(
                f"推理时间: {result.inference_time_ms:.1f}ms | {result.overall_verdict}"
            )
            self.status_label.setText(f"检测完成: {result.overall_verdict}")

        except Exception as e:
            self.status_label.setText(f"检测失败: {str(e)}")

        finally:
            self.is_detecting = False
            self.start_btn.setEnabled(True)
            self.start_btn2.setEnabled(True)

    def _start_batch_detection(self, dir_path):
        """批量检测"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = set()
        for ext in extensions:
            image_files.update(Path(dir_path).glob(ext))
            image_files.update(Path(dir_path).glob(ext.upper()))

        image_paths = [str(p) for p in image_files]

        if not image_paths:
            self.status_label.setText("没有找到图片文件")
            self.is_detecting = False
            self.start_btn.setEnabled(True)
            self.start_btn2.setEnabled(True)
            return

        # 创建后台线程
        self.detection_thread = DetectionThread(self.detector, image_paths, self)
        self.detection_thread.progress.connect(self._on_batch_progress)
        self.detection_thread.result_ready.connect(self._on_batch_result)
        self.detection_thread.finished_all.connect(self._on_batch_finished)
        self.detection_thread.error_occurred.connect(self._on_batch_error)

        self.progress_bar.setMaximum(len(image_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.detection_thread.start()

    def _on_batch_progress(self, current, total):
        """批量检测进度"""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"正在检测: {current}/{total}")

    def _on_batch_result(self, image_path, result):
        """批量检测单个结果"""
        self.all_results.append(result)
        self._update_stats()
        self._update_defect_distribution()

    def _on_batch_finished(self, results):
        """批量检测完成"""
        self.progress_bar.hide()
        self.is_detecting = False
        self.start_btn.setEnabled(True)
        self.start_btn2.setEnabled(True)

        # 显示最后一张结果
        if results:
            last_result = results[-1]
            annotated = self.detector.visualize_result(last_result.image_path, last_result)
            h, w, ch = annotated.shape
            bytes_per_line = ch * w
            rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(q_img)
            self._fit_image(pixmap)

        total = len(results)
        pass_count = sum(1 for r in results if not r.has_defect)
        self.status_label.setText(f"批量检测完成: {total} 张, 合格 {pass_count} 张")

    def _on_batch_error(self, error_msg):
        """批量检测错误"""
        self.status_label.setText(f"检测错误: {error_msg}")

    def _update_result_table(self, result):
        """更新结果表格"""
        self.result_table.setRowCount(len(result.detections))

        for i, det in enumerate(result.detections):
            for col, text in [
                (0, str(i + 1)),
                (1, det.class_name_cn),
                (2, f"{det.confidence:.2%}"),
                (3, "缺陷" if det.is_defect else "合格"),
            ]:
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                if col in (1, 3) and det.is_defect:
                    item.setForeground(QColor(COLOR_ERR))
                elif col in (1, 3):
                    item.setForeground(QColor(COLOR_OK))
                self.result_table.setItem(i, col, item)

        self.detection_count_badge.setText(f"{len(result.detections)} 项检测")

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

    def _update_defect_distribution(self):
        """更新缺陷分布"""
        # 统计各缺陷类型数量
        defect_counts = {name: 0 for name in CLASS_NAMES}
        for result in self.all_results:
            for det in result.detections:
                if det.class_name in defect_counts:
                    defect_counts[det.class_name] += 1

        # 找到最大值用于归一化
        max_count = max(defect_counts.values()) if defect_counts.values() else 1

        # 更新进度条
        for class_name, count in defect_counts.items():
            if class_name in self.defect_bars:
                widget = self.defect_bars[class_name]
                percent = int((count / max_count) * 100) if max_count > 0 else 0
                widget.progress.setValue(percent)
                widget.count_label.setText(str(count))

    def _clear_results(self):
        """清除结果"""
        self.all_results = []
        self.current_result = None
        self.current_image_path = ""

        self.result_table.setRowCount(0)
        self.detection_count_badge.setText("0 项检测")

        self._pass_count_label.setText("0")
        self._fail_count_label.setText("0")
        self._total_count_label.setText("0")
        self._rate_label.setText("0%")

        for class_name in self.defect_bars:
            widget = self.defect_bars[class_name]
            widget.progress.setValue(0)
            widget.count_label.setText("0")

        self.snapshot_table.setRowCount(0)

        self.image_display.clear()
        self.image_display.setText("拖拽图片到此处\n或通过工具栏上传")
        self.image_info_label.setText("")

        self.start_btn.setEnabled(False)
        self.start_btn2.setEnabled(False)
        self.start_btn2.setEnabled(False)
        self.status_label.setText("已清除")

    def _start_camera(self):
        """启动摄像头"""
        if self.is_video_streaming:
            return
        if not self.detector:
            QMessageBox.warning(self, "警告", "检测器未初始化")
            return
        # 优先右侧面板选择的摄像头，其次工具栏
        cam_idx = self.right_camera_combo.currentData()
        if cam_idx is None or cam_idx < 0:
            cam_idx = self.camera_combo.currentData()
        if cam_idx is None or cam_idx < 0:
            QMessageBox.warning(self, "警告", "未检测到可用摄像头")
            return
        self._start_video_stream(cam_idx)

    def _open_video_file(self):
        """打开视频文件"""
        if self.is_video_streaming:
            return
        if not self.detector:
            QMessageBox.warning(self, "警告", "检测器未初始化")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*)"
        )
        if file_path:
            self._start_video_stream(file_path)

    def _start_video_stream(self, source):
        """启动视频流"""
        if self.is_video_streaming:
            return

        self.all_results = []
        self._update_stats()
        self.result_table.setRowCount(0)

        self.is_video_streaming = True
        self._set_video_ui_state(True)

        if isinstance(source, int):
            source_name = f"摄像头 (camera {source})"
        else:
            source_name = Path(source).name

        self.image_info_label.setText(f"视频流: {source_name}")
        self.status_label.setText(f"正在打开视频流: {source_name}...")

        self.video_thread = VideoStreamThread(self.detector, source, self)
        self.video_thread.frame_ready.connect(self._on_video_frame)
        self.video_thread.fps_update.connect(self._on_fps_update)
        self.video_thread.error_occurred.connect(self._on_video_error)
        self.video_thread.stream_finished.connect(self._on_video_stream_finished)
        self.video_thread.start()

    def _on_video_frame(self, pixmap, result):
        """接收视频帧"""
        self._fit_image(pixmap)
        self.current_video_frame = pixmap
        self.current_video_result = result

        if result:
            self.current_result = result
            self._update_result_table(result)

    def _on_fps_update(self, fps):
        """更新 FPS"""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_video_error(self, error_msg):
        """视频错误"""
        QMessageBox.critical(self, "视频流错误", error_msg)
        self._stop_video()

    def _on_video_stream_finished(self):
        """视频流结束"""
        self._stop_video()
        self.status_label.setText("视频播放完毕")

    def _take_snapshot(self):
        """拍照统计 — 记录到拍照面板"""
        if not self.is_video_streaming:
            return

        if not hasattr(self, 'current_video_result') or not self.current_video_result:
            QMessageBox.information(self, "提示", "当前帧无检测结果")
            return

        result = self.current_video_result
        self.all_results.append(result)
        self._update_stats()
        self._update_defect_distribution()

        # 添加到拍照记录表
        row = self.snapshot_table.rowCount()
        self.snapshot_table.insertRow(0)
        now = datetime.now().strftime("%H:%M:%S")
        verdict = "合格" if not result.has_defect else "缺陷"
        defects = ", ".join(set(d.class_name_cn for d in result.detections)) or "-"

        for col, text in enumerate([now, verdict, defects]):
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            if verdict == "缺陷":
                item.setForeground(QColor(COLOR_ERR))
            else:
                item.setForeground(QColor(COLOR_OK))
            self.snapshot_table.setItem(0, col, item)

        total = len(self.all_results)
        status = "缺陷" if result.has_defect else "合格"
        self.status_label.setText(f"拍照: {status} | 总计 {total} 张")

    def _stop_video(self):
        """停止视频流"""
        if not self.is_video_streaming:
            return

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait(3000)
            self.video_thread = None

        self.is_video_streaming = False
        self._set_video_ui_state(False)

        self.fps_label.setText("FPS: --")
        self.image_info_label.setText("视频流已停止")
        self.status_label.setText("视频流检测已停止")

    def _set_video_ui_state(self, streaming):
        """设置视频 UI 状态 — 工具栏 + 右侧面板同步"""
        mode_idx = self.mode_combo.currentIndex()
        is_camera = (mode_idx == 1)
        is_video = (mode_idx == 2)

        # 工具栏
        self.camera_combo.setVisible(not streaming and is_camera)
        self.camera_btn.setVisible(not streaming and is_camera)
        self.video_file_btn.setVisible(not streaming and is_video)
        self.stop_video_btn.setVisible(streaming)
        self.snapshot_btn.setVisible(streaming)
        self.fps_label.setVisible(streaming)
        self.mode_combo.setEnabled(not streaming)

        # 右侧面板
        self.right_mode_combo.setEnabled(not streaming)
        self.right_camera_combo.setVisible(not streaming and is_camera)
        self.right_camera_btn.setVisible(not streaming and is_camera)
        self.right_video_btn.setVisible(not streaming and is_video)
        self.right_stop_btn.setVisible(streaming)
        self.right_panel_widgets["snapshot_frame"].setVisible(streaming and is_camera)
        self.right_panel_widgets["snapshot_btn"].setVisible(streaming and is_camera)

        # 菜单
        self.camera_menu_action.setEnabled(not streaming)
        self.video_file_menu_action.setEnabled(not streaming)
        self.stop_video_menu_action.setEnabled(streaming)
        self.snapshot_menu_action.setEnabled(streaming)

        self.start_btn.setEnabled(not streaming and bool(self.current_image_path))
        self.start_btn2.setEnabled(not streaming and bool(self.current_image_path))
        self.start_btn2.setEnabled(not streaming and bool(self.current_image_path))

    def _toggle_timer_mode(self):
        """切换计时器模式"""
        if self.timer_mode:
            self.timer_mode = False
            self.countdown_timer.stop()
            self.timer_label.hide()
            self.timer_btn.setText("  ⏱ 180秒计时")
            self.status_label.setText("计时模式已关闭")
        else:
            self.timer_mode = True
            self.timer_remaining = self.timer_seconds
            self._update_timer_display()
            self.timer_label.show()
            self.countdown_timer.start(1000)

    def _update_timer(self):
        """更新计时器"""
        self.timer_remaining -= 1
        self._update_timer_display()

        if self.timer_remaining <= 0:
            self.countdown_timer.stop()
            self._timer_finished()

    def _update_timer_display(self):
        """更新计时器显示"""
        self.timer_label.setText(f"⏱ {self.timer_remaining}s")

        if self.timer_remaining <= 30:
            self.timer_label.setStyleSheet(f"""
                font-size: 16px;
                font-weight: 700;
                color: {COLOR_ERR};
                background-color: {COLOR_ERR_BG};
                padding: 6px 12px;
                border-radius: 6px;
            """)
        else:
            self.timer_label.setStyleSheet(f"""
                font-size: 16px;
                font-weight: 600;
                color: {COLOR_WARN};
                background-color: {COLOR_WARN_BG};
                padding: 6px 12px;
                border-radius: 6px;
            """)

    def _timer_finished(self):
        """计时器结束"""
        self.timer_mode = False
        self.timer_label.hide()

        total = len(self.all_results)
        pass_count = sum(1 for r in self.all_results if not r.has_defect)
        fail_count = total - pass_count
        rate = (pass_count / total * 100) if total > 0 else 0

        QMessageBox.information(
            self, "计时结束",
            f"180 秒计时结束！\n\n"
            f"检测总数: {total}\n"
            f"合格数量: {pass_count}\n"
            f"不合格数量: {fail_count}\n"
            f"合格率: {rate:.1f}%"
        )

    def _export_report(self):
        """导出报告"""
        if not self.all_results:
            QMessageBox.information(self, "提示", "没有检测结果可导出")
            return

        file_path, file_type = QFileDialog.getSaveFileName(
            self, "导出报告", "",
            "文本文件 (*.txt);;CSV 文件 (*.csv)"
        )

        if not file_path:
            return

        try:
            if file_path.endswith('.csv'):
                self._export_csv(file_path)
            else:
                self._export_txt(file_path)

            self.status_label.setText(f"报告已导出: {Path(file_path).name}")
            QMessageBox.information(self, "成功", f"报告已导出到:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    def _export_txt(self, file_path):
        """导出 TXT 报告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("钢材缺陷检测报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            total = len(self.all_results)
            pass_count = sum(1 for r in self.all_results if not r.has_defect)
            fail_count = total - pass_count
            rate = (pass_count / total * 100) if total > 0 else 0

            f.write(f"检测总数: {total}\n")
            f.write(f"合格数量: {pass_count}\n")
            f.write(f"不合格数量: {fail_count}\n")
            f.write(f"合格率: {rate:.1f}%\n\n")

            f.write("详细结果:\n")
            f.write("-" * 60 + "\n")

            for i, result in enumerate(self.all_results, 1):
                defect_types = ', '.join(set(d.class_name_cn for d in result.detections))
                f.write(f"{i}. {Path(result.image_path).name} | "
                        f"{'合格' if not result.has_defect else '不合格'} | "
                        f"缺陷: {defect_types or '无'} | "
                        f"{result.inference_time_ms:.1f}ms\n")

    def _export_csv(self, file_path):
        """导出 CSV 报告"""
        import csv
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '图片', '判定', '缺陷数量', '缺陷类型', '推理时间(ms)'])

            for i, result in enumerate(self.all_results, 1):
                defect_types = ', '.join(set(d.class_name_cn for d in result.detections))
                writer.writerow([
                    i,
                    Path(result.image_path).name,
                    '合格' if not result.has_defect else '不合格',
                    len(result.detections),
                    defect_types or '-',
                    f"{result.inference_time_ms:.1f}"
                ])

    def _open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.settings = dialog.get_settings()

    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, "关于",
            f"<h2>{APP_TITLE_CN}</h2>"
            f"<p>版本: {APP_VERSION}</p>"
            f"<p>基于 YOLOv8 的钢材表面缺陷检测系统</p>"
            f"<p>支持 6 种缺陷类型检测</p>"
            f"<hr>"
            f"<p>功能特性:</p>"
            f"<ul>"
            f"<li>图片/文件夹批量检测</li>"
            f"<li>摄像头实时检测</li>"
            f"<li>视频文件检测</li>"
            f"<li>180秒计时演示模式</li>"
            f"<li>检测报告导出</li>"
            f"</ul>"
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """拖拽放下事件"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if Path(file_path).is_file():
                self.current_image_path = file_path
                self._display_image(file_path)
                self.start_btn.setEnabled(True)
                self.start_btn2.setEnabled(True)
                self.image_info_label.setText(f"文件: {Path(file_path).name}")
            elif Path(file_path).is_dir():
                self.current_image_path = file_path
                self._open_directory()


# ============================================================
# 设置对话框
# ============================================================

class SettingsDialog(QDialog):
    """设置对话框"""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings.copy()
        self.setWindowTitle("检测参数设置")
        self.setMinimumWidth(400)

        layout = QFormLayout(self)

        # 置信度阈值
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(settings["conf_threshold"])
        layout.addRow("置信度阈值:", self.conf_spin)

        # IoU 阈值
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(settings["iou_threshold"])
        layout.addRow("IoU 阈值:", self.iou_spin)

        # 图片尺寸
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(320, 1280)
        self.img_size_spin.setSingleStep(32)
        self.img_size_spin.setValue(settings["img_size"])
        layout.addRow("图片尺寸:", self.img_size_spin)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_settings(self):
        """获取设置"""
        self.settings["conf_threshold"] = self.conf_spin.value()
        self.settings["iou_threshold"] = self.iou_spin.value()
        self.settings["img_size"] = self.img_size_spin.value()
        return self.settings


# ============================================================
# 工作线程
# ============================================================

class VideoStreamThread(QThread):
    """视频流检测线程"""

    frame_ready = pyqtSignal(object, object)
    fps_update = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    stream_finished = pyqtSignal()

    def __init__(self, detector, source=0, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.source = source
        self._is_running = True

    def run(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.source, cv2.CAP_ANY)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self.error_occurred.emit(f"无法打开视频源: {self.source}")
            return

        fps_counter = 0
        fps_start_time = time.perf_counter()

        while self._is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    break
                continue

            try:
                t_start = time.perf_counter()
                results = self.detector.model.predict(
                    frame,
                    conf=self.detector.conf_threshold,
                    iou=self.detector.iou_threshold,
                    imgsz=self.detector.img_size,
                    device=self.detector.device,
                    half=self.detector.use_fp16,
                    verbose=False,
                )
                t_end = time.perf_counter()
                inference_time_ms = (t_end - t_start) * 1000

                result = self.detector._parse_results(results[0], "<video_frame>", frame.shape)
                result.inference_time_ms = inference_time_ms

                # 绘制检测框
                annotated = frame.copy()
                for det in result.detections:
                    x1, y1, x2, y2 = [int(c) for c in det.bbox]
                    color = (0, 0, 255) if det.is_defect else (0, 200, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    label = f"{det.class_name_cn} {det.confidence:.0%}"
                    annotated = _put_cn_text(
                        annotated, label,
                        pos=(x1 + 2, y1 - 22),
                        font_size=16,
                        color=(255, 255, 255),
                        bg_color=color,
                    )

                # 转换为 QPixmap
                rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(q_img)

                self.frame_ready.emit(pixmap, result)

            except Exception:
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(q_img)
                self.frame_ready.emit(pixmap, None)

            fps_counter += 1
            elapsed = time.perf_counter() - fps_start_time
            if elapsed >= 0.5:
                current_fps = fps_counter / elapsed
                self.fps_update.emit(current_fps)
                fps_counter = 0
                fps_start_time = time.perf_counter()

        cap.release()
        self.stream_finished.emit()

    def stop(self):
        self._is_running = False


class DetectionThread(QThread):
    """批量检测线程"""

    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(object, object)
    finished_all = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector, image_paths, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.image_paths = image_paths
        self._is_running = True

    def run(self):
        results = []
        for i, image_path in enumerate(self.image_paths):
            if not self._is_running:
                break

            try:
                result = self.detector.detect_single(image_path)
                results.append(result)
                self.result_ready.emit(image_path, result)
                self.progress.emit(i + 1, len(self.image_paths))
            except Exception as e:
                self.error_occurred.emit(f"检测失败 {image_path}: {str(e)}")

        self.finished_all.emit(results)

    def stop(self):
        self._is_running = False


# ============================================================
# 启动入口
# ============================================================

def main():
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 应用样式
    app.setStyleSheet(MODERN_STYLE)

    window = SteelDefectApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
