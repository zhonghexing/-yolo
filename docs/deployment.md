# 部署方案文档

## 文档信息

| 项目 | 内容 |
|------|------|
| 文档名称 | 基于YOLOv8的钢材表面缺陷检测系统 - 部署方案 |
| 文档版本 | V1.1 |
| 创建日期 | 2026年5月27日 |
| 最后更新 | 2026年5月28日 |

---

## 1. 部署概述

### 1.1 部署目标

| 目标 | 具体要求 |
|------|----------|
| 功能完整 | 所有检测功能正常运行 |
| 性能达标 | 满足竞赛要求（20样本/180秒） |
| 稳定可靠 | 长时间运行无崩溃 |
| 易于使用 | 用户操作简单直观 |
| 便于分发 | 支持独立打包分发 |

### 1.2 部署方式

| 部署方式 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| 源码运行 | 开发调试 | 灵活，便于修改 | 需要Python环境 |
| 打包分发 | 竞赛演示 | 独立运行，无需环境 | 包体积大 |
| Docker容器 | 服务器部署 | 环境隔离，一致性强 | 需要Docker支持 |
| 边缘部署 | 工业现场 | 实时性好，独立运行 | 需要硬件适配 |

**本项目选择**：源码运行（开发阶段）+ PyInstaller打包分发（竞赛演示）

---

## 2. 桌面应用架构

### 2.1 应用架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       桌面应用架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    PyQt5 GUI层                          │  │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│   │  │ 主窗口  │  │ 检测面板│  │ 结果面板│  │ 设置面板│   │  │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                    信号/槽机制│                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    业务逻辑层                           │  │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│   │  │检测管理 │  │结果处理 │  │报告生成 │  │配置管理 │   │  │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    推理引擎层                           │  │
│   │  ┌─────────────────────────────────────────────────┐   │  │
│   │  │              YOLOv8 Inference                   │   │  │
│   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │  │
│   │  │  │PyTorch  │  │  ONNX   │  │TensorRT │        │   │  │
│   │  │  └─────────┘  └─────────┘  └─────────┘        │   │  │
│   │  └─────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 职责 | 主要文件 |
|------|------|----------|
| GUI层 | 用户交互，界面显示 | main_window.py |
| 检测管理 | 调度检测任务，管理检测流程 | detection_manager.py |
| 结果处理 | 处理检测结果，生成可视化 | result_processor.py |
| 报告生成 | 生成检测报告，导出数据 | report_generator.py |
| 推理引擎 | 模型加载，执行推理 | inference_engine.py |

---

## 3. GUI界面设计

### 3.1 主界面布局

```
┌─────────────────────────────────────────────────────────────────┐
│  螺丝缺陷检测系统 v1.0                              [─][□][×]  │
├─────────────────────────────────────────────────────────────────┤
│  文件(F)  检测(D)  模型(M)  设置(S)  帮助(H)                  │
├─────────────────────────────────────────────────────────────────┤
│  [📂 打开图像] [📁 打开文件夹] [▶ 开始检测] [📊 导出报告]     │
├───────────────────────────────┬─────────────────────────────────┤
│                               │                                 │
│                               │  检测结果                       │
│                               │  ┌─────────────────────────┐   │
│                               │  │ 图像1.jpg    正常件 0.98 │   │
│   图像显示区域                │  │ 图像2.jpg    缺角   0.95 │   │
│                               │  │ 图像3.jpg    划痕   0.92 │   │
│   [原始图像/检测结果]         │  │ ...                     │   │
│                               │  └─────────────────────────┘   │
│                               │                                 │
│                               │  统计信息                       │
│                               │  ┌─────────────────────────┐   │
│                               │  │ 总数: 20  合格: 18      │   │
│                               │  │ 不合格: 2  合格率: 90%  │   │
│                               │  └─────────────────────────┘   │
│                               │                                 │
├───────────────────────────────┴─────────────────────────────────┤
│  状态: 就绪 | 模型: yolov8s | 推理时间: 15ms                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 界面组件

| 组件 | 功能 | 说明 |
|------|------|------|
| 菜单栏 | 系统功能入口 | 文件、检测、模型、设置、帮助 |
| 工具栏 | 快捷操作按钮 | 打开、检测、导出等 |
| 图像区域 | 显示原始/结果图像 | 支持缩放、平移 |
| 结果列表 | 显示检测结果 | 可排序、筛选 |
| 统计面板 | 显示统计信息 | 图表展示 |
| 状态栏 | 显示系统状态 | 模型信息、推理时间 |

### 3.3 GUI代码实现

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主窗口实现
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QStatusBar,
                             QMenuBar, QToolBar, QAction, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon
import cv2
import numpy as np


class DetectionThread(QThread):
    """检测线程"""
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)
    
    def __init__(self, detector, images):
        super().__init__()
        self.detector = detector
        self.images = images
    
    def run(self):
        results = []
        for i, img_path in enumerate(self.images):
            result = self.detector.detect(img_path)
            results.append(result)
            self.progress.emit(int((i + 1) / len(self.images) * 100))
        self.finished.emit(results)


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.current_images = []
        self.detection_results = []
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('螺丝缺陷检测系统 v1.0')
        self.setGeometry(100, 100, 1280, 720)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # 左侧图像区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label, stretch=2)
        
        # 右侧结果区域
        right_layout = QVBoxLayout()
        
        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(['图像', '类别', '置信度', '状态'])
        right_layout.addWidget(self.result_table)
        
        # 统计信息
        self.stats_label = QLabel()
        right_layout.addWidget(self.stats_label)
        
        layout.addLayout(right_layout, stretch=1)
        
        # 创建菜单栏
        self.create_menus()
        
        # 创建工具栏
        self.create_toolbar()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪')
    
    def create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        open_action = QAction('打开图像', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_images)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction('打开文件夹', self)
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 检测菜单
        detect_menu = menubar.addMenu('检测(&D)')
        
        start_action = QAction('开始检测', self)
        start_action.setShortcut('F5')
        start_action.triggered.connect(self.start_detection)
        detect_menu.addAction(start_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        open_btn = QPushButton('📂 打开图像')
        open_btn.clicked.connect(self.open_images)
        toolbar.addWidget(open_btn)
        
        folder_btn = QPushButton('📁 打开文件夹')
        folder_btn.clicked.connect(self.open_folder)
        toolbar.addWidget(folder_btn)
        
        detect_btn = QPushButton('▶ 开始检测')
        detect_btn.clicked.connect(self.start_detection)
        toolbar.addWidget(detect_btn)
        
        export_btn = QPushButton('📊 导出报告')
        export_btn.clicked.connect(self.export_report)
        toolbar.addWidget(export_btn)
    
    def open_images(self):
        """打开图像文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, '选择图像', '',
            '图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)'
        )
        if files:
            self.current_images = files
            self.display_image(files[0])
            self.status_bar.showMessage(f'已加载 {len(files)} 张图像')
    
    def open_folder(self):
        """打开文件夹"""
        folder = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder:
            import glob
            files = glob.glob(f'{folder}/*.jpg') + glob.glob(f'{folder}/*.png')
            self.current_images = files
            if files:
                self.display_image(files[0])
            self.status_bar.showMessage(f'已加载 {len(files)} 张图像')
    
    def display_image(self, image_path):
        """显示图像"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def start_detection(self):
        """开始检测"""
        if not self.current_images:
            QMessageBox.warning(self, '警告', '请先加载图像！')
            return
        
        self.status_bar.showMessage('正在检测...')
        
        # 创建检测线程
        self.detection_thread = DetectionThread(self.detector, self.current_images)
        self.detection_thread.progress.connect(self.update_progress)
        self.detection_thread.finished.connect(self.detection_finished)
        self.detection_thread.start()
    
    def update_progress(self, value):
        """更新进度"""
        self.status_bar.showMessage(f'检测进度: {value}%')
    
    def detection_finished(self, results):
        """检测完成"""
        self.detection_results = results
        self.update_result_table()
        self.update_statistics()
        self.status_bar.showMessage(f'检测完成，共 {len(results)} 张图像')
    
    def update_result_table(self):
        """更新结果表格"""
        self.result_table.setRowCount(len(self.detection_results))
        for i, result in enumerate(self.detection_results):
            self.result_table.setItem(i, 0, QTableWidgetItem(result['filename']))
            self.result_table.setItem(i, 1, QTableWidgetItem(result['class_name']))
            self.result_table.setItem(i, 2, QTableWidgetItem(f"{result['confidence']:.2%}"))
            status = '合格' if result['class_name'] == 'Normal' else '不合格'
            self.result_table.setItem(i, 3, QTableWidgetItem(status))
    
    def update_statistics(self):
        """更新统计信息"""
        total = len(self.detection_results)
        normal = sum(1 for r in self.detection_results if r['class_name'] == 'Normal')
        defect = total - normal
        rate = normal / total * 100 if total > 0 else 0
        
        stats_text = f"""
        统计信息:
        - 总数: {total}
        - 合格: {normal}
        - 不合格: {defect}
        - 合格率: {rate:.1f}%
        """
        self.stats_label.setText(stats_text)
    
    def export_report(self):
        """导出报告"""
        if not self.detection_results:
            QMessageBox.warning(self, '警告', '没有检测结果可导出！')
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, '导出报告', '',
            'CSV文件 (*.csv);;Excel文件 (*.xlsx)'
        )
        if file_path:
            # 实现导出逻辑
            self.status_bar.showMessage(f'报告已导出到: {file_path}')
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, '关于',
            '螺丝缺陷检测系统 v1.0\n\n'
            '基于YOLOv8的工业螺丝缺陷检测系统\n'
            '适用于竞赛演示和工业质检'
        )


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
```

---

## 4. 检测引擎封装

### 4.1 检测器接口

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检测器接口定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class DetectionResult:
    """检测结果数据类"""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    

@dataclass
class ImageDetectionResult:
    """图像检测结果"""
    image_path: str
    filename: str
    detections: List[DetectionResult]
    inference_time: float
    annotated_image: Optional[np.ndarray] = None


class BaseDetector(ABC):
    """检测器基类"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def detect(self, image_path: str) -> ImageDetectionResult:
        """检测单张图像"""
        pass
    
    @abstractmethod
    def detect_batch(self, image_paths: List[str]) -> List[ImageDetectionResult]:
        """批量检测"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """获取模型信息"""
        pass
```

### 4.2 YOLOv8检测器实现

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8检测器实现
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional
from .base_detector import BaseDetector, DetectionResult, ImageDetectionResult


class YOLOv8Detector(BaseDetector):
    """YOLOv8检测器"""
    
    # 类别名称映射
    CLASS_NAMES = {
        0: 'Normal',
        1: 'Minor_Scratch',
        2: 'Severe_Scratch',
        3: 'Missing_Corner',
        4: 'Deformation',
        5: 'Mixed_Material'
    }
    
    # 类别颜色映射 (BGR格式)
    CLASS_COLORS = {
        0: (0, 255, 0),      # 绿色 - 正常件
        1: (0, 255, 255),    # 黄色 - 轻微划痕
        2: (0, 140, 255),    # 橙色 - 严重划痕
        3: (0, 0, 255),      # 红色 - 缺角
        4: (128, 0, 128),    # 紫色 - 变形
        5: (255, 0, 0)       # 蓝色 - 混料
    }
    
    def __init__(self, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        初始化检测器
        
        Args:
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        self.model = None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_path = None
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def detect(self, image_path: str) -> ImageDetectionResult:
        """
        检测单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            ImageDetectionResult: 检测结果
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        # 记录开始时间
        start_time = time.time()
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 执行推理
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 解析结果
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    detections.append(DetectionResult(
                        class_id=class_id,
                        class_name=self.CLASS_NAMES.get(class_id, 'Unknown'),
                        confidence=confidence,
                        bbox=tuple(bbox)
                    ))
        
        # 生成标注图像
        annotated_image = self._draw_detections(image.copy(), detections)
        
        # 提取文件名
        import os
        filename = os.path.basename(image_path)
        
        return ImageDetectionResult(
            image_path=image_path,
            filename=filename,
            detections=detections,
            inference_time=inference_time,
            annotated_image=annotated_image
        )
    
    def detect_batch(self, image_paths: List[str]) -> List[ImageDetectionResult]:
        """
        批量检测图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            List[ImageDetectionResult]: 检测结果列表
        """
        results = []
        for image_path in image_paths:
            result = self.detect(image_path)
            results.append(result)
        return results
    
    def _draw_detections(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            np.ndarray: 标注后的图像
        """
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det.bbox]
            color = self.CLASS_COLORS.get(det.class_id, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"{det.class_name} {det.confidence:.2%}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
            
            # 绘制标签文字
            cv2.putText(image, label, (x1, y1 - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 模型信息
        """
        if self.model is None:
            return {"status": "未加载"}
        
        return {
            "model_path": self.model_path,
            "model_type": "YOLOv8",
            "num_classes": len(self.CLASS_NAMES),
            "class_names": self.CLASS_NAMES,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold
        }
```

---

## 5. 应用打包

### 5.1 PyInstaller打包

#### 打包脚本

```python
# build.py
"""
应用打包脚本
"""

import PyInstaller.__main__
import os
import shutil

def build_app():
    """打包应用"""
    
    # PyInstaller参数
    args = [
        'src/gui/main_window.py',           # 主脚本
        '--name=螺丝缺陷检测系统',            # 应用名称
        '--windowed',                         # 无控制台窗口
        '--onefile',                          # 单文件打包
        '--icon=assets/icon.ico',             # 应用图标
        '--add-data=configs;configs',         # 添加配置文件
        '--add-data=models;models',           # 添加模型文件
        '--hidden-import=ultralytics',        # 隐藏导入
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=torch',
        '--hidden-import=PyQt5',
        '--noconfirm',                        # 不确认覆盖
        '--clean',                            # 清理临时文件
    ]
    
    # 执行打包
    PyInstaller.__main__.run(args)
    
    print("打包完成！")
    print(f"输出目录: dist/螺丝缺陷检测系统.exe")

if __name__ == '__main__':
    build_app()
```

#### 打包配置文件

```python
# build.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/gui/main_window.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('configs', 'configs'),
        ('models', 'models'),
    ],
    hiddenimports=[
        'ultralytics',
        'cv2',
        'numpy',
        'torch',
        'torchvision',
        'PyQt5',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='螺丝缺陷检测系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
)
```

### 5.2 打包命令

```bash
# 方式1: 使用脚本打包
python build.py

# 方式2: 使用spec文件打包
pyinstaller build.spec

# 方式3: 命令行打包
pyinstaller --name="螺丝缺陷检测系统" --windowed --onefile src/gui/main_window.py
```

### 5.3 打包注意事项

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 模型文件未包含 | 未添加到datas | 在datas中添加模型路径 |
| 依赖库缺失 | 隐藏导入未包含 | 在hiddenimports中添加 |
| 打包体积过大 | 包含不必要文件 | 使用excludes排除 |
| 运行时错误 | 路径问题 | 使用相对路径或sys._MEIPASS |

### 5.4 打包优化

```python
# 优化打包体积
excludes = [
    'matplotlib',
    'scipy',
    'pandas',
    'PIL',
    'tkinter',
    'unittest',
    'xmlrpc',
    'pydoc',
]

# 使用UPX压缩
upx = True
upx_exclude = ['vcruntime140.dll']
```

---

## 6. 部署配置

### 6.1 配置文件

```yaml
# configs/app_config.yaml
# 应用配置文件

# 应用信息
app:
  name: "螺丝缺陷检测系统"
  version: "1.0.0"
  author: "项目组"
  description: "基于YOLOv8的工业螺丝缺陷检测系统"

# 模型配置
model:
  default_path: "models/weights/best.pt"
  engine: "auto"  # auto/pytorch/onnx/tensorrt
  input_size: 640
  conf_threshold: 0.5
  iou_threshold: 0.45

# 检测配置
detection:
  max_batch_size: 32
  num_workers: 4
  enable_async: true
  timeout: 30  # 单张超时时间(秒)

# 界面配置
gui:
  theme: "light"  # light/dark
  window_size: [1280, 720]
  language: "zh_CN"
  font_size: 12

# 输出配置
output:
  result_dir: "results"
  auto_save: true
  export_format: "csv"  # csv/excel/pdf
  save_annotated: true

# 日志配置
logging:
  level: "INFO"
  file: "logs/app.log"
  max_size: 10  # MB
  backup_count: 5
```

### 6.2 配置管理器

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理器
"""

import yaml
import os
from pathlib import Path
from typing import Any, Optional


class ConfigManager:
    """配置管理器"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: str = "configs/app_config.yaml") -> dict:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            dict: 配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键，支持点号分隔（如 'model.conf_threshold'）
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        if self._config is None:
            self.load()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置键
            value: 配置值
        """
        if self._config is None:
            self.load()
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_path: str = "configs/app_config.yaml") -> None:
        """
        保存配置文件
        
        Args:
            config_path: 配置文件路径
        """
        if self._config is None:
            return
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)


# 全局配置实例
config = ConfigManager()
```

---

## 7. 部署验证

### 7.1 验证清单

| 验证项 | 验证内容 | 验证方法 | 预期结果 |
|--------|----------|----------|----------|
| 启动测试 | 应用能否正常启动 | 双击运行 | 主窗口显示 |
| 模型加载 | 模型是否正确加载 | 查看状态栏 | 显示模型信息 |
| 图像加载 | 图像是否正确显示 | 打开图像 | 图像正常显示 |
| 检测功能 | 检测是否正常工作 | 执行检测 | 显示检测结果 |
| 结果导出 | 报告是否正确导出 | 导出报告 | 文件生成成功 |
| 批量检测 | 批量检测是否正常 | 批量测试 | 全部完成 |
| 性能测试 | 是否满足时间要求 | 计时测试 | <180秒/20张 |
| 稳定性测试 | 长时间运行是否稳定 | 持续运行 | 无崩溃 |

### 7.2 自动化测试脚本

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
部署验证脚本
"""

import os
import sys
import time
import unittest
from pathlib import Path


class DeploymentTest(unittest.TestCase):
    """部署测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        cls.test_images_dir = Path("data/test/images")
        cls.model_path = Path("models/weights/best.pt")
    
    def test_01_model_exists(self):
        """测试模型文件是否存在"""
        self.assertTrue(self.model_path.exists(), f"模型文件不存在: {self.model_path}")
    
    def test_02_model_load(self):
        """测试模型加载"""
        from src.core.detector import YOLOv8Detector
        
        detector = YOLOv8Detector()
        result = detector.load_model(str(self.model_path))
        self.assertTrue(result, "模型加载失败")
    
    def test_03_single_detection(self):
        """测试单张检测"""
        from src.core.detector import YOLOv8Detector
        
        detector = YOLOv8Detector()
        detector.load_model(str(self.model_path))
        
        # 获取测试图像
        test_images = list(self.test_images_dir.glob("*.jpg"))[:1]
        self.assertTrue(len(test_images) > 0, "没有测试图像")
        
        result = detector.detect(str(test_images[0]))
        self.assertIsNotNone(result, "检测结果为空")
        self.assertGreater(len(result.detections), 0, "未检测到目标")
    
    def test_04_batch_detection_performance(self):
        """测试批量检测性能"""
        from src.core.detector import YOLOv8Detector
        
        detector = YOLOv8Detector()
        detector.load_model(str(self.model_path))
        
        # 获取20张测试图像
        test_images = list(self.test_images_dir.glob("*.jpg"))[:20]
        self.assertTrue(len(test_images) >= 20, "测试图像不足20张")
        
        # 执行批量检测
        start_time = time.time()
        results = detector.detect_batch([str(img) for img in test_images])
        elapsed_time = time.time() - start_time
        
        self.assertEqual(len(results), 20, "检测结果数量不正确")
        self.assertLess(elapsed_time, 180, f"批量检测超时: {elapsed_time:.1f}秒")
        
        print(f"\n批量检测性能: {elapsed_time:.1f}秒 / 20张 = {elapsed_time/20:.2f}秒/张")
    
    def test_05_detection_accuracy(self):
        """测试检测准确率"""
        from src.core.detector import YOLOv8Detector
        
        detector = YOLOv8Detector()
        detector.load_model(str(self.model_path))
        
        # 获取所有测试图像
        test_images = list(self.test_images_dir.glob("*.jpg"))
        
        correct = 0
        total = len(test_images)
        
        for img_path in test_images:
            # 从文件名推断真实类别
            true_class = img_path.stem.split('_')[0]
            
            result = detector.detect(str(img_path))
            if result.detections:
                pred_class = result.detections[0].class_name.lower()
                if true_class in pred_class or pred_class in true_class:
                    correct += 1
        
        accuracy = correct / total * 100
        self.assertGreaterEqual(accuracy, 95, f"准确率不足95%: {accuracy:.1f}%")
        
        print(f"\n检测准确率: {accuracy:.1f}% ({correct}/{total})")


if __name__ == '__main__':
    unittest.main(verbosity=2)
```

---

## 8. 使用说明

### 8.1 启动应用

#### 方式1: 源码运行

```bash
# 激活虚拟环境
conda activate yolo

# 运行应用
python src/gui/main_window.py
```

#### 方式2: 打包后运行

```bash
# 直接运行打包后的可执行文件
dist/螺丝缺陷检测系统.exe
```

### 8.2 操作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       使用操作流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 启动应用                                                   │
│      │                                                          │
│      ▼                                                          │
│   2. 加载图像                                                   │
│      • 点击"打开图像"选择单张/多张图像                          │
│      • 或点击"打开文件夹"选择图像文件夹                         │
│      │                                                          │
│      ▼                                                          │
│   3. 开始检测                                                   │
│      • 点击"开始检测"按钮                                       │
│      • 等待检测完成                                             │
│      │                                                          │
│      ▼                                                          │
│   4. 查看结果                                                   │
│      • 在图像区域查看标注结果                                   │
│      • 在结果列表查看详细信息                                   │
│      • 在统计面板查看统计信息                                   │
│      │                                                          │
│      ▼                                                          │
│   5. 导出报告                                                   │
│      • 点击"导出报告"按钮                                       │
│      • 选择导出格式和保存位置                                   │
│      │                                                          │
│      ▼                                                          │
│   6. 完成                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+O | 打开图像 |
| Ctrl+Shift+O | 打开文件夹 |
| F5 | 开始检测 |
| Ctrl+E | 导出报告 |
| Ctrl+Q | 退出应用 |
| Ctrl+Z | 撤销 |
| Ctrl+Plus | 放大图像 |
| Ctrl+Minus | 缩小图像 |

---

## 9. 故障排除

### 9.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 应用无法启动 | 依赖库缺失 | 重新安装依赖 |
| 模型加载失败 | 模型文件损坏 | 重新下载模型 |
| 检测速度慢 | 使用CPU推理 | 切换到GPU推理 |
| 检测结果差 | 模型未训练好 | 使用更好的模型 |
| 界面显示异常 | 分辨率不兼容 | 调整DPI设置 |
| 导出失败 | 权限不足 | 以管理员运行 |

### 9.2 日志查看

```python
# 日志文件位置
LOG_FILE = "logs/app.log"

# 查看最近日志
def view_recent_logs(lines=100):
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        logs = f.readlines()
        for line in logs[-lines:]:
            print(line.strip())
```

---

## 附录A：部署检查清单

- [ ] Python环境配置正确
- [ ] 所有依赖已安装
- [ ] 模型文件存在且完整
- [ ] 配置文件正确
- [ ] GPU驱动已安装（如使用GPU）
- [ ] 应用可以正常启动
- [ ] 检测功能正常
- [ ] 结果导出正常
- [ ] 性能满足要求

## 附录B：性能优化建议

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 使用GPU | 安装CUDA和cuDNN | 推理速度提升10倍 |
| 模型量化 | FP32转FP16/INT8 | 速度提升2-4倍 |
| 使用ONNX | 导出为ONNX格式 | 推理速度提升30-50% |
| 批处理 | 多张图像同时推理 | 吞吐量提升 |
| 图像缩放 | 降低输入分辨率 | 速度提升，精度略降 |

---

**文档版本历史**

| 版本 | 日期 | 修改内容 | 作者 |
|------|------|----------|------|
| V1.0 | 2026-05-27 | 初始版本 | 项目组 |