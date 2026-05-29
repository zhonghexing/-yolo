# 基于YOLOv8的钢材表面缺陷检测系统

## 项目概述

本项目是参加"基于AI视觉的行业应用创新赛"的竞赛作品，基于 YOLOv8 深度学习模型，自动识别和分类钢材表面的 6 种常见缺陷，提供 PyQt5 桌面应用和命令行两种操作方式。

### 核心特性

- **6 类缺陷检测**：龟裂、夹杂、斑块、麻点、氧化皮、划痕
- **三种检测模式**：图片检测、摄像头实时检测、视频文件检测
- **现代化 UI**：深色主题 + 工业风配色，左右分栏布局，支持拖放
- **一键部署**：26MB 部署包，解压双击即可运行

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.12 + Ultralytics 8.4 |
| 检测模型 | YOLOv8s（部署）/ YOLOv8m（训练） |
| 桌面 GUI | PyQt5 |
| 图像处理 | OpenCV, Pillow |
| 模型导出 | ONNX, TensorRT（export_model.py） |

## 缺陷类型

| 编号 | 类别 | 英文 | 框颜色 |
|------|------|------|--------|
| 0 | 龟裂 | crazing | 红色 |
| 1 | 夹杂 | inclusion | 橙色 |
| 2 | 斑块 | patches | 黄绿色 |
| 3 | 麻点 | pitted_surface | 紫红色 |
| 4 | 氧化皮 | rolled-in_scale | 深橙色 |
| 5 | 划痕 | scratches | 天蓝色 |

## 项目结构

```
D:/yolo/
├── app.py                  # PyQt5 桌面应用主程序
├── inference.py            # 推理引擎核心
├── train.py                # 训练脚本
├── train_optimized.py      # 优化版训练（v3）
├── train_v1_optimized.py   # v1 优化训练
├── evaluate.py             # 模型评估
├── demo.py                 # 竞赛演示（20样本/180秒）
├── export_model.py         # 模型导出（ONNX/TensorRT）
├── feedback.py             # 反馈模块（语音+视觉+日志）
├── visualization.py        # 可视化报告
├── constants.py            # 类别名称和颜色常量
├── compare_models.py       # 模型对比工具
├── check_gpu.py            # GPU 状态检查
├── monitor_and_test.py     # 训练监控 + 自动测试
├── run_app.py              # 应用启动器（环境检查）
├── build_desktop.spec      # PyInstaller 打包配置
├── app_icon.ico            # 应用图标
├── requirements.txt        # Python 依赖
├── environment.yml         # Conda 环境配置
├── yolov8n.pt / yolov8s.pt / yolov8m.pt  # 预训练基础模型
├── 基于AI视觉的工业零件缺陷检测方案.pptx  # 竞赛答辩 PPT
├── datasets/
│   └── neu_det/            # NEU-DET 钢材缺陷数据集
│       ├── data.yaml
│       ├── train/images/   # 1864 张（增强后）
│       ├── val/images/     # 324 张
│       └── test/images/    # 180 张
├── deploy/                 # 部署包源文件
│   ├── app.py              # GUI 主程序
│   ├── inference.py        # 推理引擎
│   ├── best.pt             # v1 模型 (YOLOv8s, mAP50=0.768)
│   ├── yolov8n.pt          # 备用模型
│   ├── README.md           # 部署文档
│   ├── test_images/        # 4 张测试图
│   └── *.bat               # 一键启动脚本
├── deploy_package.zip      # 部署包 (26MB)
├── runs/
│   ├── train/              # 训练结果（v1/v2/v3/v1_optimized）
│   ├── detect/             # 检测结果
│   ├── eval/               # 评估报告
│   └── demo/               # 演示结果
├── tools/                  # 工具脚本（增强/标注/划分）
└── docs/                   # 项目文档
```

## 快速开始

### 环境配置（开发机）

```bash
# 1. 创建 Conda 环境
conda create -n yolo_screw python=3.10
conda activate yolo_screw

# 2. 安装 PyTorch (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证
python check_gpu.py
```

### 模型训练

```bash
# 默认训练（yolov8m, 100 epochs）
python train.py

# 指定模型和参数
python train.py --model yolov8s.pt --epochs 150 --batch 16

# 恢复训练
python train.py --resume runs/train/screw_defect_v2/weights/last.pt

# 优化训练
python train_optimized.py --model runs/train/screw_defect_v2/weights/best.pt
```

### 启动应用

```bash
python app.py            # 直接启动 GUI
python run_app.py        # 带环境检查启动
```

### 推理检测

```bash
# 单张图片
python inference.py --image test.jpg

# 批量检测
python inference.py --dir datasets/neu_det/test/images

# 比赛演示
python demo.py --real --dir datasets/neu_det/test/images --samples 20
```

### 模型导出

```bash
python export_model.py --format onnx
```

## 模型版本

| 版本 | 模型 | 大小 | mAP@0.5 | 说明 |
|------|------|------|---------|------|
| v1 (screw_defect-11) | YOLOv8s | 21.5 MB | **0.768** | ★ 当前部署版本 |
| v2 (screw_defect_v2) | YOLOv8m | 197.9 MB | — | 大模型基线 |
| v3 (v3_optimized) | YOLOv8m | 49.7 MB | 0.675 | 优化参数训练 |
| v1_optimized | YOLOv8s | 21.5 MB | — | v1 优化尝试 |

### v3 各类别 AP@0.5

| 类别 | AP@0.5 |
|------|--------|
| scratches（划痕） | 0.899 |
| patches（斑块） | 0.867 |
| pitted_surface（麻点） | 0.827 |
| inclusion（夹杂） | 0.684 |
| rolled-in_scale（氧化皮） | 0.508 |
| crazing（龟裂） | 0.264 |

## 部署

将 `deploy_package.zip`（26MB）复制到目标电脑，解压后：
1. 确保已安装 Python 3.10+
2. 双击 `启动检测系统.bat` → 自动安装依赖 → 启动 GUI
3. 或先双击 `测试环境.bat` 检查环境

**部署目标**：i5-1240P + 16GB RAM（CPU 推理，~100ms/张）

## 文档

- [部署文档](deploy/README.md) — 部署和使用说明
- [需求分析](docs/requirements.md)
- [系统架构](docs/architecture.md)
- [数据集规范](docs/dataset.md)
- [训练方案](docs/training.md)
- [部署方案](docs/deployment.md)
- [测试方案](docs/testing.md)

## 许可证

本项目仅用于学术竞赛目的。

---

**项目状态**：已完成 | **最后更新**：2026年5月30日
