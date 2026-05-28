# 基于YOLOv8的钢材表面缺陷检测系统

## 项目概述

本项目是参加"基于AI视觉的行业应用创新赛"的竞赛作品，旨在构建一套高效、准确的钢材表面缺陷视觉检测系统。系统采用YOLOv8深度学习模型，能够自动识别和分类钢材表面的多种缺陷类型，实现工业生产线上的自动化质量检测。

### 项目亮点

- **高精度检测**：基于YOLOv8先进的目标检测架构，目标95%以上的检测准确率
- **多缺陷识别**：支持6种缺陷类型的实时分类检测（龟裂、夹杂、斑块、麻点、氧化皮、划痕）
- **快速推理**：单张图像推理时间<100ms，满足实时检测需求
- **桌面应用**：提供友好的PyQt5 GUI界面，便于工业现场操作使用
- **竞赛达标**：满足20个样本/180秒的检测效率要求

## 技术栈

| 类别 | 技术选型 | 说明 |
|------|----------|------|
| 深度学习框架 | PyTorch 2.x | 模型训练与推理 |
| 目标检测模型 | YOLOv8 (Ultralytics) | 最新的YOLO系列检测模型 |
| GUI框架 | PyQt5 / Tkinter | 桌面应用界面开发 |
| 数据处理 | OpenCV, Pillow | 图像预处理与增强 |
| 模型导出 | ONNX, TensorRT | 模型格式转换与加速 |

## 缺陷类型定义

本系统检测以下6种钢材表面缺陷类型（NEU-DET数据集）：

| 编号 | 类别名称 | 英文标识 | 说明 |
|------|----------|----------|------|
| 0 | 龟裂 | crazing | 钢材表面裂纹缺陷 |
| 1 | 夹杂 | inclusion | 材料内部夹杂物 |
| 2 | 斑块 | patches | 表面斑块状缺陷 |
| 3 | 麻点 | pitted_surface | 表面麻点状缺陷 |
| 4 | 氧化皮 | rolled-in_scale | 轧制氧化皮缺陷 |
| 5 | 划痕 | scratches | 表面划痕缺陷 |

## 项目结构

```
D:/yolo/
├── README.md                    # 项目说明文档
├── app.py                       # PyQt5 桌面应用主程序
├── inference.py                 # 推理引擎核心模块
├── train.py                     # 模型训练脚本
├── evaluate.py                  # 模型评估脚本
├── demo.py                      # 竞赛演示脚本
├── export_model.py              # 模型导出脚本
├── feedback.py                  # 反馈机制模块（视觉+语音+日志）
├── visualization.py             # 可视化模块
├── run_app.py                   # 应用启动脚本
├── check_gpu.py                 # GPU状态检查
├── docs/                        # 项目文档目录
│   ├── requirements.md          # 需求分析文档
│   ├── architecture.md          # 系统架构设计
│   ├── dataset.md               # 数据集规范
│   ├── training.md              # 模型训练方案
│   ├── deployment.md            # 部署方案
│   └── testing.md               # 测试方案
├── datasets/                    # 数据集目录
│   ├── neu_det/                 # NEU-DET钢材缺陷数据集
│   │   ├── data.yaml            # 数据集配置
│   │   ├── train/               # 训练集 (1296张)
│   │   ├── val/                 # 验证集 (324张)
│   │   └── test/                # 测试集 (180张)
│   └── screws/                  # 螺丝数据集（备用）
├── runs/                        # 训练运行结果
│   └── train/
│       └── screw_defect_cpu/    # CPU训练结果
│           └── weights/
│               ├── best.pt      # 最佳模型
│               └── last.pt      # 最新模型
├── tools/                       # 工具脚本
│   ├── data_augment.py          # 数据增强
│   ├── generate_sample_data.py  # 样本生成
│   ├── labelme_to_yolo.py       # 标注格式转换
│   └── split_dataset.py         # 数据集划分
├── requirements.txt             # Python依赖
├── environment.yml              # Conda环境配置
└── setup_env.bat                # 环境设置脚本
```

## 快速开始

### 环境配置

```bash
# 1. 创建 Conda 环境
conda create -n yolo_screw python=3.10
conda activate yolo_screw

# 2. 安装 PyTorch (CUDA 12.8)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import ultralytics; print(ultralytics.__version__)"
python check_gpu.py
```

### 模型训练

```bash
# GPU训练（推荐）
python train.py --epochs 100 --batch 16 --device 0

# CPU训练
python train.py --epochs 100 --batch 8 --device cpu
```

### 启动应用

```bash
# 启动GUI应用
python run_app.py

# 或直接运行
python app.py
```

### 运行演示

```bash
# 竞赛演示模式（20样本/180秒）
python demo.py --samples 20 --limit 180
```

### 模型导出

```bash
# 导出为ONNX格式
python export_model.py --format onnx

# 导出为TorchScript格式
python export_model.py --format torchscript
```

## 性能指标

### 检测精度（目标）

| 指标 | 目标值 | 当前值 | 说明 |
|------|--------|--------|------|
| mAP@0.5 | ≥95% | 训练中 | IoU阈值为0.5时的平均精度 |
| mAP@0.5:0.95 | ≥85% | 训练中 | IoU阈值0.5-0.95的平均精度 |
| Precision | ≥95% | 训练中 | 精确率 |
| Recall | ≥94% | 训练中 | 召回率 |

### 各类别检测性能（目标）

| 缺陷类型 | Precision | Recall | mAP@0.5 |
|----------|-----------|--------|---------|
| 龟裂 | ≥93% | ≥92% | ≥94% |
| 夹杂 | ≥95% | ≥94% | ≥96% |
| 斑块 | ≥96% | ≥95% | ≥97% |
| 麻点 | ≥95% | ≥94% | ≥96% |
| 氧化皮 | ≥94% | ≥93% | ≥95% |
| 划痕 | ≥96% | ≥95% | ≥97% |

### 推理速度（目标）

| 设备 | 单张推理时间 | 批量处理(20张) | FPS |
|------|-------------|----------------|-----|
| RTX 5070 GPU | <20ms | <2s | >50 |
| CPU (i7) | <100ms | <5s | >10 |

## 竞赛指标达成

| 竞赛要求 | 目标值 | 当前状态 | 备注 |
|----------|--------|----------|------|
| 检测样本数 | 20个 | 待测试 | demo.py支持 |
| 检测时间 | ≤180秒 | 待测试 | 目标<30秒 |
| 检测准确率 | ≥95% | 训练中 | 需要完成训练 |
| 桌面应用演示 | 必需 | 已完成 | app.py可用 |

## 文档目录

- [需求分析文档](docs/requirements.md) - 详细的系统需求分析
- [系统架构设计](docs/architecture.md) - 整体架构与模块设计
- [数据集规范](docs/dataset.md) - 数据集构建与标注规范
- [模型训练方案](docs/training.md) - 训练策略与超参数配置
- [部署方案](docs/deployment.md) - 模型部署与优化方案
- [测试方案](docs/testing.md) - 系统测试与验证方案

## 团队成员

| 角色 | 职责 |
|------|------|
| 项目负责人 | 整体规划与协调 |
| 算法工程师 | 模型选型与训练 |
| 数据工程师 | 数据采集与标注 |
| 软件工程师 | GUI开发与系统集成 |
| 测试工程师 | 系统测试与质量保证 |

## 许可证

本项目仅用于学术竞赛目的，代码和模型仅供参考学习使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库：[GitHub链接]
- 邮箱：[联系邮箱]

---

**项目状态**：开发中 | **最后更新**：2026年5月28日