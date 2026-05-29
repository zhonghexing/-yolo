# CLAUDE.md - 钢材缺陷检测系统

## 项目概述
基于 YOLOv8 的钢材表面缺陷检测系统，参加"基于AI视觉的行业应用创新赛"。
目标：20 样本 / 180 秒 / 95% 准确率。

## 技术栈
- **Python 3.10** (conda env: `yolo_screw`)
- **PyTorch 2.12.0+cu126** + **Ultralytics 8.4.56**
- **PyQt5** 桌面应用
- **OpenCV** 图像处理
- **YOLOv8s** 模型（部署用，21.5MB）

## 目录结构
```
D:/yolo/
├── app.py              # PyQt5 桌面应用（主入口）
├── inference.py        # 推理引擎（ScrewDefectDetector）
├── train.py            # 训练脚本（默认 yolov8m）
├── train_optimized.py  # 优化版训练脚本（v3）
├── train_v1_optimized.py # v1 优化训练脚本
├── evaluate.py         # 模型评估（混淆矩阵、分类报告）
├── demo.py             # 比赛演示（20样本/180秒计时）
├── export_model.py     # 模型导出（ONNX/TensorRT）
├── feedback.py         # 反馈模块（语音/视觉/日志）
├── visualization.py    # 可视化报告
├── constants.py        # 常量（类别名、颜色）
├── compare_models.py   # 模型对比工具
├── check_gpu.py        # GPU 状态检查
├── monitor_and_test.py # 训练监控 + 自动测试
├── run_app.py          # 应用启动器（环境检查）
├── build_desktop.spec  # PyInstaller 打包配置
├── app_icon.ico        # 应用图标
├── CLAUDE.md / README.md
├── requirements.txt    # 依赖列表
├── environment.yml     # Conda 环境配置
├── yolov8s.pt          # 预训练权重 (22MB, v1 训练基准)
├── yolov8m.pt          # 预训练权重 (50MB, v2/v3 训练基准)
├── yolov8n.pt          # 预训练权重 (6.3MB, 快速调试/备用)
├── 基于AI视觉的工业零件缺陷检测方案.pptx  # 竞赛答辩 PPT
├── datasets/
│   ├── neu_det/        # NEU-DET 钢材缺陷数据集（主用）
│   │   ├── data.yaml   # 数据集配置
│   │   ├── train/images/  # 1864 张（增强后）
│   │   ├── val/images/    # 324 张
│   │   └── test/images/   # 180 张
│   └── screws/         # 螺丝缺陷数据集（早期）
├── deploy/             # 部署包源文件
│   ├── app.py          # GUI 主程序
│   ├── inference.py    # 推理引擎
│   ├── best.pt         # v1 训练模型 (YOLOv8s, mAP50=0.768)
│   ├── yolov8n.pt      # 备用模型
│   ├── README.md       # 部署文档
│   ├── test_images/    # 4 张测试图片
│   └── *.bat           # 一键启动脚本
├── deploy_package.zip  # 部署包 (26MB)
├── runs/
│   ├── train/
│   │   ├── screw_defect-11/         # v1: yolov8s (mAP50=0.768) ★部署用
│   │   ├── screw_defect_v2/         # v2: yolov8m + imgsz1024
│   │   ├── screw_defect_v3_optimized/  # v3: v2 基础+优化
│   │   └── screw_defect_v1_optimized/  # v1 优化版
│   ├── detect/         # 验证结果
│   ├── eval/           # 评估报告
│   └── demo/           # 演示结果
├── tools/
│   ├── data_augment.py # 数据增强
│   ├── augment_weak_classes.py # 弱类别增强
│   ├── labelme_to_yolo.py # 标注格式转换
│   ├── split_dataset.py # 数据集划分
│   └── generate_sample_data.py # 生成样本
└── docs/               # 项目文档
```

## 数据集
- **NEU-DET**: 6 类钢材表面缺陷
  - `crazing`（龟裂）、`inclusion`（夹杂）、`patches`（斑块）
  - `pitted_surface`（麻点）、`rolled-in_scale`（氧化皮）、`scratches`（划痕）
- 训练集 1864 张（弱类别已增强：crazing/rolled-in_scale 各 500 张）

## 常用命令
```bash
# 激活环境
conda activate yolo_screw

# 训练
python train.py                          # 默认参数
python train.py --model yolov8s.pt       # 指定模型
python train.py --resume runs/train/screw_defect_v2/weights/last.pt  # 恢复训练
python train_optimized.py --model runs/train/screw_defect_v2/weights/best.pt  # v3 优化训练

# 评估
python evaluate.py                       # 自动找最新 best.pt

# 比赛演示
python demo.py --real --dir datasets/neu_det/test/images --samples 20

# 启动 GUI
python app.py

# 检查 GPU
python check_gpu.py

# 数据增强
python tools/augment_weak_classes.py     # 增强弱类别
```

## 训练配置（当前 v3）
| 参数 | 值 | 说明 |
|------|------|------|
| model | v2 best.pt | 基于 v2 继续训练 |
| imgsz | 1024 | 输入尺寸 |
| batch | 8 | 批量大小 |
| epochs | 300 | 最大轮数（早停于 295） |
| patience | 50 | 早停耐心 |
| optimizer | AdamW | 优化器 |
| lr0 | 0.0005 | 初始学习率 |
| cache | disk | 磁盘缓存（省 RAM） |
| workers | 2 | 数据加载线程 |
| mixup | 0.15 | MixUp 增强 |
| copy_paste | 0.1 | 复制粘贴增强 |
| shear | 5.0 | 剪切变换 |
| flipud | 0.3 | 上下翻转 |

## 模型版本对比
| 版本 | 模型 | 大小 | mAP@0.5 | mAP@0.5:0.95 | 部署 |
|------|------|------|---------|---------------|------|
| v1 (screw_defect-11) | YOLOv8s | 21.5 MB | 0.768 | — | ★ 当前部署 |
| v2 (screw_defect_v2) | YOLOv8m | 197.9 MB | — | — | — |
| v3 (v3_optimized) | YOLOv8m | 49.7 MB | 0.675 | 0.405 | — |
| v1_optimized | YOLOv8s | 21.5 MB | — | — | — |

## v3 评估结果 (screw_defect_v3_optimized)
| 指标 | 值 |
|------|------|
| mAP@0.5 | 0.675 |
| mAP@0.5:0.95 | 0.405 |
| Precision | 0.706 |
| Recall | 0.740 |
| 推理速度 | 13.8 ms/张（GPU） |

各类别 AP@0.5:
| 类别 | AP@0.5 |
|------|--------|
| scratches（划痕） | 0.899 |
| patches（斑块） | 0.867 |
| pitted_surface（麻点） | 0.827 |
| inclusion（夹杂） | 0.684 |
| rolled-in_scale（氧化皮） | 0.508 |
| crazing（龟裂） | 0.264 |

## 部署
- **训练机**: RTX 5070 (12GB VRAM) + Windows 11
- **部署机**: i5-1240P + 16GB RAM（CPU 推理）
- **部署包**: `deploy_package.zip` (26MB)，解压后双击 `启动检测系统.bat` 一键安装依赖并启动
- **模型**: 当前部署使用 v1 (screw_defect-11, YOLOv8s, mAP50=0.768)
- 详见 `deploy/README.md`

## 注意事项
- **RAM 有限**: `cache=True` 会爆内存，必须用 `cache="disk"`
- **RTX 5070**: sm_120 架构，需要 PyTorch 2.11+（nightly）
- **类别名中英文**: `constants.py` 中 `CLASS_NAMES`（英文）和 `CLASS_NAMES_CN`（中文）
- **inference.py 的 `detect_single()`** 接受文件路径，视频帧需直接调用 `model(frame)`
- **app.py** 支持深色/浅色主题切换，图片/摄像头/视频三种检测模式
- **弱类别增强**: crazing 和 rolled-in_scale 已通过 `augment_weak_classes.py` 增强到 500 张
- **模型备用**: 若 best.pt 不存在，自动回退到 yolov8n.pt

## 代码风格
- 中文注释和文档
- 函数/变量用英文命名
- PyQt5 样式表定义在文件顶部常量中
- 检测结果用 dataclass（`SingleDetection`、`ImageDetectionResult`）
