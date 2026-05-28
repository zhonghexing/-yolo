# CLAUDE.md - 钢材缺陷检测系统

## 项目概述
基于 YOLOv8 的钢材表面缺陷检测系统，参加"基于AI视觉的行业应用创新赛"。
目标：20 样本 / 180 秒 / 95% 准确率。

## 技术栈
- **Python 3.10** (conda env: `yolo`)
- **PyTorch 2.11.0+cu128** + **Ultralytics 8.4.56**
- **PyQt5** 桌面应用
- **OpenCV** 图像处理
- **YOLOv8m** 模型（26M 参数）

## 目录结构
```
D:/yolo/
├── app.py              # PyQt5 桌面应用（主入口）
├── train.py            # 训练脚本
├── inference.py        # 推理引擎（ScrewDefectDetector）
├── evaluate.py         # 模型评估（混淆矩阵、分类报告）
├── demo.py             # 比赛演示（20样本/180秒计时）
├── export_model.py     # 模型导出（ONNX/TensorRT）
├── feedback.py         # 反馈模块（语音/视觉/日志）
├── visualization.py    # 可视化报告
├── constants.py        # 常量（类别名、颜色）
├── run_app.py          # 应用启动器（环境检查）
├── monitor_and_test.py # 训练监控 + 自动测试
├── check_gpu.py        # GPU 状态检查
├── datasets/
│   ├── neu_det/        # NEU-DET 钢材缺陷数据集（主用）
│   │   ├── data.yaml   # 数据集配置
│   │   ├── train/images/  # 2592 张
│   │   ├── val/images/    # 648 张
│   │   └── test/images/   # 180 张
│   └── screws/         # 螺丝缺陷数据集（早期）
├── runs/
│   ├── train/          # 训练结果
│   │   ├── screw_defect-11/  # v1: yolov8s + imgsz800 (mAP50=0.77)
│   │   └── screw_defect_v2/  # v2: yolov8m + imgsz1024（训练中）
│   ├── detect/         # 验证结果
│   ├── eval/           # 评估报告
│   └── demo/           # 演示结果
├── tools/
│   ├── data_augment.py # 数据增强
│   ├── labelme_to_yolo.py # 标注格式转换
│   ├── split_dataset.py # 数据集划分
│   └── generate_sample_data.py # 生成样本
├── docs/               # 项目文档
├── requirements.txt    # 依赖列表
└── yolov8m.pt          # 预训练权重
```

## 数据集
- **NEU-DET**: 6 类钢材表面缺陷
  - `crazing`（龟裂）、`inclusion`（夹杂）、`patches`（斑块）
  - `pitted_surface`（麻点）、`rolled-in_scale`（氧化皮）、`scratches`（划痕）
- 总计 3420 张图，每类约 200-280 张

## 常用命令
```bash
# 激活环境
conda activate yolo

# 训练
python train.py                          # 默认参数
python train.py --model yolov8s.pt       # 指定模型
python train.py --resume runs/train/screw_defect_v2/weights/last.pt  # 恢复训练

# 评估
python evaluate.py                       # 自动找最新 best.pt

# 比赛演示
python demo.py --real --dir datasets/neu_det/test/images --samples 20

# 启动 GUI
python app.py

# 检查 GPU
python check_gpu.py
```

## 训练配置（当前 v2）
| 参数 | 值 | 说明 |
|------|------|------|
| model | yolov8m.pt | 26M 参数 |
| imgsz | 1024 | 输入尺寸 |
| batch | 16 | 批量大小 |
| epochs | 150 | 最大轮数 |
| patience | 30 | 早停耐心 |
| optimizer | AdamW | 优化器 |
| lr0 | 0.001 | 初始学习率 |
| cache | disk | 磁盘缓存（省 RAM） |
| workers | 2 | 数据加载线程 |

## 部署目标
- **训练机**: RTX 5070 (12GB VRAM) + Windows 11
- **部署机**: i5-1240P + 16GB RAM（CPU 推理）
- 部署时导出 ONNX 格式，CPU 推理加速 2-3x

## 注意事项
- **RAM 有限**: `cache=True` 会爆内存，必须用 `cache="disk"`
- **RTX 5070**: sm_120 架构，需要 PyTorch nightly 或 2.11+
- **类别名中英文**: `constants.py` 中 `CLASS_NAMES`（英文）和 `CLASS_NAMES_CN`（中文）
- **inference.py 的 `detect_single()`** 接受文件路径，视频帧需直接调用 `model(frame)`
- **app.py** 支持深色/浅色主题切换，视频流检测（摄像头/视频文件）

## 代码风格
- 中文注释和文档
- 函数/变量用英文命名
- PyQt5 样式表定义在文件顶部常量中
- 检测结果用 dataclass（`SingleDetection`、`ImageDetectionResult`）
