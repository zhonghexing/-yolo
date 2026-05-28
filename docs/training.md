# 模型训练方案文档

## 文档信息

| 项目 | 内容 |
|------|------|
| 文档名称 | 基于YOLOv8的工业螺丝缺陷检测系统 - 模型训练方案 |
| 文档版本 | V1.0 |
| 创建日期 | 2026年5月27日 |
| 最后更新 | 2026年5月27日 |

---

## 1. 模型选型

### 1.1 YOLO系列对比

| 模型 | 发布年份 | mAP@0.5 | 速度(FPS) | 参数量 | 特点 |
|------|----------|---------|-----------|--------|------|
| YOLOv5 | 2020 | 68.9% | 140 | 7.2M | 成熟稳定，社区活跃 |
| YOLOv7 | 2022 | 69.7% | 161 | 36.9M | 高精度，结构复杂 |
| YOLOv8 | 2023 | 72.3% | 280 | 3.2M | 最新架构，性能优异 |
| YOLOv9 | 2024 | 73.8% | 256 | 7.1M | 创新架构，持续优化 |
| YOLOv10 | 2024 | 74.3% | 300 | 2.3M | NMS-free，实时性佳 |

**选择YOLOv8的理由**：

1. **架构先进**：采用C2f模块、解耦头、Anchor-free等最新技术
2. **性能均衡**：精度和速度的最佳平衡
3. **易于使用**：Ultralytics提供统一API，文档完善
4. **生态丰富**：社区活跃，预训练模型丰富
5. **部署友好**：支持多种导出格式（ONNX、TensorRT等）

### 1.2 YOLOv8版本选择

| 版本 | 参数量 | mAP@0.5 | 速度(CPU) | 速度(GPU) | 适用场景 |
|------|--------|---------|-----------|-----------|----------|
| YOLOv8n | 3.2M | 37.3% | 80ms | 0.99ms | 移动端/边缘设备 |
| YOLOv8s | 11.2M | 44.9% | 128ms | 1.20ms | 轻量级部署 |
| YOLOv8m | 25.9M | 50.2% | 234ms | 1.83ms | 平衡选择 |
| YOLOv8l | 43.7M | 52.9% | 375ms | 2.39ms | 高精度需求 |
| YOLOv8x | 68.2M | 53.9% | 479ms | 3.53ms | 最高精度 |

**选择建议**：
- **开发调试阶段**：YOLOv8n（快速迭代）
- **竞赛演示阶段**：YOLOv8s 或 YOLOv8m（精度与速度平衡）
- **生产部署阶段**：根据硬件条件选择

**本项目选择**：YOLOv8s（兼顾精度和推理速度）

---

## 2. 训练环境配置

### 2.1 硬件配置

| 组件 | 最低配置 | 推荐配置 | 说明 |
|------|----------|----------|------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 9 | 数据预处理 |
| GPU | GTX 1060 6GB | RTX 3060 12GB / RTX 4070 | 模型训练 |
| 内存 | 16GB | 32GB | 大批量训练 |
| 硬盘 | 100GB SSD | 500GB NVMe SSD | 数据和模型存储 |

### 2.2 软件配置

| 软件 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Windows 10/11 (64位) | 或 Ubuntu 20.04/22.04 |
| Python | 3.8 - 3.11 | 推荐3.10 |
| CUDA | 11.8 或 12.1 | GPU加速 |
| cuDNN | 8.x | 深度学习加速 |
| PyTorch | 2.0+ | 深度学习框架 |
| Ultralytics | 8.x | YOLOv8实现 |

### 2.3 环境安装

```bash
# 1. 创建虚拟环境
conda create -n yolo python=3.10
conda activate yolo

# 2. 安装PyTorch（CUDA 11.8版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装Ultralytics
pip install ultralytics

# 4. 验证安装
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 5. 安装其他依赖
pip install opencv-python pillow matplotlib tensorboard
```

### 2.4 requirements.txt

```txt
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# 数据处理
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

# 可视化
matplotlib>=3.7.0
tensorboard>=2.14.0

# 工具
tqdm>=4.65.0
pyyaml>=6.0
requests>=2.31.0
```

---

## 3. 数据集准备

### 3.1 数据集配置

```yaml
# configs/dataset.yaml
path: D:/yolo/data          # 数据集根目录
train: train/images         # 训练集图像路径
val: val/images             # 验证集图像路径
test: test/images           # 测试集图像路径

nc: 6                       # 类别数量

names:
  0: Normal                 # 正常件
  1: Minor_Scratch          # 轻微划痕
  2: Severe_Scratch         # 严重划痕
  3: Missing_Corner         # 缺角
  4: Deformation            # 变形
  5: Mixed_Material         # 混料
```

### 3.2 数据目录验证

```python
# 验证数据集结构
import os
from pathlib import Path

def verify_dataset(data_dir):
    """验证数据集目录结构"""
    data_path = Path(data_dir)
    
    # 检查必要目录
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            print(f"错误: 缺少目录 {dir_path}")
            return False
    
    # 统计文件数量
    for split in ['train', 'val']:
        img_dir = data_path / split / 'images'
        lbl_dir = data_path / split / 'labels'
        
        img_files = set(f.stem for f in img_dir.glob('*.jpg'))
        lbl_files = set(f.stem for f in lbl_dir.glob('*.txt'))
        
        # 检查图像和标注文件是否匹配
        missing_labels = img_files - lbl_files
        missing_images = lbl_files - img_files
        
        if missing_labels:
            print(f"警告: {split}集中有{len(missing_labels)}张图像缺少标注文件")
        if missing_images:
            print(f"警告: {split}集中有{len(missing_images)}个标注文件缺少对应图像")
        
        print(f"{split}集: {len(img_files)}张图像, {len(lbl_files)}个标注文件")
    
    return True

# 执行验证
verify_dataset('D:/yolo/data')
```

---

## 4. 训练超参数配置

### 4.1 核心超参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| epochs | 100 | 50-300 | 训练轮数 |
| batch | 16 | 8-64 | 批次大小 |
| imgsz | 640 | 416-1280 | 输入图像尺寸 |
| lr0 | 0.01 | 0.001-0.01 | 初始学习率 |
| lrf | 0.01 | 0.001-0.01 | 最终学习率(lr0 * lrf) |
| momentum | 0.937 | 0.9-0.99 | SGD动量 |
| weight_decay | 0.0005 | 0.0001-0.001 | 权重衰减 |
| warmup_epochs | 3.0 | 1-5 | 预热轮数 |
| warmup_momentum | 0.8 | 0.5-0.9 | 预热动量 |
| warmup_bias_lr | 0.1 | 0.05-0.2 | 预热偏置学习率 |

### 4.2 损失函数权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| box | 7.5 | 边界框损失权重 |
| cls | 0.5 | 分类损失权重 |
| dfl | 1.5 | 分布焦点损失权重 |

### 4.3 数据增强参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hsv_h | 0.015 | HSV色调增强范围 |
| hsv_s | 0.7 | HSV饱和度增强范围 |
| hsv_v | 0.4 | HSV亮度增强范围 |
| degrees | 0.0 | 旋转角度范围 |
| translate | 0.1 | 平移范围 |
| scale | 0.5 | 缩放范围 |
| shear | 0.0 | 剪切角度 |
| perspective | 0.0 | 透视变换 |
| flipud | 0.0 | 上下翻转概率 |
| fliplr | 0.5 | 左右翻转概率 |
| mosaic | 1.0 | Mosaic增强概率 |
| mixup | 0.0 | Mixup增强概率 |
| copy_paste | 0.0 | 复制粘贴增强概率 |

### 4.4 训练配置文件

```python
# configs/train_config.yaml
# YOLOv8训练配置

# 模型配置
model: yolov8s.pt           # 预训练模型路径
task: detect                 # 任务类型
mode: train                  # 运行模式

# 数据配置
data: configs/dataset.yaml   # 数据集配置文件

# 训练参数
epochs: 150                  # 训练轮数
batch: 16                    # 批次大小
imgsz: 640                   # 输入图像尺寸
patience: 50                 # 早停耐心值
save: true                   # 保存训练结果
save_period: 10              # 每N轮保存一次
workers: 8                   # 数据加载线程数
cache: true                  # 缓存数据集

# 优化器配置
optimizer: AdamW             # 优化器选择
lr0: 0.001                   # 初始学习率
lrf: 0.01                    # 最终学习率比例
momentum: 0.937              # SGD动量
weight_decay: 0.0005         # 权重衰减
warmup_epochs: 3.0           # 预热轮数
warmup_momentum: 0.8         # 预热动量
warmup_bias_lr: 0.1          # 预热偏置学习率

# 损失函数权重
box: 7.5                    # 边界框损失权重
cls: 0.5                    # 分类损失权重
dfl: 1.5                    # DFL权重

# 数据增强
hsv_h: 0.015                # 色调增强
hsv_s: 0.7                  # 饱和度增强
hsv_v: 0.4                  # 亮度增强
degrees: 10.0               # 旋转角度
translate: 0.1              # 平移范围
scale: 0.5                  # 缩放范围
fliplr: 0.5                 # 左右翻转
mosaic: 1.0                 # Mosaic增强
mixup: 0.1                  # Mixup增强

# 设备配置
device: 0                   # GPU设备号，'cpu'表示CPU训练
amp: true                   # 混合精度训练
```

---

## 5. 训练流程

### 5.1 训练流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                       模型训练流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                              │
│   │ 1. 环境准备 │                                              │
│   │ • 安装依赖  │                                              │
│   │ • 配置GPU   │                                              │
│   │ • 验证环境  │                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ 2. 数据准备 │                                              │
│   │ • 数据集划分│                                              │
│   │ • 格式验证  │                                              │
│   │ • 数据增强  │                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ 3. 模型初始化│                                             │
│   │ • 加载预训练│                                              │
│   │ • 配置模型  │                                              │
│   │ • 设置优化器│                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ 4. 训练循环 │◄──────────────────────────────┐             │
│   │ • 前向传播  │                               │             │
│   │ • 计算损失  │                               │             │
│   │ • 反向传播  │                               │             │
│   │ • 更新参数  │                               │             │
│   └──────┬──────┘                               │             │
│          │                                       │             │
│          ▼                                       │             │
│   ┌─────────────┐                               │             │
│   │ 5. 验证评估 │                               │             │
│   │ • 计算mAP   │                               │             │
│   │ • 记录指标  │                               │             │
│   │ • 保存最佳  │                               │             │
│   └──────┬──────┘                               │             │
│          │                                       │             │
│          ▼                                       │             │
│   ┌─────────────┐         ┌─────────┐           │             │
│   │ 6. 早停检查 │────────>│ 继续？  │──Yes──────┘             │
│   │ • 检查收敛  │         └────┬────┘                         │
│   └─────────────┘              │No                             │
│                                ▼                               │
│                         ┌─────────────┐                        │
│                         │ 7. 训练完成 │                        │
│                         │ • 保存模型  │                        │
│                         │ • 生成报告  │                        │
│                         │ • 导出模型  │                        │
│                         └─────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 训练脚本

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 模型训练脚本
"""

from ultralytics import YOLO
import argparse
import yaml
from pathlib import Path


def train(config_path: str = None, **kwargs):
    """
    执行模型训练
    
    Args:
        config_path: 配置文件路径
        **kwargs: 额外的训练参数
    """
    # 加载配置
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 合并命令行参数
    config.update(kwargs)
    
    # 加载预训练模型
    model_path = config.get('model', 'yolov8s.pt')
    model = YOLO(model_path)
    
    # 开始训练
    results = model.train(
        data=config.get('data', 'configs/dataset.yaml'),
        epochs=config.get('epochs', 150),
        batch=config.get('batch', 16),
        imgsz=config.get('imgsz', 640),
        patience=config.get('patience', 50),
        save=config.get('save', True),
        save_period=config.get('save_period', 10),
        device=config.get('device', 0),
        workers=config.get('workers', 8),
        optimizer=config.get('optimizer', 'AdamW'),
        lr0=config.get('lr0', 0.001),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3.0),
        warmup_momentum=config.get('warmup_momentum', 0.8),
        warmup_bias_lr=config.get('warmup_bias_lr', 0.1),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        hsv_h=config.get('hsv_h', 0.015),
        hsv_s=config.get('hsv_s', 0.7),
        hsv_v=config.get('hsv_v', 0.4),
        degrees=config.get('degrees', 10.0),
        translate=config.get('translate', 0.1),
        scale=config.get('scale', 0.5),
        fliplr=config.get('fliplr', 0.5),
        mosaic=config.get('mosaic', 1.0),
        mixup=config.get('mixup', 0.1),
        amp=config.get('amp', True),
        cache=config.get('cache', True),
        project=config.get('project', 'runs/train'),
        name=config.get('name', 'exp'),
        exist_ok=config.get('exist_ok', False),
        pretrained=config.get('pretrained', True),
        verbose=config.get('verbose', True),
        seed=config.get('seed', 42),
    )
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8模型训练')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default=None,
                        help='训练设备 (0, 1, cpu)')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称')
    
    args = parser.parse_args()
    
    # 构建参数字典
    kwargs = {k: v for k, v in vars(args).items() 
              if v is not None and k != 'config'}
    
    # 执行训练
    results = train(args.config, **kwargs)
    
    print(f"\n训练完成！")
    print(f"最佳模型保存路径: {results.save_dir}/weights/best.pt")


if __name__ == '__main__':
    main()
```

### 5.3 训练命令

```bash
# 使用默认配置训练
python src/train.py

# 指定参数训练
python src/train.py --model yolov8s.pt --epochs 150 --batch 16 --device 0

# 使用配置文件训练
python src/train.py --config configs/train_config.yaml

# 继续之前的训练
python src/train.py --model runs/train/exp/weights/last.pt --resume
```

---

## 6. 训练监控

### 6.1 TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir runs/train --port 6006

# 浏览器访问
# http://localhost:6006
```

### 6.2 监控指标

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| train/box_loss | 边界框损失 | 持续下降 |
| train/cls_loss | 分类损失 | 持续下降 |
| train/dfl_loss | DFL损失 | 持续下降 |
| metrics/precision | 精确率 | 持续上升 |
| metrics/recall | 召回率 | 持续上升 |
| metrics/mAP50 | mAP@0.5 | 持续上升 |
| metrics/mAP50-95 | mAP@0.5:0.95 | 持续上升 |
| val/box_loss | 验证边界框损失 | 下降后平稳 |
| val/cls_loss | 验证分类损失 | 下降后平稳 |
| lr/pg0 | 学习率 | 先升后降 |

### 6.3 训练曲线示例

```
损失曲线                              精度曲线
    │                                      │
1.0 ┤ ╲                              1.0 ┤        ╭────────
    │  ╲                                  │       ╱
0.8 ┤   ╲                           0.8 ┤      ╱
    │    ╲                                │     ╱
0.6 ┤     ╲                         0.6 ┤    ╱
    │      ╲                              │   ╱
0.4 ┤       ╲╲                       0.4 ┤  ╱
    │         ╲╲                          │ ╱
0.2 ┤           ╲╲╲                  0.2 ┤╱
    │              ╲╲╲────────            │
0.0 ┼──────────────────────────      0.0 ┼──────────────────
    0    20    40    60    80   100       0    20    40    60    80   100
              Epoch                                  Epoch
    ── train_loss    ── val_loss           ── mAP50    ── mAP50-95
```

### 6.4 训练日志示例

```
Epoch 1/150
---------
GPU_mem: 4.2G | box_loss: 1.234 | cls_loss: 0.856 | dfl_loss: 0.432 | 
Val: mAP50: 0.456 | mAP50-95: 0.234

Epoch 2/150
---------
GPU_mem: 4.2G | box_loss: 0.987 | cls_loss: 0.654 | dfl_loss: 0.398 | 
Val: mAP50: 0.567 | mAP50-95: 0.345

...

Epoch 150/150
---------
GPU_mem: 4.2G | box_loss: 0.234 | cls_loss: 0.089 | dfl_loss: 0.156 | 
Val: mAP50: 0.968 | mAP50-95: 0.892

训练完成！
最佳模型: runs/train/exp/weights/best.pt
最佳mAP50: 0.968 @ epoch 145
```

---

## 7. 超参数调优

### 7.1 调优策略

```
┌─────────────────────────────────────────────────────────────────┐
│                       超参数调优流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                              │
│   │ 1. 基准测试 │                                              │
│   │ • 默认参数  │                                              │
│   │ • 记录基准  │                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ 2. 学习率   │                                              │
│   │ • 学习率扫描│                                              │
│   │ • 选择最优  │                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ 3. 批次大小 │                                              │
│   │ • 尝试不同  │                                              │
│   │ • 平衡显存  │                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ 4. 数据增强 │                                              │
│   │ • 调整参数  │                                              │
│   │ • 验证效果  │                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ 5. 模型结构 │                                              │
│   │ • 尝试不同  │                                              │
│   │ • n/s/m/l/x │                                              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ 6. 集成优化 │                                              │
│   │ • 组合最优  │                                              │
│   │ • 最终训练  │                                              │
│   └─────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 学习率调优

```python
# 学习率扫描
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

# 执行学习率扫描
results = model.train(
    data='configs/dataset.yaml',
    epochs=10,
    imgsz=640,
    batch=16,
    optimizer='Adam',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    device=0,
)
```

**学习率推荐范围**：

| 优化器 | 推荐lr0范围 | 说明 |
|--------|-------------|------|
| SGD | 0.001 - 0.01 | 传统优化器 |
| Adam | 0.0001 - 0.001 | 自适应学习率 |
| AdamW | 0.0001 - 0.001 | 权重衰减正则化 |

### 7.3 批次大小调优

| 批次大小 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| 8 | 显存占用小 | 训练不稳定 | 显存受限 |
| 16 | 平衡选择 | - | 通用场景 |
| 32 | 训练稳定 | 显存占用大 | 充足显存 |
| 64 | 收敛快 | 需要大显存 | 高端GPU |

### 7.4 数据增强调优

```yaml
# 增强策略对比实验
# 配置1: 基础增强
augment_basic:
  fliplr: 0.5
  mosaic: 1.0
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4

# 配置2: 强增强
augment_strong:
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.2
  degrees: 15.0
  translate: 0.15
  scale: 0.6
  hsv_h: 0.02
  hsv_s: 0.8
  hsv_v: 0.5

# 配置3: 保守增强
augment_conservative:
  fliplr: 0.5
  mosaic: 0.8
  hsv_h: 0.01
  hsv_s: 0.5
  hsv_v: 0.3
```

---

## 8. 模型评估

### 8.1 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| Precision | TP / (TP + FP) | 精确率 |
| Recall | TP / (TP + FN) | 召回率 |
| F1-Score | 2 * P * R / (P + R) | F1分数 |
| AP | ∫ P(r) dr | 单类别平均精度 |
| mAP | Σ AP / N | 所有类别平均精度 |
| mAP@0.5 | IoU阈值0.5的mAP | 常用指标 |
| mAP@0.5:0.95 | IoU阈值0.5-0.95平均 | 严格指标 |

### 8.2 评估脚本

```python
from ultralytics import YOLO

# 加载最佳模型
model = YOLO('runs/train/exp/weights/best.pt')

# 在测试集上评估
results = model.val(
    data='configs/dataset.yaml',
    split='test',
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.45,
    device=0,
    verbose=True,
)

# 打印评估结果
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")

# 各类别精度
for i, name in enumerate(results.names):
    print(f"{name}: AP@0.5={results.box.ap50[i]:.4f}")
```

### 8.3 混淆矩阵

```
                        预测类别
                 N   MS   SS   MC   D   MM
              ┌─────────────────────────────┐
         N    │ 43   1    0    0    1   0   │  正常件
              ├─────────────────────────────┤
         MS   │  1   34   2    0    0   1   │  轻微划痕
              ├─────────────────────────────┤
实      SS    │  0    1   35   1    0   1   │  严重划痕
际           ├─────────────────────────────┤
类      MC    │  0    0    1   36   1   0   │  缺角
别           ├─────────────────────────────┤
         D    │  1    0    0    1   31   0   │  变形
              ├─────────────────────────────┤
         MM   │  0    1    1    0    0   31  │  混料
              └─────────────────────────────┘

N=Normal, MS=Minor Scratch, SS=Severe Scratch
MC=Missing Corner, D=Deformation, MM=Mixed Material
```

### 8.4 评估报告

```json
{
  "evaluation_results": {
    "dataset": "test",
    "num_images": 225,
    "num_instances": 225,
    "mAP50": 0.968,
    "mAP50-95": 0.892,
    "precision": 0.954,
    "recall": 0.947,
    "f1_score": 0.950,
    "per_class": {
      "Normal": {"AP50": 0.975, "AP50-95": 0.912},
      "Minor_Scratch": {"AP50": 0.942, "AP50-95": 0.867},
      "Severe_Scratch": {"AP50": 0.961, "AP50-95": 0.895},
      "Missing_Corner": {"AP50": 0.968, "AP50-95": 0.901},
      "Deformation": {"AP50": 0.954, "AP50-95": 0.878},
      "Mixed_Material": {"AP50": 0.959, "AP50-95": 0.889}
    }
  }
}
```

---

## 9. 模型导出

### 9.1 导出格式

| 格式 | 后缀 | 推理速度 | 兼容性 | 说明 |
|------|------|----------|--------|------|
| PyTorch | .pt | 中 | PyTorch | 训练格式 |
| ONNX | .onnx | 快 | 通用 | 跨平台推理 |
| TensorRT | .engine | 最快 | NVIDIA | GPU极致优化 |
| OpenVINO | - | 快 | Intel | CPU优化 |
| CoreML | .mlmodel | 中 | Apple | macOS/iOS |
| TFLite | .tflite | 中 | 移动端 | Android/嵌入式 |

### 9.2 导出脚本

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/train/exp/weights/best.pt')

# 导出为ONNX格式
model.export(format='onnx', imgsz=640, dynamic=True, simplify=True)

# 导出为TensorRT格式（需要NVIDIA GPU）
model.export(format='engine', imgsz=640, half=True, device=0)

# 导出为OpenVINO格式（需要Intel CPU）
model.export(format='openvino', imgsz=640)
```

### 9.3 导出配置

```python
# export.py
from ultralytics import YOLO
import argparse

def export_model(model_path, format='onnx', **kwargs):
    """
    导出模型
    
    Args:
        model_path: 模型路径
        format: 导出格式
        **kwargs: 额外参数
    """
    model = YOLO(model_path)
    
    export_params = {
        'format': format,
        'imgsz': kwargs.get('imgsz', 640),
        'dynamic': kwargs.get('dynamic', True),
        'simplify': kwargs.get('simplify', True),
        'half': kwargs.get('half', False),
        'device': kwargs.get('device', 0),
    }
    
    model.export(**export_params)
    print(f"模型已导出为{format}格式")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'engine', 'openvino', 'coreml'])
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--half', action='store_true')
    
    args = parser.parse_args()
    export_model(args.model, args.format, **vars(args))
```

---

## 10. 训练问题排查

### 10.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 损失不下降 | 学习率过大/过小 | 调整学习率 |
| 过拟合 | 数据量不足/模型过大 | 增加数据/减小模型 |
| 欠拟合 | 模型容量不足 | 使用更大模型 |
| 显存不足 | 批次太大/图像太大 | 减小批次/图像尺寸 |
| 训练震荡 | 学习率不稳定 | 增加预热/降低学习率 |
| mAP低 | 数据质量差/增强不足 | 改进数据/调整增强 |

### 10.2 调试技巧

```python
# 1. 检查数据加载
from ultralytics.data import YOLODataset

dataset = YOLODataset(
    img_path='data/train/images',
    data='configs/dataset.yaml',
    augment=False
)

# 可视化几个样本
for i in range(5):
    dataset.plot_image(i)

# 2. 验证模型结构
model = YOLO('yolov8s.pt')
model.info()

# 3. 小数据集快速测试
results = model.train(
    data='configs/dataset.yaml',
    epochs=5,
    batch=4,
    imgsz=320,
    device='cpu'
)
```

---

## 附录A：训练检查清单

- [ ] 环境配置正确
- [ ] GPU可用且CUDA正常
- [ ] 数据集格式正确
- [ ] 配置文件路径正确
- [ ] 预训练模型已下载
- [ ] 磁盘空间充足
- [ ] 训练参数合理

## 附录B：参考资源

1. Ultralytics官方文档: https://docs.ultralytics.com
2. YOLOv8 GitHub: https://github.com/ultralytics/ultralytics
3. PyTorch文档: https://pytorch.org/docs

---

**文档版本历史**

| 版本 | 日期 | 修改内容 | 作者 |
|------|------|----------|------|
| V1.0 | 2026-05-27 | 初始版本 | 项目组 |