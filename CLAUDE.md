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
- **Flask + SocketIO** Web 远程监控
- **pyttsx3** 离线语音播报

## 目录结构
```
D:/yolo/
├── app.py              # PyQt5 桌面应用（主入口，已集成反馈机制+计时器+统计对话框）
├── inference.py        # 推理引擎（ScrewDefectDetector）
├── train.py            # 训练脚本（基于 v1 最佳参数，yolov8s）
├── evaluate.py         # 模型评估（混淆矩阵、分类报告）
├── demo.py             # 比赛演示（20样本/180秒计时）
├── export_model.py     # 模型导出（ONNX/TensorRT）
├── feedback.py         # 反馈模块（语音/视觉/日志，已接入app.py）
├── db.py               # SQLite 数据库（检测记录存储、历史查询、删除）
├── web_dashboard.py    # Web 远程监控面板（Flask+SocketIO，含反馈活动面板）
├── visualization.py    # 可视化报告
├── constants.py        # 常量（类别名、颜色，6种缺陷各分配独立颜色）
├── compare_models.py   # 模型对比工具
├── check_gpu.py        # GPU 状态检查
├── monitor_and_test.py # 训练监控 + 自动测试
├── run_app.py          # 应用启动器（环境检查）
├── build_desktop.spec  # PyInstaller 打包配置
├── app_icon.ico        # 应用图标
├── CLAUDE.md / README.md
├── requirements.txt    # 依赖列表
├── environment.yml     # Conda 环境配置
├── yolov8s.pt          # 预训练权重 (22MB, 训练基准)
├── yolov8n.pt          # 预训练权重 (6.3MB, 快速调试/备用)
├── 基于AI视觉的工业零件缺陷检测方案(7)_fixed.pptx  # 最新竞赛答辩 PPT
├── 基于AI视觉的行业应用创新赛-赛项说明.pdf  # 赛项规则
├── 基于AI视觉的工业零件缺陷检测方案(5).pdf  # 参考文档
├── 参赛作品报告_优化版v4.docx              # 最新参赛作品报告
├── 答辩问答准备.docx                       # 答辩问答文档（25 个问题+答案）
├── datasets/
│   └── neu_det/        # NEU-DET 钢材缺陷数据集
│       ├── data.yaml   # 数据集配置
│       ├── train/images/  # 训练集
│       ├── val/images/    # 验证集
│       └── test/images/   # 测试集 (180张)
├── deploy/             # 部署包源文件
│   ├── app.py          # GUI 主程序
│   ├── inference.py    # 推理引擎
│   ├── best.pt         # 部署模型 (YOLOv8s)
│   ├── yolov8n.pt      # 备用模型
│   ├── README.md       # 部署文档
│   ├── test_images/    # 测试图片
│   └── *.bat           # 一键启动脚本
├── defect_records/     # 缺陷记录
│   ├── snapshots/      # 拍照统计保存的标注图
│   └── annotated/      # 图片检测保存的标注图
├── data/               # 应用数据库
├── runs/
│   ├── train/
│   │   └── screw_defect-11/  # v1: yolov8s (mAP50=0.761) ★最佳模型
│   ├── detect/         # 验证结果
│   ├── eval/           # 评估报告
│   └── demo/           # 演示结果
├── tools/
│   ├── data_augment.py # 数据增强
│   ├── augment_weak_classes.py # 弱类别增强
│   ├── labelme_to_yolo.py # 标注格式转换
│   ├── split_dataset.py # 数据集划分
│   └── generate_sample_data.py # 生成样本
├── docs/
│   └── 手机摄像头使用指南.md  # 手机视频流、图片保存、Web监控
└── 决赛测试/           # 决赛测试图片目录
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

# 训练（基于 v1 最佳参数）
python train.py                          # 默认参数（yolov8s, 150 epochs）
python train.py --epochs 50              # 自定义轮数
python train.py --resume runs/train/screw_defect-11/weights/last.pt  # 恢复训练

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

# Web 监控
# 菜单 → 监控 → 启动 Web 监控（或按 Ctrl+W）
# 访问 http://localhost:5000
```

## 训练配置（当前 v1 最佳参数）
| 参数 | 值 | 说明 |
|------|------|------|
| model | yolov8s.pt | 预训练模型 |
| imgsz | 800 | 输入尺寸 |
| batch | 32 | 批量大小 |
| epochs | 150 | 最大轮数 |
| patience | 50 | 早停耐心 |
| optimizer | AdamW | 优化器 |
| lr0 | 0.001 | 初始学习率 |
| cache | disk | 磁盘缓存（省 RAM） |
| workers | 2 | 数据加载线程 |
| mixup | 0.1 | MixUp 增强 |
| erasing | 0.4 | 随机擦除 |

## 模型版本对比
| 版本 | 模型 | 大小 | mAP@0.5 | mAP@0.5:0.95 | 部署 |
|------|------|------|---------|---------------|------|
| v1 (screw_defect-11) | YOLOv8s | 21.5 MB | 0.761 | 0.421 | ★ 当前部署 |
| v2 (screw_defect_v2) | YOLOv8m | 197.9 MB | 0.733 | 0.397 | — |
| v3 (v3_optimized) | YOLOv8l | 49.7 MB | 0.675 | 0.405 | — |

## v1 评估结果 (screw_defect-11) ★ 最佳模型
| 指标 | 值 |
|------|------|
| mAP@0.5 | 0.761 |
| mAP@0.5:0.95 | 0.421 |
| Precision | 0.764 |
| Recall | 0.697 |
| 推理速度（GPU） | 12.6ms/张（RTX 5070） |
| 推理速度（CPU） | ~100ms/张（i5-1240P） |

> 注：v2/v3 因模型更大导致过拟合，效果反而不如 v1。当前训练脚本已统一为 v1 参数。

## 比赛准确率测试结果（180 张测试集）
| 指标 | 值 |
|------|------|
| 图片级分类准确率 | **98.9%**（178/180） |
| 类型错误 | 2 张（inclusion→scratches, pitted_surface→patches） |
| 漏检 | 0 张 |
| 最佳置信度阈值 | 0.10~0.20（均 98.9%） |
| 默认阈值 0.25 准确率 | 98.3%（177/180） |

### 各类别准确率
| 类别 | 准确率 | 备注 |
|------|--------|------|
| crazing（龟裂） | 100% (30/30) | |
| inclusion（夹杂） | 96.7% (29/30) | 1 张误判为 scratches |
| patches（斑块） | 100% (30/30) | |
| pitted_surface（麻点） | 96.7% (29/30) | 1 张误判为 patches |
| rolled-in_scale（氧化皮） | 100% (30/30) | |
| scratches（划痕） | 100% (30/30) | |

### 比赛模拟（1 万次随机抽 20 张）
| 指标 | 值 |
|------|------|
| 平均准确率 | 98.9% |
| 20/20 完美概率 | 78.6% |
| ≥19/20 概率 | 98.8% |
| ≥18/20 概率 | 100% |
| 最差情况 | 18/20 |

> 比赛时建议置信度阈值设为 **0.15**，兼顾检出率和准确率。

## 部署
- **训练机**: RTX 5070 (12GB VRAM) + Windows 11
- **部署机**: i5-1240P + 16GB RAM（CPU 推理）
- **部署包**: `deploy_package.zip` (25.62MB)，解压后双击 `启动检测系统.bat` 一键安装依赖并启动
- **部署包内容**: app.py + inference.py + feedback.py + db.py + web_dashboard.py + constants.py + best.pt + 启动脚本
- **模型**: 当前部署使用 v1 (screw_defect-11, YOLOv8s, mAP50=0.768)
- 详见 `deploy/README.md`

## 注意事项
- **RAM 有限**: `cache=True` 会爆内存，必须用 `cache="disk"`
- **RTX 5070**: sm_120 架构，需要 PyTorch 2.11+（nightly）
- **类别名中英文**: `constants.py` 中 `CLASS_NAMES`（英文列表）和 `CLASS_NAMES_CN`（中文字典，不是列表）
- **inference.py 的 `detect_single()`** 接受文件路径，视频帧需直接调用 `model(frame)`
- **app.py** 支持深色/浅色主题切换，图片/摄像头/视频三种检测模式
- **弱类别增强**: crazing 和 rolled-in_scale 已通过 `augment_weak_classes.py` 增强到 500 张
- **模型备用**: 若 best.pt 不存在，自动回退到 yolov8n.pt
- **手机摄像头**: 支持 DroidCam、IP Webcam 等，详见 `docs/手机摄像头使用指南.md`
- **图片自动保存**: 拍照统计保存到 `defect_records/snapshots/`，图片检测保存到 `defect_records/annotated/`
- **反馈机制**: app.py 已集成 FeedbackManager，语音播报默认开启（enable_voice=True），视频流模式仅在缺陷状态变化时播报（防刷屏）
- **Web 监控**: Flask+SocketIO 实现，含反馈活动面板、检测历史（支持删除）、可视化分析（24h/7天/30天筛选），启动后访问 http://localhost:5000
- **数据库**: db.py 使用 SQLite，路径为 `data/detections.db`，支持检测记录的增删查和统计
- **比赛准确率定义**: 图片级分类准确率（缺陷类型是否正确），不是 mAP
- **比赛时置信度阈值**: 建议设为 0.15，默认 0.25 会漏检 1 张
- **mAP vs 分类准确率**: mAP 要求框位置+类别同时正确；分类准确率只要类别对即可
- **部署包同步**: 修改主目录代码后需手动复制到 deploy/ 目录并重新打包
- **批量检测性能**: `_flush_batch_queue()` 限制 UI 更新频率（500ms），`_update_defect_distribution()` 使用缓存计数器避免重复遍历
- **批量标注图生成**: `_batch_save_worker()` 在后台线程生成标注图并保存，确保 Web 面板能显示缺陷图
- **计时器模式**: 支持 180 秒计时，包含暂停/完成按钮，完成按钮可提前结束并弹出统计对话框
- **统计对话框**: `StatsDialog` 类，卡片式布局展示检测总数/合格/不合格/合格率/缺陷分布
- **Web 图片预览**: 模态框使用 `width:90vw; height:85vh` 强制拉伸图片，确保小图也能清晰显示
- **缺陷类型颜色**: 6种缺陷各分配独立颜色（龟裂=青色、夹杂=橙色、斑块=绿色、麻点=紫色、氧化皮=红橙色、划痕=天蓝色）

## 答辩准备
- **PPT 已更新页**: 第7页（样本说明）、第8页（数据采集）、第9页（训练参数）、第10页（模型性能）、第13页（反馈机制）
- **答辩问答文档**: `答辩问答准备.docx`，6 大类 25 个问题+答案
- **核心话术**: "180 张测试图 98.9% 准确率，20 张检测仅需 0.25 秒，远超 95%/180 秒的比赛要求"
- **mAP 较低的解释**: mAP 是检测指标（框位置+类别），分类准确率是图片级指标（只看类别），两者衡量维度不同
- **PPT 注意**: 合格件无绿色框标注，直接语音播报"检测合格"；缺陷件红色框定位+语音播报类型

## 代码风格
- 中文注释和文档
- 函数/变量用英文命名
- PyQt5 样式表定义在文件顶部常量中
- 检测结果用 dataclass（`SingleDetection`、`ImageDetectionResult`）
