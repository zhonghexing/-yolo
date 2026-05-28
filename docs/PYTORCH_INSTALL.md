# PyTorch 安装指南（RTX 5070）

## 问题背景

RTX 5070 使用 CUDA 架构 sm_120，需要 CUDA 13.2。PyTorch 稳定版 (cu126) 不支持此架构。

## 正确安装步骤

### 1. 安装 PyTorch nightly（支持 CUDA 13.2）

```bash
conda activate yolo_screw
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
```

**注意**：文件约 1.9GB，下载可能不稳定。如果失败，重试即可。

### 2. 安装其他依赖

```bash
pip install -r requirements.txt
```

requirements.txt 已注释掉 torch 相关行，不会触发重复下载。

## 防止重复下载的配置

### 环境级 pip 配置（已创建）

位置：`D:/Anaconda3/envs/yolo_screw/pip.conf`

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn

# PyTorch CUDA 版本从官方源下载
extra-index-url =
    https://download.pytorch.org/whl/cu132
    https://download.pytorch.org/whl/nightly/cu132
```

### 为什么需要这个配置？

- 全局 pip 配置使用清华镜像源（加速国内下载）
- 但清华源只有 CPU 版本的 PyTorch
- 没有 `extra-index-url` 时，pip 会从清华源下载 CPU 版本，导致：
  - 重复下载 2GB+ 文件
  - 安装后无法使用 GPU

## 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

预期输出：
```
PyTorch 2.x.x+cu132
CUDA available: True
GPU: NVIDIA GeForce RTX 5070
```

## 常见问题

### Q: 为什么不用 conda 安装 PyTorch？

A: conda 的 pytorch 频道更新较慢，nightly 版本需要从 pip 安装。

### Q: 下载太慢怎么办？

A: 使用代理或多次重试。PyTorch 官方源在国内有 CDN，通常比 conda 快。

### Q: 如何确认没有重复下载？

A: 运行 `pip list | grep torch`，如果显示 `+cu132` 后缀，说明已正确安装。
