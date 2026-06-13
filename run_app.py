"""
钢材缺陷检测系统启动脚本
Application Launcher with Environment Check

功能：
    - 检查 Python 环境
    - 检查依赖包
    - 检查模型文件
    - 启动主应用

使用方法：
    python run_app.py
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("  钢材缺陷检测系统 v1.0.0")
    print("  Screw Defect Detection System")
    print("=" * 60)
    print()


def check_python_version():
    """检查 Python 版本"""
    print("[检查] Python 版本...", end=" ")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"失败 (当前: {version.major}.{version.minor})")
        print("  错误: 需要 Python 3.8 或更高版本")
        return False
    print(f"通过 ({version.major}.{version.minor}.{version.micro})")
    return True


def check_package(package_name, import_name=None):
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_dependencies():
    """检查所有依赖包"""
    print("[检查] 依赖包...")

    dependencies = [
        ("PyQt5", "PyQt5"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
    ]

    missing = []

    for package_name, import_name in dependencies:
        print(f"  {package_name}...", end=" ")
        if check_package(package_name, import_name):
            print("OK")
        else:
            print("缺失")
            missing.append(package_name)

    if missing:
        print(f"\n[警告] 缺少以下依赖包: {', '.join(missing)}")
        print("请运行以下命令安装:")
        print(f"  pip install {' '.join(missing)}")

        # 尝试自动安装
        response = input("\n是否自动安装? (y/n): ").strip().lower()
        if response == 'y':
            print("\n正在安装依赖...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install"
                ] + missing)
                print("依赖安装完成!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"安装失败: {e}")
                return False
        else:
            return False

    return True


def check_model():
    """检查模型文件"""
    print("\n[检查] 模型文件...", end=" ")

    # 检查常见模型路径
    search_paths = [
        Path("runs/train"),
        Path("models"),
        Path("."),
    ]

    found_models = []

    for search_path in search_paths:
        if search_path.exists():
            models = list(search_path.glob("**/best.pt"))
            models.extend(search_path.glob("**/*.pt"))
            found_models.extend(models)

    if found_models:
        # 按修改时间排序
        found_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = found_models[0]
        print(f"找到 {len(found_models)} 个模型")
        print(f"  最新模型: {latest}")
        return True
    else:
        print("未找到训练模型")
        print("  将使用 YOLOv8n 预训练模型（未针对钢材缺陷微调）")
        return True  # 不阻止启动


def check_gpu():
    """检查 GPU 状态"""
    print("\n[检查] GPU 状态...", end=" ")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"可用")
            print(f"  设备: {gpu_name}")
            print(f"  显存: {gpu_mem:.1f} GB")
        else:
            print("不可用 (将使用 CPU)")
    except ImportError:
        print("未安装 PyTorch")


def main():
    """主启动函数"""
    print_banner()

    # 检查环境
    if not check_python_version():
        input("\n按 Enter 键退出...")
        sys.exit(1)

    if not check_dependencies():
        print("\n[错误] 缺少必要的依赖包，无法启动应用")
        input("\n按 Enter 键退出...")
        sys.exit(1)

    check_model()
    check_gpu()

    print("\n" + "=" * 60)
    print("[启动] 正在启动应用...")
    print("=" * 60)
    print()

    # 确保当前目录是项目根目录
    os.chdir(Path(__file__).parent)

    # 启动主应用
    try:
        from app import main as run_app
        run_app()
    except ImportError as e:
        print(f"[错误] 导入应用模块失败: {e}")
        print("请确保 app.py 文件存在且完整")
        input("\n按 Enter 键退出...")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 应用运行出错: {e}")
        import traceback
        traceback.print_exc()
        input("\n按 Enter 键退出...")
        sys.exit(1)


if __name__ == "__main__":
    main()
