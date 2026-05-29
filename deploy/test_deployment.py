"""
钢材缺陷检测系统 - 部署测试脚本
用于验证部署环境是否正常
"""

import sys
import os
from pathlib import Path


def check_python():
    """检查 Python 版本"""
    print("[检查] Python 版本...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  [错误] 需要 Python 3.8 或更高版本")
        return False
    print("  [通过] Python 版本符合要求")
    return True


def check_dependencies():
    """检查依赖包"""
    print("\n[检查] 依赖包...")
    required = [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
    ]

    missing = []
    for module, package in required:
        try:
            __import__(module)
            print(f"  [通过] {package}")
        except ImportError:
            print(f"  [缺失] {package}")
            missing.append(package)

    if missing:
        print(f"\n[错误] 缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install -r requirements_cpu.txt")
        return False

    print("  [通过] 所有依赖包已安装")
    return True


def check_model():
    """检查模型文件"""
    print("\n[检查] 模型文件...")
    model_path = Path("best.pt")

    if not model_path.exists():
        print("  [错误] 模型文件不存在: best.pt")
        return False

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  [通过] 模型文件存在 ({size_mb:.1f} MB)")
    return True


def check_model_loading():
    """检查模型加载"""
    print("\n[检查] 模型加载...")
    try:
        from ultralytics import YOLO
        model = YOLO("best.pt")
        print("  [通过] 模型加载成功")
        return True
    except Exception as e:
        print(f"  [错误] 模型加载失败: {e}")
        return False


def check_files():
    """检查必要文件"""
    print("\n[检查] 必要文件...")
    required_files = [
        "app.py",
        "inference.py",
        "constants.py",
        "run_app.py",
        "feedback.py",
        "visualization.py",
    ]

    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"  [通过] {file}")
        else:
            print(f"  [缺失] {file}")
            missing.append(file)

    if missing:
        print(f"\n[错误] 缺少文件: {', '.join(missing)}")
        return False

    print("  [通过] 所有必要文件存在")
    return True


def main():
    """主函数"""
    print("=" * 50)
    print("钢材缺陷检测系统 - 部署测试")
    print("=" * 50)

    checks = [
        check_python,
        check_dependencies,
        check_model,
        check_files,
        check_model_loading,
    ]

    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"  [错误] 检查失败: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    if all(results):
        print("[结果] 所有检查通过！系统已就绪。")
        print("\n可以运行以下命令启动系统:")
        print("  - 双击 '启动检测系统.bat'")
        print("  - 或运行: python app.py")
    else:
        print("[结果] 部分检查失败，请根据上述提示修复问题。")
    print("=" * 50)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
