"""
训练监控与自动测试脚本
Monitor training progress and auto-run evaluation after training completes

使用方法：
    python monitor_and_test.py                    # 监控默认训练目录
    python monitor_and_test.py --train-dir runs/train/screw_defect
    python monitor_and_test.py --interval 60      # 每60秒检查一次
"""

import time
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.absolute()


def find_best_model(train_dir: Path) -> Path | None:
    """在训练目录中查找 best.pt"""
    best = train_dir / "weights" / "best.pt"
    if best.exists():
        return best
    return None


def find_latest_train_dir() -> Path | None:
    """查找最新的训练目录"""
    root = get_project_root()
    runs_dir = root / "runs" / "train"

    if not runs_dir.exists():
        return None

    train_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return train_dirs[0] if train_dirs else None


def check_training_done(train_dir: Path) -> str:
    """
    检查训练是否完成

    返回：
        "success" - 训练完成且模型存在
        "running" - 训练仍在进行
        "failed"  - 训练失败或目录不存在
    """
    best_model = find_best_model(train_dir)
    last_model = train_dir / "weights" / "last.pt"

    # 检查 results.csv 来判断训练状态
    results_csv = train_dir / "results.csv"
    if results_csv.exists():
        try:
            with open(results_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # 如果 results.csv 有数据行，检查最后是否完整
                    last_line = lines[-1].strip()
                    if last_line and best_model.exists():
                        return "success"
        except Exception:
            pass

    # 如果 best.pt 存在，认为训练完成
    if best_model.exists():
        return "success"

    # 检查训练进程是否还在运行（通过检查 last.pt 的修改时间）
    if last_model.exists():
        mtime = last_model.stat().st_mtime
        age_seconds = time.time() - mtime
        if age_seconds < 300:  # 5 分钟内有更新
            return "running"

    return "running"


def get_training_progress(train_dir: Path) -> str | None:
    """从 results.csv 获取训练进度"""
    results_csv = train_dir / "results.csv"
    if not results_csv.exists():
        return None

    try:
        with open(results_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                if last_line:
                    # 返回最后一行的 epoch 信息
                    parts = last_line.split(',')
                    if parts:
                        return f"epoch {parts[0].strip()}"
    except Exception:
        pass
    return None


def run_evaluation(model_path: Path):
    """运行模型评估"""
    root = get_project_root()
    eval_script = root / "evaluate.py"

    if not eval_script.exists():
        print(f"[错误] 评估脚本不存在: {eval_script}")
        return

    print("\n" + "=" * 60)
    print("开始运行模型评估...")
    print("=" * 60)

    cmd = [sys.executable, str(eval_script), "--model", str(model_path)]
    print(f"[信息] 执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root))

    if result.returncode == 0:
        print("\n[成功] 模型评估完成!")
    else:
        print(f"\n[警告] 评估退出码: {result.returncode}")


def run_demo(model_path: Path):
    """运行演示测试"""
    root = get_project_root()
    demo_script = root / "demo.py"

    if not demo_script.exists():
        print(f"[提示] 演示脚本不存在: {demo_script}")
        return

    print("\n" + "=" * 60)
    print("运行演示测试...")
    print("=" * 60)

    cmd = [sys.executable, str(demo_script), "--model", str(model_path)]
    print(f"[信息] 执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root))

    if result.returncode == 0:
        print("\n[成功] 演示测试完成!")
    else:
        print(f"\n[警告] 演示退出码: {result.returncode}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练监控与自动测试")
    parser.add_argument(
        "--train-dir", type=str, default="",
        help="训练输出目录 (默认: 自动查找最新的)"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="检查间隔/秒 (默认: 60)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="仅运行评估，不运行演示"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = get_project_root()

    print("=" * 60)
    print("训练监控与自动测试")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 确定训练目录
    if args.train_dir:
        train_dir = Path(args.train_dir)
        if not train_dir.is_absolute():
            train_dir = root / train_dir
    else:
        train_dir = find_latest_train_dir()
        if train_dir is None:
            print("[错误] 未找到训练目录，请使用 --train-dir 参数指定")
            sys.exit(1)

    print(f"[信息] 监控训练目录: {train_dir}")
    print(f"[信息] 检查间隔: {args.interval} 秒")
    print("=" * 60)

    # 监控循环
    last_progress = None
    while True:
        status = check_training_done(train_dir)

        if status == "success":
            best_model = find_best_model(train_dir)
            print(f"\n[监控] 训练完成! 最佳模型: {best_model}")
            break
        elif status == "failed":
            print("\n[监控] 训练目录状态异常")
            break

        # 显示进度
        progress = get_training_progress(train_dir)
        if progress and progress != last_progress:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 训练进行中: {progress}")
            last_progress = progress
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待中...")

        time.sleep(args.interval)

    # 训练完成后运行测试
    best_model = find_best_model(train_dir)
    if best_model:
        print("\n[监控] 等待 5 秒后开始测试...")
        time.sleep(5)

        # 运行评估
        run_evaluation(best_model)

        # 运行演示（可选）
        if not args.eval_only:
            run_demo(best_model)

        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
    else:
        print("[错误] 未找到训练好的模型")


if __name__ == "__main__":
    main()
