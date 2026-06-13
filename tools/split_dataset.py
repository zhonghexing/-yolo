#!/usr/bin/env python3
"""
钢材缺陷检测数据集 - 数据集划分工具
支持按比例划分训练集、验证集、测试集

使用方法:
    python tools/split_dataset.py --images all_images \
                                  --labels all_labels \
                                  --output datasets/screws \
                                  --train 0.7 --val 0.2 --test 0.1
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


def get_image_label_pairs(
    img_dir: str,
    label_dir: str,
) -> List[Tuple[Path, Path]]:
    """获取图片和标签文件对"""

    img_path = Path(img_dir)
    label_path = Path(label_dir)

    # 支持的图片格式
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # 获取所有图片文件
    img_files = [
        f for f in img_path.iterdir()
        if f.suffix.lower() in img_extensions
    ]

    # 匹配图片和标签
    pairs = []
    missing_labels = []

    for img_file in img_files:
        label_file = label_path / f"{img_file.stem}.txt"
        if label_file.exists():
            pairs.append((img_file, label_file))
        else:
            missing_labels.append(img_file.name)

    if missing_labels:
        print(f"警告: 找到 {len(missing_labels)} 个图片没有对应标签")
        if len(missing_labels) <= 10:
            for name in missing_labels:
                print(f"  - {name}")

    return pairs


def split_by_class(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """按类别分层划分数据集"""

    random.seed(seed)

    # 读取所有标签，按类别分组
    class_pairs = {}
    for img_file, label_file in pairs:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id not in class_pairs:
                        class_pairs[class_id] = []
                    class_pairs[class_id].append((img_file, label_file))
                    break  # 只看第一行确定主类别

    train_set = []
    val_set = []
    test_set = []

    # 对每个类别进行划分
    for class_id, class_pair_list in class_pairs.items():
        random.shuffle(class_pair_list)

        n = len(class_pair_list)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set.extend(class_pair_list[:n_train])
        val_set.extend(class_pair_list[n_train:n_train + n_val])
        test_set.extend(class_pair_list[n_train + n_val:])

    # 打乱顺序
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def split_random(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """随机划分数据集"""

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = pairs[:n_train]
    val_set = pairs[n_train:n_train + n_val]
    test_set = pairs[n_train + n_val:]

    return train_set, val_set, test_set


def copy_files(
    pairs: List[Tuple[Path, Path]],
    output_img_dir: str,
    output_label_dir: str,
    prefix: str = "",
):
    """复制文件到目标目录"""

    output_img_path = Path(output_img_dir)
    output_label_path = Path(output_label_dir)

    # 创建目录
    output_img_path.mkdir(parents=True, exist_ok=True)
    output_label_path.mkdir(parents=True, exist_ok=True)

    for i, (img_file, label_file) in enumerate(pairs):
        # 生成新文件名
        if prefix:
            new_name = f"{prefix}_{i:04d}{img_file.suffix}"
        else:
            new_name = img_file.name

        # 复制图片
        shutil.copy2(img_file, output_img_path / new_name)

        # 复制标签（更改扩展名）
        new_label_name = Path(new_name).stem + ".txt"
        shutil.copy2(label_file, output_label_path / new_label_name)

    return len(pairs)


def print_split_statistics(
    train_set: list,
    val_set: list,
    test_set: list,
):
    """打印划分统计信息"""

    total = len(train_set) + len(val_set) + len(test_set)

    print("\n数据集划分统计:")
    print("=" * 50)
    print(f"总计: {total} 个样本")
    print(f"训练集: {len(train_set)} ({len(train_set)/total*100:.1f}%)")
    print(f"验证集: {len(val_set)} ({len(val_set)/total*100:.1f}%)")
    print(f"测试集: {len(test_set)} ({len(test_set)/total*100:.1f}%)")

    # 统计各类别分布
    print("\n各类别分布:")
    print("-" * 50)

    for split_name, split_set in [("训练集", train_set), ("验证集", val_set), ("测试集", test_set)]:
        class_counts = {}
        for _, label_file in split_set:
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1

        print(f"\n{split_name}:")
        for class_id in sorted(class_counts.keys()):
            print(f"  类别 {class_id}: {class_counts[class_id]} 个")


def split_dataset(
    img_dir: str,
    label_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    stratified: bool = True,
    seed: int = 42,
    prefix: str = "",
):
    """划分数据集"""

    output_path = Path(output_dir)

    print(f"正在读取数据...")
    print(f"图片目录: {img_dir}")
    print(f"标签目录: {label_dir}")

    # 获取文件对
    pairs = get_image_label_pairs(img_dir, label_dir)

    if not pairs:
        print("错误: 找不到有效的图片-标签对")
        return

    print(f"找到 {len(pairs)} 个有效的图片-标签对")

    # 验证比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        print("错误: 比例之和必须为1.0")
        return

    # 划分数据集
    print(f"\n正在划分数据集 (方法: {'分层抽样' if stratified else '随机'})...")

    if stratified:
        train_set, val_set, test_set = split_by_class(
            pairs, train_ratio, val_ratio, test_ratio, seed
        )
    else:
        train_set, val_set, test_set = split_random(
            pairs, train_ratio, val_ratio, test_ratio, seed
        )

    # 打印统计信息
    print_split_statistics(train_set, val_set, test_set)

    # 复制文件
    print(f"\n正在复制文件...")

    splits = [
        ("train", train_set),
        ("val", val_set),
        ("test", test_set),
    ]

    for split_name, split_set in splits:
        if split_set:
            img_output = output_path / "images" / split_name
            label_output = output_path / "labels" / split_name

            count = copy_files(
                split_set,
                str(img_output),
                str(label_output),
                prefix=prefix,
            )
            print(f"  {split_name}: 复制 {count} 个样本到 {img_output}")

    print("\n" + "=" * 50)
    print(f"数据集划分完成!")
    print(f"输出目录: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="图片目录",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="标签目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="训练集比例 (默认: 0.7)",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.2,
        help="验证集比例 (默认: 0.2)",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="测试集比例 (默认: 0.1)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="使用分层抽样 (默认: True)",
    )
    parser.add_argument(
        "--no-stratified",
        action="store_false",
        dest="stratified",
        help="不使用分层抽样",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="输出文件名前缀",
    )

    args = parser.parse_args()

    split_dataset(
        img_dir=args.images,
        label_dir=args.labels,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        stratified=args.stratified,
        seed=args.seed,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
