#!/usr/bin/env python3
"""
螺丝缺陷检测数据集 - 模拟数据生成器
用于开发测试阶段生成带模拟缺陷的图片和YOLO格式标注

使用方法:
    python tools/generate_sample_data.py --num 100 --output datasets/screws
"""

import os
import random
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("请安装Pillow: pip install Pillow")
    exit(1)


# 类别配置
CLASSES = [
    "normal",
    "minor_scratch",
    "severe_scratch",
    "missing_corner",
    "deformation",
    "mixed_material",
]

# 螺丝基础参数
SCREW_CONFIG = {
    "head_radius": (30, 50),  # 螺丝头半径范围
    "shaft_length": (80, 150),  # 螺杆长度范围
    "shaft_width": (15, 25),  # 螺杆宽度范围
    "colors": {
        "head": [(180, 180, 180), (160, 160, 170), (200, 200, 200)],
        "shaft": [(150, 150, 160), (140, 140, 150), (170, 170, 180)],
    },
}


def draw_screw_base(draw, cx, cy, config):
    """绘制螺丝基础形状"""
    head_r = random.randint(*config["head_radius"])
    shaft_l = random.randint(*config["shaft_length"])
    shaft_w = random.randint(*config["shaft_width"])

    head_color = random.choice(config["colors"]["head"])
    shaft_color = random.choice(config["colors"]["shaft"])

    # 绘制螺丝头（六边形近似为圆形）
    draw.ellipse(
        [cx - head_r, cy - head_r, cx + head_r, cy + head_r],
        fill=head_color,
        outline=(100, 100, 100),
    )

    # 绘制十字槽
    slot_len = head_r * 0.6
    draw.line(
        [(cx - slot_len, cy), (cx + slot_len, cy)],
        fill=(80, 80, 80),
        width=3,
    )
    draw.line(
        [(cx, cy - slot_len), (cx, cy + slot_len)],
        fill=(80, 80, 80),
        width=3,
    )

    # 绘制螺杆
    shaft_top = cy + head_r
    draw.rectangle(
        [cx - shaft_w // 2, shaft_top, cx + shaft_w // 2, shaft_top + shaft_l],
        fill=shaft_color,
        outline=(100, 100, 100),
    )

    # 绘制螺纹
    thread_spacing = 8
    for y in range(shaft_top + 5, shaft_top + shaft_l - 5, thread_spacing):
        draw.line(
            [(cx - shaft_w // 2 - 3, y), (cx + shaft_w // 2 + 3, y)],
            fill=(120, 120, 130),
            width=2,
        )

    # 返回边界框
    bbox = {
        "x_min": cx - head_r,
        "y_min": cy - head_r,
        "x_max": cx + shaft_w // 2,
        "y_max": shaft_top + shaft_l,
    }
    return bbox, head_r, shaft_l, shaft_w, shaft_top


def draw_normal_defect(draw, cx, cy, head_r, shaft_top, shaft_l, shaft_w):
    """正常件 - 无缺陷"""
    return None


def draw_minor_scratch(draw, cx, cy, head_r, shaft_top, shaft_l, shaft_w):
    """轻微划痕 - 细线"""
    scratch_x = cx + random.randint(-head_r // 2, head_r // 2)
    scratch_y = shaft_top + random.randint(10, shaft_l - 20)
    length = random.randint(15, 30)
    draw.line(
        [(scratch_x, scratch_y), (scratch_x + length, scratch_y + length // 2)],
        fill=(100, 100, 100),
        width=1,
    )
    # 返回缺陷区域
    return {
        "x_min": scratch_x,
        "y_min": scratch_y,
        "x_max": scratch_x + length,
        "y_max": scratch_y + length // 2 + 5,
    }


def draw_severe_scratch(draw, cx, cy, head_r, shaft_top, shaft_l, shaft_w):
    """严重划痕 - 多条粗线"""
    defects = []
    for _ in range(random.randint(3, 5)):
        sx = cx + random.randint(-head_r, head_r // 2)
        sy = shaft_top + random.randint(5, shaft_l - 15)
        length = random.randint(20, 40)
        draw.line(
            [(sx, sy), (sx + length, sy + random.randint(-10, 10))],
            fill=(60, 60, 60),
            width=random.randint(2, 4),
        )
        defects.append({"x_min": sx, "y_min": sy, "x_max": sx + length, "y_max": sy + 15})

    # 合并缺陷区域
    if defects:
        return {
            "x_min": min(d["x_min"] for d in defects),
            "y_min": min(d["y_min"] for d in defects),
            "x_max": max(d["x_max"] for d in defects),
            "y_max": max(d["y_max"] for d in defects),
        }
    return None


def draw_missing_corner(draw, cx, cy, head_r, shaft_top, shaft_l, shaft_w):
    """缺角 - 遮盖一个角落"""
    # 随机选择一个角落
    corner = random.choice(["tl", "tr", "bl", "br"])
    size = random.randint(15, 25)

    if corner == "tl":
        points = [(cx - head_r, cy - head_r), (cx - head_r + size, cy - head_r),
                  (cx - head_r, cy - head_r + size)]
    elif corner == "tr":
        points = [(cx + head_r, cy - head_r), (cx + head_r - size, cy - head_r),
                  (cx + head_r, cy - head_r + size)]
    elif corner == "bl":
        points = [(cx - head_r, cy + head_r), (cx - head_r + size, cy + head_r),
                  (cx - head_r, cy + head_r - size)]
    else:
        points = [(cx + head_r, cy + head_r), (cx + head_r - size, cy + head_r),
                  (cx + head_r, cy + head_r - size)]

    # 用背景色填充（模拟缺角）
    bg_color = (240, 240, 240)
    draw.polygon(points, fill=bg_color)

    return {
        "x_min": min(p[0] for p in points),
        "y_min": min(p[1] for p in points),
        "x_max": max(p[0] for p in points),
        "y_max": max(p[1] for p in points),
    }


def draw_deformation(draw, cx, cy, head_r, shaft_top, shaft_l, shaft_w):
    """变形 - 弯曲的螺杆"""
    # 绘制弯曲效果（用多段线模拟）
    bend_offset = random.randint(10, 20)
    points = []
    for i in range(0, shaft_l, 5):
        x_offset = int(bend_offset * (i / shaft_l) * (1 - i / shaft_l) * 4)
        if random.random() > 0.5:
            x_offset = -x_offset
        points.append((cx + x_offset, shaft_top + i))

    if len(points) > 1:
        draw.line(points, fill=(140, 140, 150), width=shaft_w)

    return {
        "x_min": cx - bend_offset - shaft_w,
        "y_min": shaft_top,
        "x_max": cx + bend_offset + shaft_w,
        "y_max": shaft_top + shaft_l,
    }


def draw_mixed_material(draw, cx, cy, head_r, shaft_top, shaft_l, shaft_w):
    """混料 - 不同颜色区域"""
    # 在螺杆上绘制不同颜色的区域
    patch_y = shaft_top + random.randint(10, shaft_l - 30)
    patch_h = random.randint(20, 40)
    patch_color = random.choice([(200, 180, 150), (180, 200, 180), (200, 200, 180)])

    draw.rectangle(
        [cx - shaft_w // 2 - 2, patch_y, cx + shaft_w // 2 + 2, patch_y + patch_h],
        fill=patch_color,
        outline=(100, 100, 100),
    )

    return {
        "x_min": cx - shaft_w // 2 - 2,
        "y_min": patch_y,
        "x_max": cx + shaft_w // 2 + 2,
        "y_max": patch_y + patch_h,
    }


# 缺陷绘制函数映射
DEFECT_DRAWERS = {
    0: draw_normal_defect,
    1: draw_minor_scratch,
    2: draw_severe_scratch,
    3: draw_missing_corner,
    4: draw_deformation,
    5: draw_mixed_material,
}


def generate_screw_image(
    img_size: tuple = (640, 640),
    class_id: int = None,
) -> tuple:
    """生成单张螺丝图片和标注"""

    # 随机选择类别
    if class_id is None:
        class_id = random.randint(0, len(CLASSES) - 1)

    # 创建图片
    bg_color = (random.randint(230, 250), random.randint(230, 250), random.randint(230, 250))
    img = Image.new("RGB", img_size, bg_color)
    draw = ImageDraw.Draw(img)

    # 随机螺丝位置
    margin = 80
    cx = random.randint(margin, img_size[0] - margin)
    cy = random.randint(margin, img_size[1] // 2)

    # 绘制螺丝
    bbox, head_r, shaft_l, shaft_w, shaft_top = draw_screw_base(
        draw, cx, cy, SCREW_CONFIG
    )

    # 绘制缺陷
    defect_func = DEFECT_DRAWERS[class_id]
    defect_bbox = defect_func(draw, cx, cy, head_r, shaft_top, shaft_l, shaft_w)

    # 添加轻微噪声
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # 计算YOLO格式标注（归一化）
    img_w, img_h = img_size

    # 整个螺丝的边界框
    screw_bbox = {
        "x_min": max(0, bbox["x_min"]),
        "y_min": max(0, bbox["y_min"]),
        "x_max": min(img_w, bbox["x_max"]),
        "y_max": min(img_h, bbox["y_max"]),
    }

    # 转换为YOLO格式 (class_id, x_center, y_center, width, height)
    x_center = (screw_bbox["x_min"] + screw_bbox["x_max"]) / 2 / img_w
    y_center = (screw_bbox["y_min"] + screw_bbox["y_max"]) / 2 / img_h
    width = (screw_bbox["x_max"] - screw_bbox["x_min"]) / img_w
    height = (screw_bbox["y_max"] - screw_bbox["y_min"]) / img_h

    label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return img, label, CLASSES[class_id]


def generate_dataset(
    output_dir: str,
    num_samples: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    img_size: tuple = (640, 640),
):
    """生成完整数据集"""

    output_path = Path(output_dir)

    # 计算各集合数量
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    splits = [
        ("train", num_train),
        ("val", num_val),
        ("test", num_test),
    ]

    # 确保每个类别都有样本
    samples_per_class = max(1, num_samples // (len(CLASSES) * 3))

    print(f"开始生成螺丝缺陷检测数据集...")
    print(f"总样本数: {num_samples}")
    print(f"训练集: {num_train}, 验证集: {num_val}, 测试集: {num_test}")
    print(f"图片尺寸: {img_size}")
    print("-" * 50)

    total_generated = 0

    for split_name, num_split in splits:
        img_dir = output_path / "images" / split_name
        label_dir = output_path / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n生成 {split_name} 集...")

        for i in range(num_split):
            # 确保类别分布均匀
            if i < num_split:
                class_id = i % len(CLASSES)
            else:
                class_id = random.randint(0, len(CLASSES) - 1)

            # 生成图片和标注
            img, label, class_name = generate_screw_image(img_size, class_id)

            # 保存文件
            filename = f"screw_{split_name}_{i:04d}_{class_name}"
            img_path = img_dir / f"{filename}.jpg"
            label_path = label_dir / f"{filename}.txt"

            img.save(img_path, quality=95)
            with open(label_path, "w") as f:
                f.write(label + "\n")

            total_generated += 1
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  已生成: {i + 1}/{num_split}")

    print("\n" + "=" * 50)
    print(f"数据集生成完成!")
    print(f"总计生成: {total_generated} 张图片")
    print(f"输出目录: {output_path.absolute()}")

    # 生成类别统计
    print("\n类别分布:")
    for split_name, _ in splits:
        label_dir = output_path / "labels" / split_name
        class_counts = {c: 0 for c in CLASSES}
        for label_file in label_dir.glob("*.txt"):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[CLASSES[class_id]] += 1
        print(f"\n  {split_name}:")
        for cls, count in class_counts.items():
            print(f"    {cls}: {count}")


def main():
    parser = argparse.ArgumentParser(description="生成螺丝缺陷检测模拟数据集")
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="生成样本总数 (默认: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/screws",
        help="输出目录 (默认: datasets/screws)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="图片尺寸 (默认: 640 640)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="训练集比例 (默认: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例 (默认: 0.2)",
    )

    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)

    # 生成数据集
    generate_dataset(
        output_dir=args.output,
        num_samples=args.num,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        img_size=tuple(args.img_size),
    )


if __name__ == "__main__":
    main()
