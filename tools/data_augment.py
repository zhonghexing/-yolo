#!/usr/bin/env python3
"""
钢材缺陷检测数据集 - 数据增强脚本
支持多种增强策略，用于扩充训练数据

使用方法:
    python tools/data_augment.py --input datasets/screws/images/train \
                                 --labels datasets/screws/labels/train \
                                 --output datasets/screws/images/train_augmented \
                                 --num 3
"""

import os
import math
import random
import argparse
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    print("请安装Pillow: pip install Pillow")
    exit(1)


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _transform_bbox_corners(bbox, matrix, img_w, img_h):
    """通过仿射矩阵变换 bbox 四角点，返回新的 YOLO 格式 bbox，或 None 表示被裁掉"""
    cx, cy, w, h = bbox
    corners = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]
    # matrix 是 PIL getAffineTransform 的逆矩阵形式，这里用 3x3 矩阵
    # [a, b, tx]   [x]
    # [c, d, ty] * [y]
    # [0, 0, 1 ]   [1]
    a, b, tx, c, d, ty = matrix
    new_corners = []
    for x, y in corners:
        nx = a * x + b * y + tx
        ny = c * x + d * y + ty
        new_corners.append((nx, ny))

    xs = [p[0] for p in new_corners]
    ys = [p[1] for p in new_corners]
    x_min, x_max = max(0, min(xs)), min(1, max(xs))
    y_min, y_max = max(0, min(ys)), min(1, max(ys))

    new_w = x_max - x_min
    new_h = y_max - y_min
    if new_w < 0.01 or new_h < 0.01:
        return None
    return ((x_min + x_max) / 2, (y_min + y_max) / 2, new_w, new_h)


class DataAugmentor:
    """数据增强器（同步变换图片和标注框）"""

    def __init__(self, seed=42):
        random.seed(seed)
        self._img_w = 640
        self._img_h = 640

    def random_brightness(self, img, bboxes):
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor), bboxes

    def random_contrast(self, img, bboxes):
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor), bboxes

    def random_saturation(self, img, bboxes):
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor), bboxes

    def random_blur(self, img, bboxes):
        if random.random() > 0.5:
            radius = random.uniform(0.5, 1.5)
            return img.filter(ImageFilter.GaussianBlur(radius=radius)), bboxes
        return img, bboxes

    def random_noise(self, img, bboxes):
        if random.random() > 0.5:
            img_array = img.load()
            width, height = img.size
            num_noise = int(width * height * 0.01)
            for _ in range(num_noise):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                if random.random() > 0.5:
                    img_array[x, y] = (255, 255, 255)
                else:
                    img_array[x, y] = (0, 0, 0)
        return img, bboxes

    def random_rotation(self, img, bboxes):
        if random.random() <= 0.5:
            return img, bboxes

        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        # PIL.rotate 绕中心旋转，归一化坐标系下中心是 (0.5, 0.5)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        matrix = (cos_a, -sin_a, 0.5 - 0.5 * cos_a + 0.5 * sin_a,
                  sin_a, cos_a, 0.5 - 0.5 * sin_a - 0.5 * cos_a)
        new_bboxes = []
        for bbox in bboxes:
            nb = _transform_bbox_corners(bbox, matrix, self._img_w, self._img_h)
            new_bboxes.append(nb if nb is not None else bbox)
        return img, new_bboxes

    def random_flip_horizontal(self, img, bboxes):
        if random.random() <= 0.5:
            return img, bboxes

        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        new_bboxes = [(1.0 - b[0], b[1], b[2], b[3]) for b in bboxes]
        return img, new_bboxes

    def random_flip_vertical(self, img, bboxes):
        if random.random() <= 0.7:
            return img, bboxes

        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        new_bboxes = [(b[0], 1.0 - b[1], b[2], b[3]) for b in bboxes]
        return img, new_bboxes

    def random_scale(self, img, bboxes):
        if random.random() <= 0.5:
            return img, bboxes

        orig_w, orig_h = img.size
        scale = random.uniform(0.8, 1.2)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        img = img.resize(new_size, Image.BILINEAR)

        new_bboxes = []
        if scale > 1:
            left = (img.width - orig_w) // 2
            top = (img.height - orig_h) // 2
            img = img.crop((left, top, left + orig_w, top + orig_h))
            for bbox in bboxes:
                new_x = (bbox[0] - 0.5) / scale + 0.5
                new_y = (bbox[1] - 0.5) / scale + 0.5
                new_bboxes.append((new_x, new_y, bbox[2] / scale, bbox[3] / scale))
        else:
            new_img = Image.new("RGB", (orig_w, orig_h), (128, 128, 128))
            left = (orig_w - img.width) // 2
            top = (orig_h - img.height) // 2
            new_img.paste(img, (left, top))
            img = new_img
            for bbox in bboxes:
                new_x = (bbox[0] - 0.5) * scale + 0.5
                new_y = (bbox[1] - 0.5) * scale + 0.5
                new_bboxes.append((new_x, new_y, bbox[2] * scale, bbox[3] * scale))

        return img, new_bboxes

    def random_perspective(self, img, bboxes):
        if random.random() <= 0.7:
            return img, bboxes

        width, height = img.size
        coeffs = [
            1 + random.uniform(-0.05, 0.05),
            random.uniform(-0.02, 0.02),
            random.uniform(-10, 10),
            random.uniform(-0.02, 0.02),
            1 + random.uniform(-0.05, 0.05),
            random.uniform(-10, 10),
            random.uniform(-0.0001, 0.0001),
            random.uniform(-0.0001, 0.0001),
        ]
        img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

        # 透视变换近似为仿射，用于变换 bbox
        matrix = (coeffs[0], coeffs[1], coeffs[2] / width,
                  coeffs[3], coeffs[4], coeffs[5] / height)
        new_bboxes = []
        for bbox in bboxes:
            nb = _transform_bbox_corners(bbox, matrix, self._img_w, self._img_h)
            new_bboxes.append(nb if nb is not None else bbox)
        return img, new_bboxes

    def augment(self, img, bboxes, num_augmentations=3):
        """对单张图片进行多次增强，返回 [(aug_img, aug_bboxes), ...]"""
        self._img_w, self._img_h = img.size
        results = []
        all_transforms = [
            self.random_brightness, self.random_contrast,
            self.random_saturation, self.random_blur,
            self.random_noise, self.random_rotation,
            self.random_flip_horizontal, self.random_flip_vertical,
            self.random_scale, self.random_perspective,
        ]

        for _ in range(num_augmentations):
            aug_img = img.copy()
            aug_bboxes = list(bboxes)

            num_transforms = random.randint(2, 4)
            transforms = random.sample(all_transforms, num_transforms)

            for transform in transforms:
                aug_img, aug_bboxes = transform(aug_img, aug_bboxes)

            results.append((aug_img, aug_bboxes))

        return results


def read_yolo_labels(label_path):
    """读取 YOLO 格式标签，返回 [(class_id, x_center, y_center, width, height), ...]"""
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append((class_id, x_center, y_center, width, height))
    return labels


def save_augmented_data(img, bboxes_with_class, output_img_path, output_label_path):
    """保存增强后的图片和标签"""
    img.save(output_img_path, quality=95)
    with open(output_label_path, "w") as f:
        for cls_id, xc, yc, w, h in bboxes_with_class:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def augment_dataset(
    input_img_dir, input_label_dir, output_img_dir, output_label_dir, num_augmentations=3
):
    """增强整个数据集"""
    input_img_path = Path(input_img_dir)
    input_label_path = Path(input_label_dir)
    output_img_path = Path(output_img_dir)
    output_label_path = Path(output_label_dir)

    output_img_path.mkdir(parents=True, exist_ok=True)
    output_label_path.mkdir(parents=True, exist_ok=True)

    augmentor = DataAugmentor()

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in input_img_path.iterdir() if f.suffix.lower() in img_extensions]

    print(f"找到 {len(img_files)} 张图片")
    print(f"每张图片将生成 {num_augmentations} 个增强版本")
    print("-" * 50)

    total_augmented = 0

    for img_file in img_files:
        img = Image.open(img_file).convert("RGB")
        label_file = input_label_path / f"{img_file.stem}.txt"
        if not label_file.exists():
            print(f"警告: 找不到标签文件 {label_file}")
            continue

        labels = read_yolo_labels(str(label_file))
        # 转为 (x_center, y_center, width, height) 元组用于增强
        bboxes = [(l[1], l[2], l[3], l[4]) for l in labels]

        results = augmentor.augment(img, bboxes, num_augmentations)

        for i, (aug_img, aug_bboxes) in enumerate(results):
            aug_filename = f"{img_file.stem}_aug_{i:02d}"
            aug_img_path = output_img_path / f"{aug_filename}.jpg"
            aug_label_path = output_label_path / f"{aug_filename}.txt"

            # 重新附加 class_id（取原始第一个标签的 class_id，每张图只有一个钢材）
            cls_id = labels[0][0] if labels else 0
            bboxes_with_class = [(cls_id, *b) for b in aug_bboxes]
            save_augmented_data(aug_img, bboxes_with_class, str(aug_img_path), str(aug_label_path))
            total_augmented += 1

        if total_augmented % 10 == 0 and total_augmented > 0:
            print(f"已生成 {total_augmented} 个增强版本")

    print(f"\n{'=' * 50}")
    print(f"数据增强完成!")
    print(f"原始图片: {len(img_files)} 张")
    print(f"增强图片: {total_augmented} 张")
    print(f"输出目录: {output_img_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="钢材缺陷检测数据增强")
    parser.add_argument("--input", type=str, required=True, help="输入图片目录")
    parser.add_argument("--labels", type=str, required=True, help="输入标签目录")
    parser.add_argument("--output", type=str, required=True, help="输出图片目录")
    parser.add_argument("--num", type=int, default=3, help="每个样本生成的增强版本数 (默认: 3)")

    args = parser.parse_args()
    output_label_dir = str(Path(args.output).parent / "labels" / Path(args.output).name)

    augment_dataset(
        input_img_dir=args.input,
        input_label_dir=args.labels,
        output_img_dir=args.output,
        output_label_dir=output_label_dir,
        num_augmentations=args.num,
    )


if __name__ == "__main__":
    main()
