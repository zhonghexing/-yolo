"""
针对弱类别（crazing, rolled-in_scale）的数据增强脚本
增强这些类别的样本数量和多样性
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random

def augment_image(img, label_path, augment_type):
    """对图片进行指定类型的增强"""
    h, w = img.shape[:2]

    if augment_type == "rotate_90":
        # 90度旋转
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif augment_type == "rotate_180":
        # 180度旋转
        return cv2.rotate(img, cv2.ROTATE_180)

    elif augment_type == "rotate_270":
        # 270度旋转
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif augment_type == "flip_h":
        # 水平翻转
        return cv2.flip(img, 1)

    elif augment_type == "flip_v":
        # 垂直翻转
        return cv2.flip(img, 0)

    elif augment_type == "brightness_up":
        # 增加亮度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif augment_type == "brightness_down":
        # 降低亮度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.7, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif augment_type == "contrast_up":
        # 增加对比度
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    elif augment_type == "noise":
        # 添加高斯噪声
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    elif augment_type == "blur":
        # 轻微模糊
        return cv2.GaussianBlur(img, (3, 3), 0)

    else:
        return img

def transform_label(label_path, img_shape, augment_type):
    """变换标签坐标"""
    h, w = img_shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = parts[0]
        cx, cy, bw, bh = map(float, parts[1:5])

        if augment_type in ["rotate_90", "rotate_270"]:
            # 90/270度旋转：交换宽高，调整中心点
            new_cx = 1 - cy
            new_cy = cx
            new_bw = bh
            new_h = bw
            new_lines.append(f"{cls} {new_cx} {new_cy} {new_bw} {new_h}\n")

        elif augment_type == "rotate_180":
            # 180度旋转
            new_cx = 1 - cx
            new_cy = 1 - cy
            new_lines.append(f"{cls} {new_cx} {new_cy} {bw} {bh}\n")

        elif augment_type == "flip_h":
            # 水平翻转
            new_cx = 1 - cx
            new_lines.append(f"{cls} {new_cx} {cy} {bw} {bh}\n")

        elif augment_type == "flip_v":
            # 垂直翻转
            new_cy = 1 - cy
            new_lines.append(f"{cls} {cx} {new_cy} {bw} {bh}\n")

        else:
            # 其他增强不改变标签
            new_lines.append(line)

    return new_lines

def augment_weak_classes(data_dir, weak_classes=None, target_count=500):
    """
    增强弱类别样本

    Args:
        data_dir: 数据集目录
        weak_classes: 弱类别列表，默认为 crazing 和 rolled-in_scale
        target_count: 目标样本数量
    """
    if weak_classes is None:
        weak_classes = ["crazing", "rolled-in_scale"]

    data_path = Path(data_dir)
    train_images = data_path / "train" / "images"
    train_labels = data_path / "train" / "labels"

    augment_types = [
        "rotate_90", "rotate_180", "rotate_270",
        "flip_h", "flip_v",
        "brightness_up", "brightness_down",
        "contrast_up", "noise", "blur"
    ]

    for cls_name in weak_classes:
        print(f"\n处理类别: {cls_name}")

        # 获取该类别的所有图片
        cls_images = list(train_images.glob(f"{cls_name}_*.jpg"))
        current_count = len(cls_images)

        if current_count >= target_count:
            print(f"  当前样本数: {current_count} >= 目标: {target_count}，跳过")
            continue

        print(f"  当前样本数: {current_count}")
        print(f"  目标样本数: {target_count}")
        print(f"  需要增强: {target_count - current_count} 张")

        # 计算每个原始图片需要生成多少张增强图片
        augment_per_image = (target_count - current_count) // current_count + 1
        total_augmented = 0

        for i, img_path in enumerate(cls_images):
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(cls_images)}")
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 对应的标签文件
            label_path = train_labels / (img_path.stem + ".txt")

            # 对每张图片进行多种增强
            for aug_type in augment_types:
                if total_augmented >= target_count - current_count:
                    break

                # 生成新文件名
                new_name = f"{img_path.stem}_aug_{aug_type}_{total_augmented:04d}"
                new_img_path = train_images / f"{new_name}.jpg"
                new_label_path = train_labels / f"{new_name}.txt"

                # 增强图片
                aug_img = augment_image(img, str(label_path), aug_type)
                cv2.imwrite(str(new_img_path), aug_img)

                # 变换标签
                if label_path.exists():
                    new_labels = transform_label(str(label_path), img.shape, aug_type)
                    with open(new_label_path, 'w') as f:
                        f.writelines(new_labels)
                else:
                    # 如果没有标签文件，创建空文件
                    new_label_path.touch()

                total_augmented += 1

        print(f"  完成！新增 {total_augmented} 张增强图片")

if __name__ == "__main__":
    data_dir = "datasets/neu_det"
    augment_weak_classes(data_dir, target_count=500)
