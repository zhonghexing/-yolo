#!/usr/bin/env python3
"""
LabelMe 标注格式转 YOLO 格式转换工具

LabelMe 格式: JSON文件，包含多边形/矩形标注
YOLO 格式: txt文件，每行 "class_id x_center y_center width height"

使用方法:
    python tools/labelme_to_yolo --input labelme_annotations \
                                 --output yolo_labels \
                                 --classes datasets/screws/classes.txt
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def load_classes(classes_file: str) -> List[str]:
    """加载类别列表"""
    classes = []
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)
    return classes


def parse_labelme_json(json_path: str) -> Dict:
    """解析LabelMe JSON文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_image_size(json_data: Dict) -> Tuple[int, int]:
    """从LabelMe数据获取图片尺寸"""
    # LabelMe格式中可能包含imageWidth和imageHeight
    if "imageWidth" in json_data and "imageHeight" in json_data:
        return json_data["imageWidth"], json_data["imageHeight"]

    # 如果没有，尝试从图片文件获取
    if "imagePath" in json_data:
        # 这里需要实际读取图片，暂时返回默认值
        pass

    return None, None


def polygon_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """将多边形转换为边界框 (x_min, y_min, x_max, y_max)"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    return x_min, y_min, x_max, y_max


def convert_to_yolo_format(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """转换为YOLO格式 (x_center, y_center, width, height) 归一化"""
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return x_center, y_center, width, height


def validate_yolo_label(
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> bool:
    """验证YOLO标签是否有效"""
    if class_id < 0:
        return False
    if not (0 <= x_center <= 1):
        return False
    if not (0 <= y_center <= 1):
        return False
    if not (0 < width <= 1):
        return False
    if not (0 < height <= 1):
        return False
    return True


def convert_single_file(
    json_path: str,
    classes: List[str],
    img_width: int = None,
    img_height: int = None,
) -> List[str]:
    """转换单个LabelMe文件为YOLO格式"""

    data = parse_labelme_json(json_path)

    # 获取图片尺寸
    w, h = get_image_size(data)
    if w is None:
        w = img_width
    if h is None:
        h = img_height

    if w is None or h is None:
        print(f"错误: 无法获取图片尺寸 {json_path}")
        return []

    yolo_labels = []

    # 处理每个形状
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])

        # 查找类别ID
        if label not in classes:
            print(f"警告: 未知类别 '{label}' in {json_path}")
            continue

        class_id = classes.index(label)

        # 根据形状类型处理
        if shape_type == "rectangle":
            # 矩形: [[x1,y1], [x2,y2]]
            if len(points) == 2:
                x_min = min(points[0][0], points[1][0])
                y_min = min(points[0][1], points[1][1])
                x_max = max(points[0][0], points[1][0])
                y_max = max(points[0][1], points[1][1])
            else:
                print(f"警告: 无效的矩形格式 {json_path}")
                continue

        elif shape_type == "polygon":
            # 多边形: 转换为边界框
            if len(points) >= 3:
                x_min, y_min, x_max, y_max = polygon_to_bbox(points)
            else:
                print(f"警告: 多边形点数不足 {json_path}")
                continue

        elif shape_type == "circle":
            # 圆形: [[center], [edge]]
            if len(points) == 2:
                cx, cy = points[0]
                ex, ey = points[1]
                radius = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
                x_min = cx - radius
                y_min = cy - radius
                x_max = cx + radius
                y_max = cy + radius
            else:
                print(f"警告: 无效的圆形格式 {json_path}")
                continue

        else:
            print(f"警告: 不支持的形状类型 '{shape_type}' in {json_path}")
            continue

        # 转换为YOLO格式
        x_center, y_center, width, height = convert_to_yolo_format(
            x_min, y_min, x_max, y_max, w, h
        )

        # 验证标签
        if validate_yolo_label(class_id, x_center, y_center, width, height):
            yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_labels.append(yolo_label)
        else:
            print(f"警告: 无效的YOLO标签 {json_path}")

    return yolo_labels


def convert_dataset(
    input_dir: str,
    output_dir: str,
    classes: List[str],
    img_width: int = None,
    img_height: int = None,
):
    """转换整个数据集"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有JSON文件
    json_files = list(input_path.glob("*.json"))

    print(f"找到 {len(json_files)} 个LabelMe标注文件")
    print(f"类别: {classes}")
    print("-" * 50)

    converted_count = 0
    error_count = 0

    for json_file in json_files:
        try:
            # 转换标签
            yolo_labels = convert_single_file(
                str(json_file),
                classes,
                img_width,
                img_height,
            )

            if yolo_labels:
                # 保存YOLO格式标签
                output_file = output_path / f"{json_file.stem}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    for label in yolo_labels:
                        f.write(label + "\n")

                converted_count += 1
            else:
                print(f"跳过空文件: {json_file.name}")

        except Exception as e:
            print(f"错误处理 {json_file.name}: {str(e)}")
            error_count += 1

    print("\n" + "=" * 50)
    print(f"转换完成!")
    print(f"成功转换: {converted_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"输出目录: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="LabelMe标注转YOLO格式")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="LabelMe标注目录 (包含JSON文件)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="YOLO标签输出目录",
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="类别文件路径",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=None,
        help="图片宽度 (如果JSON中没有)",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=None,
        help="图片高度 (如果JSON中没有)",
    )

    args = parser.parse_args()

    # 加载类别
    classes = load_classes(args.classes)
    print(f"加载类别: {classes}")

    # 转换数据集
    convert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        classes=classes,
        img_width=args.img_width,
        img_height=args.img_height,
    )


if __name__ == "__main__":
    main()
