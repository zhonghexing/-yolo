"""
共享常量定义
所有模块统一引用此处的类别名称、颜色等常量
"""

# ============================================================
# 类别定义 - 与训练数据集保持一致
# ============================================================

# 根据当前训练数据集选择类别定义
# 如果使用 NEU-DET 数据集：
CLASS_NAMES = [
    "crazing",            # 0 - 龟裂
    "inclusion",          # 1 - 夹杂
    "patches",            # 2 - 斑块
    "pitted_surface",     # 3 - 麻点
    "rolled-in_scale",    # 4 - 氧化皮
    "scratches",          # 5 - 划痕
]

CLASS_NAMES_CN = {
    "crazing":            "龟裂",
    "inclusion":          "夹杂",
    "patches":            "斑块",
    "pitted_surface":     "麻点",
    "rolled-in_scale":    "氧化皮",
    "scratches":          "划痕",
}

# 各类别对应的颜色 (BGR 格式，用于 OpenCV 绘制)
CLASS_COLORS_BGR = {
    "crazing":            (0, 0, 255),       # 红色
    "inclusion":          (0, 128, 255),     # 橙色
    "patches":            (0, 200, 200),     # 黄绿色
    "pitted_surface":     (128, 0, 255),     # 紫红色
    "rolled-in_scale":    (0, 100, 255),     # 深橙色
    "scratches":          (255, 200, 0),     # 天蓝色
}

# 各类别对应的颜色 (RGB 格式，用于 Qt 绘制)
CLASS_COLORS_RGB = {
    "crazing":            (255, 0, 0),
    "inclusion":          (255, 128, 0),
    "patches":            (200, 200, 0),
    "pitted_surface":     (255, 0, 128),
    "rolled-in_scale":    (255, 100, 0),
    "scratches":          (0, 200, 255),
}

# matplotlib 绘图用的颜色 (十六进制)
CLASS_COLORS_HEX = [
    '#e74c3c',  # 龟裂 - 红色
    '#f39c12',  # 夹杂 - 橙色
    '#2ecc71',  # 斑块 - 黄绿色
    '#9b59b6',  # 麻点 - 紫红色
    '#e67e22',  # 氧化皮 - 深橙色
    '#3498db',  # 划痕 - 天蓝色
]

NC = len(CLASS_NAMES)  # 类别数量
