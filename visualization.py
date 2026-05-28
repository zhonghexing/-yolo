"""
螺丝缺陷检测可视化模块
Screw Defect Detection Visualization Module

功能：
    - 检测结果绘制（检测框、标签、信息面板）
    - 统计图表生成（缺陷分布饼图、柱状图、置信度分布）
    - 混淆矩阵可视化
    - 批量检测报告图表

使用方法：
    python visualization.py --log runs/logs/detection_log_xxx.csv --output runs/viz
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，服务器环境也能用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from constants import CLASS_NAMES, CLASS_NAMES_CN, CLASS_COLORS_HEX

# 尝试使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ============================================================
# 常量
# ============================================================

CLASS_NAMES_CN_LIST = [CLASS_NAMES_CN[name] for name in CLASS_NAMES]

# 配色方案
COLORS_6 = CLASS_COLORS_HEX
COLORS_PASS_FAIL = ['#2ecc71', '#e74c3c']  # 绿色=合格, 红色=不合格


# ============================================================
# 统计图表生成器
# ============================================================

class StatisticsVisualizer:
    """
    统计图表生成器

    根据检测结果生成各类统计图表，支持保存为图片。
    """

    def __init__(self, output_dir: str = "runs/visualization", dpi: int = 150):
        """
        参数：
            output_dir: 图表输出目录
            dpi: 输出图片分辨率
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_defect_distribution(
        self,
        results: list,
        save_name: str = "defect_distribution.png",
    ) -> str:
        """
        绘制缺陷类型分布饼图

        参数：
            results: List[ImageDetectionResult]
            save_name: 保存文件名

        返回：
            保存路径
        """
        # 统计各类型数量
        type_counts = {name: 0 for name in CLASS_NAMES_CN_LIST}

        for result in results:
            for det in result.detections:
                cn_name = det.class_name_cn
                if cn_name in type_counts:
                    type_counts[cn_name] += 1

        # 过滤掉数量为0的类别
        labels = []
        sizes = []
        colors = []
        for i, (name, count) in enumerate(type_counts.items()):
            if count > 0:
                labels.append(f"{name} ({count})")
                sizes.append(count)
                colors.append(COLORS_6[i])

        if not sizes:
            print("[可视化] 警告: 无检测数据可用于饼图")
            return ""

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            textprops={'fontsize': 11},
        )

        # 美化百分比文字
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title('螺丝缺陷类型分布', fontsize=16, fontweight='bold', pad=20)

        save_path = str(self.output_dir / save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[可视化] 缺陷分布饼图已保存: {save_path}")
        return save_path

    def plot_defect_bar_chart(
        self,
        results: list,
        save_name: str = "defect_bar_chart.png",
    ) -> str:
        """
        绘制缺陷类型柱状图

        参数：
            results: List[ImageDetectionResult]
            save_name: 保存文件名

        返回：
            保存路径
        """
        type_counts = {name: 0 for name in CLASS_NAMES_CN_LIST}

        for result in results:
            for det in result.detections:
                cn_name = det.class_name_cn
                if cn_name in type_counts:
                    type_counts[cn_name] += 1

        labels = list(type_counts.keys())
        values = list(type_counts.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(labels, values, color=COLORS_6, edgecolor='white', linewidth=0.5)

        # 在柱子上方标注数量
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(val),
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                )

        ax.set_xlabel('缺陷类型', fontsize=12)
        ax.set_ylabel('检测数量', fontsize=12)
        ax.set_title('各类缺陷检测数量统计', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)

        # 设置 y 轴从 0 开始
        ax.set_ylim(bottom=0)

        save_path = str(self.output_dir / save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[可视化] 缺陷柱状图已保存: {save_path}")
        return save_path

    def plot_confidence_distribution(
        self,
        results: list,
        save_name: str = "confidence_distribution.png",
    ) -> str:
        """
        绘制置信度分布直方图

        参数：
            results: List[ImageDetectionResult]
            save_name: 保存文件名

        返回：
            保存路径
        """
        conf_pass = []   # 合格品置信度
        conf_defect = [] # 缺陷品置信度

        for result in results:
            for det in result.detections:
                if det.is_defect:
                    conf_defect.append(det.confidence)
                else:
                    conf_pass.append(det.confidence)

        fig, ax = plt.subplots(figsize=(10, 6))

        bins = np.linspace(0, 1, 21)  # 0.05 的间隔

        if conf_pass:
            ax.hist(conf_pass, bins=bins, alpha=0.7, color=COLORS_PASS_FAIL[0],
                    label=f'合格 (n={len(conf_pass)})', edgecolor='white')
        if conf_defect:
            ax.hist(conf_defect, bins=bins, alpha=0.7, color=COLORS_PASS_FAIL[1],
                    label=f'缺陷 (n={len(conf_defect)})', edgecolor='white')

        ax.set_xlabel('置信度', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.set_title('检测置信度分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_xlim(0, 1)

        save_path = str(self.output_dir / save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[可视化] 置信度分布图已保存: {save_path}")
        return save_path

    def plot_pass_fail_summary(
        self,
        results: list,
        save_name: str = "pass_fail_summary.png",
    ) -> str:
        """
        绘制合格/不合格汇总饼图

        参数：
            results: List[ImageDetectionResult]
            save_name: 保存文件名

        返回：
            保存路径
        """
        pass_count = sum(1 for r in results if not r.has_defect)
        fail_count = len(results) - pass_count

        fig, ax = plt.subplots(figsize=(7, 7))

        sizes = [pass_count, fail_count]
        labels = [f'合格 ({pass_count})', f'不合格 ({fail_count})']

        # 过滤掉为 0 的
        filtered = [(s, l, c) for s, l, c in zip(sizes, labels, COLORS_PASS_FAIL) if s > 0]
        if not filtered:
            plt.close()
            return ""

        sizes_f, labels_f, colors_f = zip(*filtered)

        wedges, texts, autotexts = ax.pie(
            sizes_f,
            labels=labels_f,
            colors=colors_f,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12},
        )

        for autotext in autotexts:
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')

        total = len(results)
        ax.set_title(f'检测结果汇总 (共{total}个样本)', fontsize=16, fontweight='bold', pad=20)

        save_path = str(self.output_dir / save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[可视化] 合格/不合格汇总图已保存: {save_path}")
        return save_path

    def plot_inference_time(
        self,
        results: list,
        save_name: str = "inference_time.png",
    ) -> str:
        """
        绘制逐张推理耗时折线图

        参数：
            results: List[ImageDetectionResult]
            save_name: 保存文件名

        返回：
            保存路径
        """
        times = [r.inference_time_ms for r in results]
        indices = list(range(1, len(times) + 1))

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(indices, times, 'b-o', markersize=4, linewidth=1.5, label='单张耗时')
        avg_time = sum(times) / len(times) if times else 0
        ax.axhline(y=avg_time, color='r', linestyle='--', linewidth=1.5, label=f'平均耗时 ({avg_time:.1f}ms)')

        ax.set_xlabel('样本序号', fontsize=12)
        ax.set_ylabel('推理耗时 (ms)', fontsize=12)
        ax.set_title('逐张推理耗时', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(bottom=0)

        save_path = str(self.output_dir / save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[可视化] 推理耗时图已保存: {save_path}")
        return save_path


# ============================================================
# 混淆矩阵可视化
# ============================================================

class ConfusionMatrixVisualizer:
    """
    混淆矩阵可视化器

    支持从检测日志 CSV 或直接从结果列表生成混淆矩阵。
    """

    def __init__(self, output_dir: str = "runs/visualization", dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_from_results(
        self,
        results: list,
        ground_truths: Optional[List[int]] = None,
        save_name: str = "confusion_matrix.png",
    ) -> str:
        """
        从检测结果绘制混淆矩阵

        参数：
            results: List[ImageDetectionResult]
            ground_truths: 真实类别 ID 列表（与 results 一一对应）
                           如果为 None，则从文件名推断
            save_name: 保存文件名

        返回：
            保存路径
        """
        n_classes = len(CLASS_NAMES)

        # 获取预测类别
        y_pred = []
        for result in results:
            if result.detections:
                # 取置信度最高的检测结果作为该图的预测类别
                best_det = max(result.detections, key=lambda d: d.confidence)
                y_pred.append(best_det.class_id)
            else:
                y_pred.append(-1)  # 无检测结果

        # 获取真实类别
        if ground_truths is not None:
            y_true = ground_truths
        else:
            y_true = self._infer_ground_truth(results)

        if len(y_true) != len(y_pred):
            print("[可视化] 警告: 真实标签和预测标签数量不匹配")
            return ""

        # 构建混淆矩阵
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true_cls, pred_cls in zip(y_true, y_pred):
            if 0 <= true_cls < n_classes and 0 <= pred_cls < n_classes:
                cm[true_cls][pred_cls] += 1

        return self._render_confusion_matrix(cm, save_name)

    def _infer_ground_truth(self, results: list) -> List[int]:
        """从文件名推断真实类别"""
        y_true = []
        for result in results:
            fname = Path(result.image_path).stem.lower()
            found = False
            for i, cls_name in enumerate(CLASS_NAMES):
                if cls_name in fname:
                    y_true.append(i)
                    found = True
                    break
            if not found:
                y_true.append(0)  # 默认为 normal
        return y_true

    def _render_confusion_matrix(self, cm: np.ndarray, save_name: str) -> str:
        """渲染混淆矩阵图表"""
        n_classes = cm.shape[0]

        # 归一化
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # ---- 左图: 绝对数值 ----
        im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[0].set_title('混淆矩阵 (绝对数值)', fontsize=14, fontweight='bold')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # 标注数值
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                color = 'white' if cm[i, j] > thresh else 'black'
                axes[0].text(
                    j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color=color, fontsize=12, fontweight='bold',
                )

        axes[0].set_xticks(range(n_classes))
        axes[0].set_yticks(range(n_classes))
        axes[0].set_xticklabels(CLASS_NAMES_CN_LIST, rotation=45, ha='right', fontsize=10)
        axes[0].set_yticklabels(CLASS_NAMES_CN_LIST, fontsize=10)
        axes[0].set_xlabel('预测类别', fontsize=12)
        axes[0].set_ylabel('真实类别', fontsize=12)

        # ---- 右图: 归一化 ----
        im2 = axes[1].imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title('混淆矩阵 (归一化)', fontsize=14, fontweight='bold')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                axes[1].text(
                    j, i, f'{val:.2f}',
                    ha='center', va='center',
                    color=color, fontsize=12, fontweight='bold',
                )

        axes[1].set_xticks(range(n_classes))
        axes[1].set_yticks(range(n_classes))
        axes[1].set_xticklabels(CLASS_NAMES_CN_LIST, rotation=45, ha='right', fontsize=10)
        axes[1].set_yticklabels(CLASS_NAMES_CN_LIST, fontsize=10)
        axes[1].set_xlabel('预测类别', fontsize=12)
        axes[1].set_ylabel('真实类别', fontsize=12)

        plt.tight_layout()

        save_path = str(self.output_dir / save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"[可视化] 混淆矩阵已保存: {save_path}")
        return save_path

    def plot_from_csv(
        self,
        csv_path: str,
        save_name: str = "confusion_matrix_from_log.png",
    ) -> str:
        """
        从检测日志 CSV 生成混淆矩阵

        CSV 需包含 class_name, class_name_cn 字段。
        注意：CSV 日志只记录了预测结果，需要额外的真实标签信息。
        如果文件名包含类别信息，可自动推断真实标签。

        参数：
            csv_path: CSV 文件路径
            save_name: 保存文件名

        返回：
            保存路径
        """
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # 按图片分组
        grouped = df.groupby('image_path')

        results_data = []
        for img_path, group in grouped:
            first_row = group.iloc[0]
            # 取置信度最高的检测
            if first_row['class_name'] != 'none':
                best_idx = group['confidence'].astype(float).idxmax()
                best_row = group.loc[best_idx]
                pred_class = best_row['class_name']
            else:
                pred_class = 'none'

            # 从文件名推断真实类别
            fname = Path(img_path).stem.lower()
            true_class = 'normal'
            for cls_name in CLASS_NAMES:
                if cls_name in fname:
                    true_class = cls_name
                    break

            results_data.append({
                'image_path': img_path,
                'true_class': true_class,
                'pred_class': pred_class,
            })

        # 构建混淆矩阵
        n_classes = len(CLASS_NAMES)
        cm = np.zeros((n_classes, n_classes), dtype=int)

        class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        for item in results_data:
            true_idx = class_to_idx.get(item['true_class'], -1)
            pred_idx = class_to_idx.get(item['pred_class'], -1)
            if 0 <= true_idx < n_classes and 0 <= pred_idx < n_classes:
                cm[true_idx][pred_idx] += 1

        return self._render_confusion_matrix(cm, save_name)


# ============================================================
# 综合报告生成器
# ============================================================

class ReportGenerator:
    """
    综合检测报告生成器

    将所有可视化结果整合为一份完整的检测报告。
    """

    def __init__(self, output_dir: str = "runs/visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats_viz = StatisticsVisualizer(str(self.output_dir))
        self.cm_viz = ConfusionMatrixVisualizer(str(self.output_dir))

    def generate_full_report(
        self,
        results: list,
        ground_truths: Optional[List[int]] = None,
    ) -> Dict[str, str]:
        """
        生成完整的可视化报告

        参数：
            results: List[ImageDetectionResult]
            ground_truths: 真实类别列表（可选）

        返回：
            dict: 各图表保存路径
        """
        print("\n" + "=" * 50)
        print("生成可视化报告")
        print("=" * 50)

        paths = {}

        # 1. 合格/不合格汇总
        paths['pass_fail'] = self.stats_viz.plot_pass_fail_summary(results)

        # 2. 缺陷类型分布饼图
        paths['defect_pie'] = self.stats_viz.plot_defect_distribution(results)

        # 3. 缺陷类型柱状图
        paths['defect_bar'] = self.stats_viz.plot_defect_bar_chart(results)

        # 4. 置信度分布
        paths['confidence'] = self.stats_viz.plot_confidence_distribution(results)

        # 5. 推理耗时
        paths['inference_time'] = self.stats_viz.plot_inference_time(results)

        # 6. 混淆矩阵
        paths['confusion_matrix'] = self.cm_viz.plot_from_results(
            results, ground_truths=ground_truths
        )

        # 过滤掉空路径
        paths = {k: v for k, v in paths.items() if v}

        print(f"\n[报告] 共生成 {len(paths)} 张图表，保存在: {self.output_dir}")
        print("=" * 50)

        return paths

    def generate_text_report(self, results: list) -> str:
        """
        生成文本报告

        参数：
            results: List[ImageDetectionResult]

        返回：
            报告文本
        """
        total = len(results)
        pass_count = sum(1 for r in results if not r.has_defect)
        defect_count = total - pass_count
        avg_time = sum(r.inference_time_ms for r in results) / total if total > 0 else 0
        total_time = sum(r.inference_time_ms for r in results)

        # 缺陷类型统计
        defect_types = {}
        for r in results:
            for det in r.detections:
                if det.is_defect:
                    name = det.class_name_cn
                    defect_types[name] = defect_types.get(name, 0) + 1

        lines = [
            "=" * 60,
            "螺丝缺陷检测报告",
            "=" * 60,
            "",
            f"总样本数:       {total}",
            f"合格数:         {pass_count}",
            f"不合格数:       {defect_count}",
            f"合格率:         {pass_count/total*100:.1f}%" if total > 0 else "合格率: N/A",
            f"总推理耗时:     {total_time:.1f} ms ({total_time/1000:.2f} s)",
            f"平均推理耗时:   {avg_time:.1f} ms/张",
            "",
        ]

        if defect_types:
            lines.append("缺陷类型统计:")
            lines.append("-" * 40)
            for dtype, cnt in sorted(defect_types.items(), key=lambda x: -x[1]):
                lines.append(f"  {dtype}: {cnt} 处")
            lines.append("")

        lines.append("=" * 60)

        report_text = "\n".join(lines)

        # 保存文本报告
        report_path = self.output_dir / "detection_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"[报告] 文本报告已保存: {report_path}")
        return report_text


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="螺丝缺陷检测可视化工具")
    parser.add_argument(
        "--log", type=str, default="",
        help="检测日志 CSV 文件路径"
    )
    parser.add_argument(
        "--output", type=str, default="runs/visualization",
        help="输出目录 (默认: runs/visualization)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="输出图片 DPI (默认: 150)"
    )
    return parser.parse_args()


def main():
    """命令行入口：从 CSV 日志生成可视化报告"""
    args = parse_args()

    if not args.log:
        print("[提示] 请指定检测日志 CSV 文件路径")
        print("示例: python visualization.py --log runs/logs/detection_log_xxx.csv")
        return

    csv_path = Path(args.log)
    if not csv_path.exists():
        print(f"[错误] CSV 文件不存在: {csv_path}")
        return

    # 从 CSV 读取数据并重建结果对象
    # 这里简化处理，直接使用 CSV 绘制混淆矩阵
    cm_viz = ConfusionMatrixVisualizer(args.output, args.dpi)
    cm_viz.plot_from_csv(str(csv_path))

    print(f"\n[完成] 可视化结果保存在: {args.output}")


if __name__ == "__main__":
    main()
