import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import numpy as np


def plot_mi_cost_dual_axis(
        mi_csv_path: str,
        cost_csv_path: str,
        gamma_values: List[float],
        cap_values: List[float],
        output_path: str = "mi_cost_dual.pdf",
        figsize: Tuple[int, int] = (4, 3)
) -> None:
    """
    Plot MI and Cost with dual y-axes in one figure.
    MI on left axis (solid lines), Cost on right axis (dashed lines).

    Args:
        mi_csv_path: Path to MI.csv
        cost_csv_path: Path to Cost.csv
        gamma_values: List of gamma values to plot (e.g., [2.0, 3.0])
        cap_values: List of cap values to plot (e.g., [1.5, 2.0])
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """

    # 读取数据
    df_mi = pd.read_csv(mi_csv_path)
    df_cost = pd.read_csv(cost_csv_path)

    # 设置绘图样式
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.5

    # 数学公式使用Cambria Math
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Cambria Math'
    plt.rcParams['mathtext.it'] = 'Cambria Math:italic'
    plt.rcParams['mathtext.bf'] = 'Cambria Math:bold'

    # 定义颜色方案
    colors = [
        '#000000',  # 黑色
        '#1e3a8a',  # 深蓝
        '#ff7f0e',  # 橙
        '#2ca02c',  # 绿
        '#d62728',  # 红
        '#9467bd',  # 紫
        '#8c564b',  # 棕
        '#e377c2',  # 粉
        '#7f7f7f',  # 灰
        '#bcbd22',  # 黄绿
        '#17becf',  # 青
        '#aec7e8',  # 浅蓝
        '#ffbb78',  # 浅橙
        '#98df8a',  # 浅绿
        '#ff9896',  # 浅红
        '#c5b0d5',  # 浅紫
        '#c49c94',  # 浅棕
        '#f7b6d2',  # 浅粉
        '#c7c7c7',  # 浅灰
        '#dbdb8d',  # 浅黄绿
        '#9edae5',  # 浅青
    ]

    # 如果还不够，用colormap生成
    total_lines = len(gamma_values) * len(cap_values)
    if total_lines > len(colors):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20', total_lines)
        colors = [cmap(i) for i in range(total_lines)]

    # ============ 创建双y轴图 ============
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
    ax2 = ax1.twinx()  # 创建右y轴

    color_idx = 0
    lines_mi = []
    lines_cost = []
    labels = []

    for gamma in gamma_values:
        for cap in cap_values:
            col_name = f"gamma={gamma},cap={cap}"

            if col_name not in df_mi.columns or col_name not in df_cost.columns:
                print(f"Warning: {col_name} not found in data, skipping")
                continue

            steps = df_mi['step'].values
            mi_values = df_mi[col_name].values
            cost_values = df_cost[col_name].values

            # 标签: α=2.0, λ_penalty=1.5
            label = f"α={gamma}, λ$_{{penalty}}$={cap}"
            labels.append(label)

            color = colors[color_idx % len(colors)]

            # 绘制MI (左轴, 实线)
            line1 = ax1.plot(
                steps, mi_values,
                color=color,
                linestyle='-',
                linewidth=2.5,
                marker='',
                label=label
            )[0]
            lines_mi.append(line1)

            # 绘制Cost (右轴, 虚线)
            line2 = ax2.plot(
                steps, cost_values,
                color=color,
                linestyle='--',  # 虚线
                linewidth=2.5,
                marker='',
                alpha=0.8
            )[0]
            lines_cost.append(line2)

            color_idx += 1

    # 设置左轴 (MI)
    ax1.set_xlabel('Iteration', fontsize=17, fontweight='normal')
    ax1.set_ylabel(r'$I(C; M_{\mathrm{post}})$', fontsize=17, fontweight='normal', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(steps.min(), 2001)
    ax1.locator_params(axis='y', nbins=5)

    # 设置右轴 (Cost)
    ax2.set_ylabel('Cost', fontsize=17, fontweight='normal', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 10)
    ax2.locator_params(axis='y', nbins=5)

    # 图例: 组合MI和线型说明
    from matplotlib.lines import Line2D

    # 主图例: 参数组合
    ncol = 1 if len(labels) <= 8 else 2
    legend1 = ax1.legend(
        lines_mi, labels,
        loc='upper left',
        frameon=True,
        framealpha=0.95,
        edgecolor='black',
        fontsize=16,
        ncol=ncol,
        title='Parameters',
        title_fontsize=16
    )

    # 线型图例
    custom_lines = [
        Line2D([0], [0], color='black', linewidth=2.5, linestyle='-'),
        Line2D([0], [0], color='black', linewidth=2.5, linestyle='--', alpha=0.8)
    ]
    legend2 = ax1.legend(
        custom_lines,
        [r'$I(C; M_{\mathrm{post}})$ (left)', 'Cost (right)'],
        loc='lower right',
        frameon=True,
        framealpha=0.95,
        edgecolor='black',
        fontsize=14
    )

    # 添加第一个图例
    ax1.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dual-axis figure saved to {output_path}")
    plt.show()
    plt.close()

    print(f"Total curves plotted: {len(labels)}")


def plot_mi_and_cost_separate(
        mi_csv_path: str,
        cost_csv_path: str,
        gamma_values: List[float],
        cap_values: List[float],
        output_prefix: str = "plot",
        figsize: Tuple[int, int] = (4, 3)
) -> None:
    """
    Plot MI and Cost in two separate figures.

    Args:
        mi_csv_path: Path to MI.csv
        cost_csv_path: Path to Cost.csv
        gamma_values: List of gamma values to plot (e.g., [2.0, 3.0])
        cap_values: List of cap values to plot (e.g., [1.5, 2.0])
        output_prefix: Prefix for output files (will generate {prefix}_MI.pdf and {prefix}_Cost.pdf)
        figsize: Figure size (width, height)
    """

    # 读取数据
    df_mi = pd.read_csv(mi_csv_path)
    df_cost = pd.read_csv(cost_csv_path)

    # 设置绘图样式
    plt.rcParams['font.family'] = 'Arial'  # 普通文本用Arial
    plt.rcParams['font.size'] = 28
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['lines.linewidth'] = 3.0

    # 数学公式使用Cambria Math
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Cambria Math'
    plt.rcParams['mathtext.it'] = 'Cambria Math:italic'
    plt.rcParams['mathtext.bf'] = 'Cambria Math:bold'

    # 定义颜色方案
    colors = [
        '#000000',  # 黑色
        '#1e3a8a',  # 深蓝
        '#ff7f0e',  # 橙
        '#2ca02c',  # 绿
        '#d62728',  # 红
        '#9467bd',  # 紫
        '#8c564b',  # 棕
        '#e377c2',  # 粉
        '#7f7f7f',  # 灰
        '#bcbd22',  # 黄绿
        '#17becf',  # 青
        '#aec7e8',  # 浅蓝
        '#ffbb78',  # 浅橙
        '#98df8a',  # 浅绿
        '#ff9896',  # 浅红
        '#c5b0d5',  # 浅紫
        '#c49c94',  # 浅棕
        '#f7b6d2',  # 浅粉
        '#c7c7c7',  # 浅灰
        '#dbdb8d',  # 浅黄绿
        '#9edae5',  # 浅青
    ]

    # 如果还不够，用colormap生成
    total_lines = len(gamma_values) * len(cap_values)
    if total_lines > len(colors):
        from matplotlib import cm
        cmap = cm.get_cmap('tab20', total_lines)
        colors = [cmap(i) for i in range(total_lines)]

    # ============ 绘制 MI 图 ============
    fig1, ax1 = plt.subplots(figsize=figsize, dpi=100)

    color_idx = 0
    lines_mi = []
    labels = []

    for gamma in gamma_values:
        for cap in cap_values:
            col_name = f"gamma={gamma},cap={cap}"

            if col_name not in df_mi.columns:
                print(f"Warning: {col_name} not found in MI data, skipping")
                continue

            steps = df_mi['step'].values
            mi_values = df_mi[col_name].values

            # 标签: α=2.0, λ_penalty=1.5
            label = f"({gamma}, {cap})"
            labels.append(label)

            color = colors[color_idx % len(colors)]

            # 绘制MI
            line = ax1.plot(
                steps, mi_values,
                color=color,
                linestyle='-',
                linewidth=3.0,
                marker='',
                label=label
            )[0]
            lines_mi.append(line)

            color_idx += 1

    # 设置MI图的坐标轴
    ax1.set_xlabel('Iteration', fontsize=28, fontweight='normal')
    ax1.set_ylabel(r'$I(C; M_{\mathrm{post}})$', fontsize=28, fontweight='normal')  # Cambria Math
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # x轴范围: 2000往里拉一些
    ax1.set_xlim(steps.min(), 1999)

    # y轴刻度减少
    ax1.locator_params(axis='y', nbins=5)

    # 图例
    ncol = 1 if len(labels) <= 5 else 2
    ax1.legend(
        lines_mi, labels,
        loc='lower right',
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor='black',
        fontsize=25,
        ncol=ncol,
        columnspacing=0.5,
        handlelength=1,
        handletextpad=0.3
    )

    plt.tight_layout()
    mi_output = f"{output_prefix}_MI.pdf"
    plt.savefig(mi_output, dpi=300, bbox_inches='tight')
    print(f"MI figure saved to {mi_output}")
    plt.show()
    plt.close()

    # ============ 绘制 Cost 图 ============
    fig2, ax2 = plt.subplots(figsize=figsize, dpi=100)

    color_idx = 0
    lines_cost = []
    labels = []

    for gamma in gamma_values:
        for cap in cap_values:
            col_name = f"gamma={gamma},cap={cap}"

            if col_name not in df_cost.columns:
                print(f"Warning: {col_name} not found in Cost data, skipping")
                continue

            steps = df_cost['step'].values
            cost_values = df_cost[col_name].values

            # label = f"α={gamma}, λ$_{{P}}$={cap}"
            label = f"({gamma}, {cap})"
            labels.append(label)

            color = colors[color_idx % len(colors)]

            # 绘制Cost
            line = ax2.plot(
                steps, cost_values,
                color=color,
                linestyle='-',
                linewidth=3.0,
                marker='',
                label=label
            )[0]
            lines_cost.append(line)

            color_idx += 1

    # 设置Cost图的坐标轴
    ax2.set_xlabel('Iteration', fontsize=28, fontweight='normal')
    ax2.set_ylabel('Overhead(T)', fontsize=28, fontweight='normal')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # x轴范围: 2000往里拉一些
    ax2.set_ylim(0, 14)
    ax2.set_xlim(steps.min(), 1999)

    # y轴刻度减少
    ax2.locator_params(axis='y', nbins=5)

    # 图例
    ncol = 1 if len(labels) <= 5 else 2
    ax2.legend(
        lines_cost, labels,
        loc='upper right',
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor='black',
        fontsize=25,
        ncol=ncol,
        columnspacing=0.5,
        handlelength=1,
        handletextpad=0.3
    )

    plt.tight_layout()
    cost_output = f"{output_prefix}_OVERHEAD.pdf"
    plt.savefig(cost_output, dpi=300, bbox_inches='tight')
    print(f"Cost figure saved to {cost_output}")
    plt.show()
    plt.close()

    print(f"Total curves plotted: {len(labels)}")

# 使用示例
if __name__ == "__main__":
    # 测试16条线的情况
    # plot_mi_cost_dual_axis(
    #     mi_csv_path="./out/grid-gamma-cap-ipd-32/MI.csv",
    #     cost_csv_path="./out/grid-gamma-cap-ipd-32/Cost.csv",
    #     gamma_values=[0.5, 1.0, 2.0], # 4个gamma
    #     cap_values=[0.5, 1.0, 2.0],  # 4个cap → 16条线
    #     output_path="./out/figs/mi_cost_dual_axis.pdf",
    #     figsize=(12, 7)  # 16条线时建议用更大的图
    # )

    plot_mi_and_cost_separate(
        mi_csv_path="out/grid-gamma-cap-ipd-32/MI.csv",
        cost_csv_path="out/grid-gamma-cap-ipd-32/Cost.csv",
        gamma_values=[0.5, 1.0],
        cap_values=[0.5, 1.0, 2.0],
        output_prefix="TRAIN_IPD",  # 会生成 result_MI.pdf 和 result_Cost.pdf
        figsize=(8.5, 5.5)
    )