import numpy as np
import matplotlib.pyplot as plt
from math import pi
import textwrap

# ------------------ 可编辑区域 ------------------
dimensions = [
    "Google Robot Match", "Google Robot Agg", "WidowX Robot Zero-shot",
    "Franka Robot Multi-tasks", "LIBERO", "WidowX Bridge Match",
    "WidowX Match Tuning", "Google Robot Match Tuning"
]

ranges = [
    (20, 80), (20, 70), (10, 70),
    (0, 70), (0, 80), (0, 50),
    (0, 40), (0, 30)
]

data_series = {
    "SpatialVLA (Ours)": [71.9, 68.9, 79.2, 65.4, 78.1, 44.4, 37.7, 25.1],
    "OpenVLA": [60, 62, 50, 40, 55, 30, 10, 25],
    "RT-1-X": [30, 25, 20, 15, 10, 5, 0, 2],
    "RT-1-X (2)": [45, 50, 65, 55, 60, 40, 35, 20],
}
# ------------------------------------------------

# Basic checks
N = len(dimensions)
assert len(ranges) == N, "ranges 数量必须和 dimensions 一致"
for name, vals in data_series.items():
    assert len(vals) == N, f"数据系列 '{name}' 的长度必须等于维度数 {N}"

# Angles (每个轴的角度)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合


def normalize(values, ranges):
    normed = []
    for v, (mn, mx) in zip(values, ranges):
        if mx == mn:
            normed.append(0.0)
        else:
            normed.append((v - mn) / (mx - mn))
    return normed


# 创建极坐标子图
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
plt.subplots_adjust(top=0.92, bottom=0.22)  # 给下方图例留空间

# 背景网格
num_levels = 5
grid_levels = np.linspace(0, 1, num_levels + 1)
theta_line = np.linspace(0, 2 * np.pi, 400)
for lvl in grid_levels:
    ax.plot(theta_line, [lvl] * len(theta_line), linestyle='--', linewidth=0.6, color='gray', zorder=0)

# 为每个维度绘制从中心到边缘的放射线
for angle in angles[:-1]:
    ax.plot([angle, angle], [0, 1.2], linestyle='-', linewidth=0.6, color='gray', zorder=0)

# 隐藏默认的刻度与轴标签（我们手工放置）
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_ylim(0, 1.2)  # 显式设置y轴限制，确保有足够的空间显示放射线

# ------------------ 关键修改：把多行标签作为一个文本块绘制 ------------------
label_r = 1.35  # 文本块中心半径（可调，越大越远离雷达）
max_chars_per_line = 15  # 每行最大字符数（可调）
line_spacing_pt = 3  # 行间距以点(point)为单位（我们用 matplotlib 的行间距参数）
# 注意：因为我们一次性绘制整块文本，所以只需设置文本对象的行间距参数（linespacing）

for i, angle in enumerate(angles[:-1]):
    label_text = dimensions[i]
    wrapped = textwrap.wrap(label_text, width=max_chars_per_line)
    block = "\n".join(wrapped)

    # 将整个文本块放在 (angle, label_r)，并水平+垂直居中对齐
    # 使用 linespacing 控制行间距（1.0 表示默认， >1 拉开， <1 收紧）
    # 这里我们用相对值：linespacing = 1.2 可以微调；如果想用更精细像素控制，可用 transform 混合 offset。
    ax.text(angle, label_r, block,
            size=10,
            horizontalalignment='center',  # 整块水平居中
            verticalalignment='center',  # 整块垂直居中
            rotation=0,
            linespacing=1.1,  # 行间距，可调（默认 1.0）
            fontweight='bold',  # 设置为加粗样式
            clip_on=False)

# ------------------ 其余绘图（不变） ------------------
colors = plt.cm.get_cmap("tab10")
for idx, (name, values) in enumerate(data_series.items()):
    normed = normalize(values, ranges)
    vals_plot = normed + normed[:1]
    ax.plot(angles, vals_plot, linewidth=2, label=name, color=colors(idx), marker='o', markersize=6)
    ax.fill(angles, vals_plot, alpha=0.15, color=colors(idx))

    if name == "SpatialVLA (Ours)":
        for i, (angle, value, norm_val) in enumerate(zip(angles[:-1], values, normed)):
            label_radius = norm_val - 0.1
            label_radius = max(0.2, label_radius)
            ax.text(angle, label_radius, f"{value}%", size=9,
                    horizontalalignment='center', verticalalignment='center',
                    color=colors(idx), weight='bold')

# 图例放在下方居中
n_series = len(data_series)
ncols = min(n_series, 4)
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   ncol=ncols, fontsize=10, frameon=False)

# plt.title("Customizable Multi-series Radar Chart", y=1.02, fontsize=14)
plt.show()