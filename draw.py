import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import textwrap

# 任务列表
tasks = [
    "Pick up the [TOY NAME]",
    "Put the toy into the basket",
    "Flip the pot upright",
    "Pull out a tissue paper",
    "Short-horizon average",
    "Put all cups in the basket",
    "Put the toy into the drawer",
    "Long-horizon average",
    "All tasks average"
]

# 自动换行
def wrap_labels(labels, width=14):
    return ['\n'.join(textwrap.wrap(l, width)) for l in labels]

labels_wrapped = wrap_labels(tasks, width=14)

# 示例数据
baseline = [37, 20, 30, 8, 25, 15, 10, 18, 22]
vq0 = [33, 35, 45, 20, 33, 18, 14, 24, 29]
vq0_l = [40, 33, 45, 22, 34, 40, 25, 32, 36]
vq0_lm = [55, 45, 60, 25, 46, 50, 40, 45, 47]

# x轴间距调整
x_spacing = 2.0
x = np.arange(len(tasks)) * x_spacing
bar_width = 0.35

# 基础风格
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "#faf8ff",
    "axes.facecolor": "#faf8ff",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.labelcolor": "#222222"
})

fig, ax = plt.subplots(figsize=(13, 5))

# 背景三大圆角方框
# 1. Short-Horizon tasks
rect1 = FancyBboxPatch(
    (x[0] - 1.0, 0),
    (x[4] - x[0]) + 1.8, 80,
    boxstyle="round,pad=0.02,rounding_size=8",
    linewidth=0,
    facecolor="#f2edf9",
    alpha=1.0,
    zorder=0
)
ax.add_patch(rect1)

# 2. Long-Horizon tasks
rect2 = FancyBboxPatch(
    (x[5] - 1.0, 0),
    (x[7] - x[5]) + 1.8, 80,
    boxstyle="round,pad=0.02,rounding_size=8",
    linewidth=0,
    facecolor="#f6f1fb",
    alpha=1.0,
    zorder=0
)
ax.add_patch(rect2)

# 3. All tasks
rect3 = FancyBboxPatch(
    (x[8] - 1.0, 0),
    2.0, 80,
    boxstyle="round,pad=0.02,rounding_size=8",
    linewidth=0,
    facecolor="#f9f7fd",
    alpha=1.0,
    zorder=0
)
ax.add_patch(rect3)

# 柱状图
ax.bar(x - 1.5 * bar_width, baseline, width=bar_width, color="#6fa8dc", label="Baseline")
ax.bar(x - 0.5 * bar_width, vq0, width=bar_width, color="#76d7c4", label=r"$VQ_{0}$")
ax.bar(x + 0.5 * bar_width, vq0_l, width=bar_width, color="#f7b26a", label=r"$VQ_{0+L}$")
ax.bar(x + 1.5 * bar_width, vq0_lm, width=bar_width, color="#f36c60", label=r"$VQ_{0+L+M}$")

# y 轴
ax.set_ylabel("Success Rate", fontsize=12)
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.25, color="#999999")
ax.set_axisbelow(True)

# 去掉上右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# X轴标签
ax.set_xticks(x)
ax.set_xticklabels(labels_wrapped, ha='center', fontsize=10, color="#444444", linespacing=1.3)

# 分组标题
ax.text((x[0] + x[4]) / 2, 73, "Short-Horizon tasks", color="#7d6fc7", fontsize=13, weight='bold', ha='center')
ax.text((x[5] + x[7]) / 2, 73, "Long-Horizon tasks", color="#7d6fc7", fontsize=13, weight='bold', ha='center')
ax.text(x[8], 73, "All tasks", color="#7d6fc7", fontsize=13, weight='bold', ha='center')

# 图例靠右上角
ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.0, 1.10),
    frameon=False,
    ncol=4,
    fontsize=10
)

plt.tight_layout(pad=2.0)

# 导出矢量图
os.makedirs("output_draw", exist_ok=True)
plt.savefig("output_draw/task_comparison_grouped.svg", format="svg", bbox_inches="tight")
plt.savefig("output_draw/task_comparison_grouped.pdf", format="pdf", bbox_inches="tight")

plt.show()
