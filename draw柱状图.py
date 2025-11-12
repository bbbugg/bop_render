import numpy as np
import matplotlib.pyplot as plt

# ------------------ 可编辑区域 ------------------
tasks = ["Task A", "Task B", "Task C", "Task D"]

# 三个baseline及其+Ours
base_methods = ["Base1", "Base2", "Base3"]
methods = [m for pair in zip(base_methods, [m + "+Ours" for m in base_methods]) for m in pair]
# => ["Base1", "Base1+Ours", "Base2", "Base2+Ours", "Base3", "Base3+Ours"]

# 示例性能（每行对应任务，每列对应方法）
# 你只需要改下面的矩阵为你自己的实验数据
performance = np.array([
    [0.72, 0.85, 0.74, 0.88, 0.69, 0.83],  # Task A
    [0.68, 0.82, 0.70, 0.86, 0.66, 0.81],  # Task B
    [0.75, 0.89, 0.73, 0.87, 0.70, 0.85],  # Task C
    [0.71, 0.84, 0.69, 0.83, 0.68, 0.82],  # Task D
])

# ------------------ 图像外观设置 ------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

n_tasks = len(tasks)
n_methods = len(methods)

# 颜色和风格
cmap = plt.get_cmap("tab10")
colors_base = [cmap(i) for i in range(len(base_methods))]         # baseline颜色
colors_ours = [(*c[:3], 0.55) for c in colors_base]               # ours颜色(同色更浅或带透明度)

bar_width = 0.12  # 每个柱宽
x = np.arange(n_tasks)

fig, ax = plt.subplots(figsize=(9, 5))

# 绘制柱子
for i, base in enumerate(base_methods):
    base_x = x + (i - 1) * 2 * bar_width   # baseline 位置
    ours_x = base_x + bar_width             # ours位置

    # baseline 柱
    ax.bar(base_x, performance[:, 2*i], width=bar_width,
           color=colors_base[i], label=base if i == 0 else "", edgecolor='black', linewidth=0.4)
    # ours 柱
    ax.bar(ours_x, performance[:, 2*i+1], width=bar_width,
           color=colors_ours[i], label=base + "+Ours" if i == 0 else "", edgecolor='black', linewidth=0.4)

# 坐标轴与标签
ax.set_xticks(x + bar_width * (len(base_methods) - 1) / 2)
ax.set_xticklabels(tasks)
ax.set_ylabel("Performance (e.g. Accuracy / F1)")
ax.set_xlabel("Task")

ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)

# 图例：在下方居中
# 因为 legend 只显示一次base和一次ours，为美观单独定义
legend_elements = []
for i, base in enumerate(base_methods):
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=colors_base[i], label=base))
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=colors_ours[i], label=base + "+Ours"))

ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
          ncol=3, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig("bar_performance_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("bar_performance_comparison.pdf", bbox_inches='tight')
plt.show()
