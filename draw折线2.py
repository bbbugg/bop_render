import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator

# ------------------ 示例数据（按你原结构） ------------------
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

tasks = ["任务1", "任务2", "任务3", "任务4"]
data_amounts = np.array([10, 30, 50, 100])
data_label = "数据量 (%)"
base_methods = ["基准方法1", "基准方法2", "基准方法3"]

performance = {}
performance["任务1"] = np.array([
    [0.45, 0.50, 0.55, 0.60],  # bottom baseline1
    [0.55, 0.60, 0.65, 0.70],  # bottom baseline1+Ours
    [0.70, 0.75, 0.80, 0.85],  # middle baseline2
    [0.75, 0.80, 0.85, 0.90],  # middle baseline2+Ours
    [0.90, 0.92, 0.94, 0.96],  # top baseline3
    [0.92, 0.94, 0.96, 0.98],  # top baseline3+Ours
])
performance["任务2"] = np.array([
    [0.42, 0.48, 0.53, 0.58],
    [0.52, 0.58, 0.63, 0.68],
    [0.68, 0.74, 0.79, 0.84],
    [0.73, 0.79, 0.84, 0.89],
    [0.89, 0.91, 0.93, 0.95],
    [0.91, 0.93, 0.95, 0.97]
])
performance["任务3"] = performance["任务1"].copy()
performance["任务4"] = performance["任务1"].copy()

# ------------------ 绘图风格 ------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})

n_tasks = len(tasks)
ncols = 2
nrows = int(np.ceil(n_tasks / ncols))

fig = plt.figure(figsize=(12, 10))

# ===== 参数：调整 "任务间距"（outer_hspace）和 "task 内部子图间距"（inner_hspace） =====
outer_hspace = 0.25   # 外层 GridSpec 的 hspace：控制不同 task block 之间的距离（增加这个值可以让 task 之间更分开）
inner_hspace = 0.08   # 内层 GridSpecFromSubplotSpec 的 hspace：控制同一 task 内 top/mid/bottom 的间距（减小这个值会让它们更紧）
# 内层三行高度比（top, middle, bottom），总和决定每个 block 的高度分配
inner_height_ratios = [0.35, 0.32, 0.33]  # 可微调：sum 保持为 1 的倾向，但不强制

# 外部 GridSpec：每个 row 对应一个 task block（每个 block 会再细分成 3 个子图）
outer_gs = GridSpec(nrows, ncols, figure=fig, hspace=outer_hspace, wspace=0.28)

# 图例/颜色等
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(base_methods))]
linestyle_base = "-"
linestyle_ours = "--"
marker_base = "o"
marker_ours = "s"
x = data_amounts

# 存放所有轴，便于后面计算全域 bbox
all_axes = []

# --- Helper function to calculate padded ylim ---
def get_padded_ylim(data_arrays, padding_factor=0.2):
    """根据数据动态计算y轴范围，并增加一些留白"""
    min_val = np.min(data_arrays)
    max_val = np.max(data_arrays)
    range_val = max_val - min_val
    padding = range_val * padding_factor
    # 确保在数据范围极小或为零时仍有最小留白
    if padding < 0.02:
        padding = 0.02
    return min_val - padding, max_val + padding

# ===== 为每个 task block 创建内层 GridSpec 并绘图 =====
for task_idx, task in enumerate(tasks):
    col = task_idx % ncols
    row = task_idx // ncols

    # 在 outer_gs[row, col] 这个格子内部再分 3 行 1 列
    inner_gs = GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer_gs[row, col],
        height_ratios=inner_height_ratios,
        hspace=inner_hspace
    )

    # 创建按上->中->下的三个子图（ax_top, ax_middle, ax_bottom）
    ax_top = fig.add_subplot(inner_gs[0, 0])      # 最上
    ax_middle = fig.add_subplot(inner_gs[1, 0])   # 中
    ax_bottom = fig.add_subplot(inner_gs[2, 0])   # 最下

    all_axes.extend([ax_top, ax_middle, ax_bottom])

    arr = performance[task]

    # top (baseline3)
    ax_top.plot(x, arr[4], linestyle=linestyle_base, marker=marker_base, color=colors[2],
                label=base_methods[2], linewidth=1.8, markersize=5)
    ax_top.plot(x, arr[5], linestyle=linestyle_ours, marker=marker_ours, color=colors[2],
                label=base_methods[2]+"+Ours", linewidth=1.6, markersize=5)
    ax_top.fill_between(x, arr[4], arr[5], color=colors[2], alpha=0.15, interpolate=True)
    ax_top.set_ylim(get_padded_ylim([arr[4], arr[5]]))
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))  # 自动优化刻度数量

    # middle (baseline2)
    ax_middle.plot(x, arr[2], linestyle=linestyle_base, marker=marker_base, color=colors[1],
                   label=base_methods[1], linewidth=1.8, markersize=5)
    ax_middle.plot(x, arr[3], linestyle=linestyle_ours, marker=marker_ours, color=colors[1],
                   label=base_methods[1]+"+Ours", linewidth=1.6, markersize=5)
    ax_middle.fill_between(x, arr[2], arr[3], color=colors[1], alpha=0.15, interpolate=True)
    ax_middle.set_ylim(get_padded_ylim([arr[2], arr[3]]))
    ax_middle.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))  # 自动优化刻度数量

    # bottom (baseline1)
    ax_bottom.plot(x, arr[0], linestyle=linestyle_base, marker=marker_base, color=colors[0],
                   label=base_methods[0], linewidth=1.8, markersize=5)
    ax_bottom.plot(x, arr[1], linestyle=linestyle_ours, marker=marker_ours, color=colors[0],
                   label=base_methods[0]+"+Ours", linewidth=1.6, markersize=5)
    ax_bottom.fill_between(x, arr[0], arr[1], color=colors[0], alpha=0.15, interpolate=True)
    ax_bottom.set_ylim(get_padded_ylim([arr[0], arr[1]]))
    ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))  # 自动优化刻度数量

    # 标题只放在最上层子图
    ax_top.set_title(task, fontfamily='SimHei', fontweight='bold', fontsize=15)

    # x 轴只在 bottom 显示
    ax_bottom.set_xlabel(data_label)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([f"{int(v)}%" for v in x])
    ax_bottom.tick_params(axis='x', rotation=0, labelsize=8)

    # top & middle 不显示 x-axis labels
    ax_top.tick_params(axis='x', labelbottom=False, bottom=False)
    ax_middle.tick_params(axis='x', labelbottom=False, bottom=False)

    # 不在各子图放单独 y label（使用全局统一 y label）
    ax_top.set_ylabel("")
    ax_middle.set_ylabel("")
    ax_bottom.set_ylabel("")

    # 网格
    for a in (ax_top, ax_middle, ax_bottom):
        a.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)

# ===== 添加图例（图底部居中） =====
handles = []
labels = []
for i, base in enumerate(base_methods):
    handles.append(Line2D([0], [0], color=colors[i], linestyle=linestyle_base, marker=marker_base))
    labels.append(base)
    handles.append(Line2D([0], [0], color=colors[i], linestyle=linestyle_ours, marker=marker_ours))
    labels.append(base + "+Ours")

legend = fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
                    bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, columnspacing=15.0)

# ===== 计算所有 task 的垂直 union bbox，并在其垂直中心放置全局 y-label =====
# 我们使用所有 top-of-top 到 bottom-of-bottom 的 union bbox
all_pos = [ax.get_position() for ax in all_axes]
x0 = min(p.x0 for p in all_pos)
x1 = max(p.x1 for p in all_pos)
y0 = min(p.y0 for p in all_pos)  # 最低点
y1 = max(p.y1 for p in all_pos)  # 最高点
# 纵向中心放置全局 y-label
x_fig = x0 - 0.06  # 左侧偏移（可微调）
y_fig = (y0 + y1) / 2
fig.text(x_fig, y_fig, "性能指标", va='center', ha='center', rotation='vertical', fontsize=15, fontweight='bold')



# 最后微调布局
plt.tight_layout()
# bottom/left 留出一些空白以免图例/标签被遮挡
plt.subplots_adjust(bottom=0.10, left=0.12)

# 保存并显示
os.makedirs("output_draw", exist_ok=True)
plt.savefig("output_draw/折线2.svg", bbox_inches='tight')
plt.savefig("output_draw/折线2.pdf", bbox_inches='tight')
plt.show()
