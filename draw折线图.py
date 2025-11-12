import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np  # 添加numpy库导入

# ==============================================================================
# ============================ 可编辑配置区域 ===================================
# ==============================================================================

# -------------------------- 基本设置 --------------------------
# 设置中文字体支持 - 仅使用Windows系统常见的中文字体
plt.rcParams["font.family"] = ["SimHei"]  # SimHei(黑体)在Windows系统上通常默认安装
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# -------------------------- 任务设置 --------------------------
# 定义要比较的任务列表（可自定义任务名称）
tasks = ["任务1", "任务2", "任务3", "任务4"]

# -------------------------- 数据量设置 --------------------------
# 数据量点（可以是百分比或样本数），按升序排列
data_amounts = np.array([10, 30, 50, 100])  # 例如：10%,30%,50%,100%
data_label = "数据量 (%)"  # 数据量的单位标签

# -------------------------- 方法设置 --------------------------
# 基准方法列表
base_methods = ["基准方法1", "基准方法2", "基准方法3"]
# 自动生成完整方法列表（每个基准方法对应一个"基准方法+Ours"版本）
methods = []
for m in base_methods:
    methods.append(m)
    methods.append(m + "+Ours")
n_methods = len(methods)

# -------------------------- 性能数据设置 --------------------------
# 性能数据：每个任务对应一个 (n_methods, n_data_amounts) 的数组
# 数值范围建议为 0..1（如准确率/ F1分数）
performance = {}

# ===== 以下为示例数据生成代码，实际使用时请替换为您的真实实验数据 =====
# 直接用列表定义每个任务的性能数据，格式为：
# performance[任务名称] = np.array([
#     [基准方法1在各数据量下的表现],
#     [基准方法1+Ours在各数据量下的表现],
#     [基准方法2在各数据量下的表现],
#     [基准方法2+Ours在各数据量下的表现],
#     [基准方法3在各数据量下的表现],
#     [基准方法3+Ours在各数据量下的表现]
# ])

# 任务1的数据
performance["任务1"] = np.array([
    [0.48, 0.56, 0.64, 0.78],  # 基准方法1在10%、30%、50%、100%数据量下的表现
    [0.56, 0.62, 0.68, 0.80],  # 基准方法1+Ours的表现
    [0.50, 0.58, 0.66, 0.80],  # 基准方法2的表现
    [0.58, 0.64, 0.70, 0.82],  # 基准方法2+Ours的表现
    [0.52, 0.60, 0.68, 0.82],  # 基准方法3的表现
    [0.60, 0.66, 0.72, 0.84]   # 基准方法3+Ours的表现
])

# 任务2的数据
performance["任务2"] = np.array([
    [0.48, 0.56, 0.64, 0.78],  # 基准方法1的表现
    [0.56, 0.62, 0.68, 0.80],  # 基准方法1+Ours的表现
    [0.50, 0.58, 0.66, 0.80],  # 基准方法2的表现
    [0.58, 0.64, 0.70, 0.82],  # 基准方法2+Ours的表现
    [0.52, 0.60, 0.68, 0.82],  # 基准方法3的表现
    [0.60, 0.66, 0.72, 0.84]   # 基准方法3+Ours的表现
])

# 任务3的数据
performance["任务3"] = np.array([
    [0.48, 0.56, 0.64, 0.78],  # 基准方法1的表现
    [0.56, 0.62, 0.68, 0.80],  # 基准方法1+Ours的表现
    [0.50, 0.58, 0.66, 0.80],  # 基准方法2的表现
    [0.58, 0.64, 0.70, 0.82],  # 基准方法2+Ours的表现
    [0.52, 0.60, 0.68, 0.82],  # 基准方法3的表现
    [0.60, 0.66, 0.72, 0.84]   # 基准方法3+Ours的表现
])

# 任务4的数据
performance["任务4"] = np.array([
    [0.48, 0.56, 0.64, 0.78],  # 基准方法1的表现
    [0.56, 0.62, 0.68, 0.80],  # 基准方法1+Ours的表现
    [0.50, 0.58, 0.66, 0.80],  # 基准方法2的表现
    [0.58, 0.64, 0.70, 0.82],  # 基准方法2+Ours的表现
    [0.52, 0.60, 0.68, 0.82],  # 基准方法3的表现
    [0.60, 0.66, 0.72, 0.84]   # 基准方法3+Ours的表现
])

# ==============================================================================
# ============================ 绘图代码区域 ====================================
# ==============================================================================

# --------------- 绘图参数设置 ---------------
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

# --------------- 图表布局设置 ---------------
n_tasks = len(tasks)
ncols = 2  # 列数
nrows = int(np.ceil(n_tasks / ncols))  # 行数
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=False, sharey=True)
axes = axes.flatten()

# --------------- 样式设置 ---------------
# 为每个基准方法分配颜色
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(base_methods))]

# 线条样式设置
linestyle_base = "-"  # 基准方法实线
linestyle_ours = "--"  # +Ours虚线
marker_base = "o"  # 基准方法圆形标记
marker_ours = "s"  # +Ours方形标记

x = data_amounts

# --------------- 绘制折线图 ---------------
for ax_idx, task in enumerate(tasks):
    ax = axes[ax_idx]
    arr = performance[task]  # 该任务的性能数据
    # 为每个基准方法及其+Ours版本绘制折线
    for i, base in enumerate(base_methods):
        idx_base = 2 * i
        idx_ours = idx_base + 1
        # 绘制基准方法
        ax.plot(x, arr[idx_base], linestyle=linestyle_base, marker=marker_base,
                label=base, color=colors[i], linewidth=1.8, markersize=5)
        # 绘制基准方法+Ours
        ax.plot(x, arr[idx_ours], linestyle=linestyle_ours, marker=marker_ours,
                label=base + "+Ours", color=colors[i], linewidth=1.6, markersize=5)
    # 设置子图标题和x轴标签
    ax.set_title(task)
    ax.set_xlabel(data_label)
    # 添加网格线
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)

# 移除多余的子图
for j in range(n_tasks, nrows * ncols):
    fig.delaxes(axes[j])

# 添加共用的y轴标签 - 移至图片最左侧
fig.text(0.01, 0.5, "性能指标 (如准确率 / F1分数)", va='center', rotation='vertical', fontsize=10)

# --------------- 添加图例 ---------------
# 创建自定义图例
handles = []
labels = []
for i, base in enumerate(base_methods):
    handles.append(Line2D([0], [0], color=colors[i], linestyle=linestyle_base, marker=marker_base))
    labels.append(base)
    handles.append(Line2D([0], [0], color=colors[i], linestyle=linestyle_ours, marker=marker_ours))
    labels.append(base + "+Ours")

# 在图表下方居中显示图例
legend = fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
                    bbox_to_anchor=(0.5, 0.03), bbox_transform=fig.transFigure)

# 设置y轴范围
plt.ylim(0.4, 1.0)  # 可以根据您的数据范围调整

# 设置x轴刻度显示为百分比（在布局调整前设置一次）
for ax in axes[:n_tasks]:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}%" for v in x])
    ax.tick_params(axis='x', rotation=0)  # 确保x轴标签不旋转

# --------------- 保存和显示图表 ---------------
# 先应用tight_layout，然后再调整所有布局参数以确保各元素正确显示
plt.tight_layout()
# 调整布局参数 - 增加底部边距确保图例和刻度不超出范围，增加左侧边距避免纵坐标标签重叠
plt.subplots_adjust(hspace=0.28, wspace=0.18, bottom=0.15, left=0.08)

# 再次设置x轴刻度显示为百分比（在布局调整后再次设置，确保不被覆盖）
for ax in axes[:n_tasks]:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}%" for v in x])
    ax.tick_params(axis='x', labelsize=9)  # 确保刻度标签大小合适
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

os.makedirs("output_draw", exist_ok=True)
plt.savefig("output_draw/折线1.svg", bbox_inches='tight')
plt.savefig("output_draw/折线1.pdf", bbox_inches='tight')
plt.show()