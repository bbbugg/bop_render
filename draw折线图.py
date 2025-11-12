import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------ 可编辑区域（替换为你自己的数据） ------------------
tasks = ["Task A", "Task B", "Task C", "Task D"]

# 数据量点（可以是百分比或样本数），按升序排列
# 例如：[10, 30, 50, 100] 表示 10%,30%,50%,100%
data_amounts = np.array([10, 30, 50, 100])

# 方法定义：3 个 baseline，和对应的 baseline + Ours
base_methods = ["Base1", "Base2", "Base3"]
methods = []
for m in base_methods:
    methods.append(m)
    methods.append(m + "+Ours")
n_methods = len(methods)

# performance 示例数据：字典 -> 每个 task 对应一个 (n_methods, n_data_amounts) 数组
# 数值范围为 0..1（比如 accuracy / F1）；替换为你真实的实验值
# 这里用合成数据作为示例（你需要把这些数换成真实结果）
performance = {}
np.random.seed(0)
for t in tasks:
    # 生成示例：baseline 随数据增多慢慢提升；+Ours 在低数据点有较高提升
    perf = []
    for i, base in enumerate(base_methods):
        baseline_curve = 0.5 + 0.3 * (data_amounts / data_amounts.max())  # baseline 趋势
        # 给每个 baseline 加一点差异
        baseline_curve += 0.02 * (i - 1)
        ours_curve = baseline_curve.copy()
        # 我们的插件在小数据量有明显提升（示例）
        ours_curve += 0.08 * (1.0 - data_amounts / data_amounts.max())  # 小数据增益
        # Append baseline then baseline+ours
        perf.append(baseline_curve)
        perf.append(ours_curve)
    # perf 当前是 list of arrays length n_methods; stack 为 (n_methods, n_points)
    performance[t] = np.vstack(perf)
# ------------------ 可编辑区域结束 ----------------------------------------

# --------------- 绘图参数（你可以调整风格） -----------------
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

n_tasks = len(tasks)
ncols = 2
nrows = int(np.ceil(n_tasks / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), sharex=True, sharey=True)
axes = axes.flatten()

# 颜色：为每个 baseline 分配一个颜色（+Ours 使用相同颜色但不同线型/marker）
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(base_methods))]

linestyle_base = "-"
linestyle_ours = "--"
marker_base = "o"
marker_ours = "s"

x = data_amounts

for ax_idx, task in enumerate(tasks):
    ax = axes[ax_idx]
    arr = performance[task]  # shape (n_methods, n_points); 顺序与 methods 一致
    # 对每个 baseline 和 +Ours 画线
    for i, base in enumerate(base_methods):
        idx_base = 2 * i
        idx_ours = idx_base + 1
        # baseline
        ax.plot(x, arr[idx_base], linestyle=linestyle_base, marker=marker_base,
                label=base, color=colors[i], linewidth=1.8, markersize=5)
        # baseline + ours
        ax.plot(x, arr[idx_ours], linestyle=linestyle_ours, marker=marker_ours,
                label=base + "+Ours", color=colors[i], linewidth=1.6, markersize=5)
    ax.set_title(task)
    ax.set_xlabel("Data amount (%)")
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)

# Remove empty subplots if tasks < nrows*ncols
for j in range(n_tasks, nrows * ncols):
    fig.delaxes(axes[j])

# 共用 y 轴标签
fig.text(0.04, 0.5, "Performance (e.g. Accuracy / F1)", va='center', rotation='vertical')

# 全局图例放在下方居中
# 制作自定义 legend handles（按 methods 顺序）
handles = []
labels = []
for i, base in enumerate(base_methods):
    handles.append(Line2D([0], [0], color=colors[i], linestyle=linestyle_base, marker=marker_base))
    labels.append(base)
    handles.append(Line2D([0], [0], color=colors[i], linestyle=linestyle_ours, marker=marker_ours))
    labels.append(base + "+Ours")

# 在子图外部绘制 legend
legend = fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
                    bbox_to_anchor=(0.5, -0.02), bbox_transform=fig.transFigure)

plt.subplots_adjust(hspace=0.28, wspace=0.18, bottom=0.16)

# 设置 x 轴刻度（显示为百分比）
for ax in axes[:n_tasks]:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}%" for v in x])

# y 轴范围可以固定（例如 0~1）或根据数据自适应。这里设置为 0.4~1.0 以示意
plt.ylim(0.4, 1.0)

# 输出保存
plt.tight_layout()
plt.savefig("performance_vs_data_amount.png", bbox_inches='tight', dpi=300)
plt.savefig("performance_vs_data_amount.pdf", bbox_inches='tight')
plt.show()
