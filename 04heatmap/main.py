import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy

# -----------------------------
# 1) 读入数据：行=gene, 列=sample
# -----------------------------
df = pd.read_csv("TPM_LiY_WT_excludeSpecials.tsv", sep="\t", index_col=0)
# df = df.drop(columns=["WT_p18_1"])

# 可选：去掉全0基因（会让聚类更稳定，也更快）
df = df.loc[df.sum(axis=1) > 0]

# -----------------------------
# 2) 预处理：log1p + 按基因做Z-score（常见热图做法）
#    注意：按行（gene）标准化，这样不同基因可比较模式
# -----------------------------
X = np.log1p(df)  # TPM 常用 log1p
X = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1).replace(0, np.nan), axis=0)
X = X.fillna(0)

# -----------------------------
# 3) 从样本名解析 group / timepoint，并做列注释颜色条
#    样本名形如 WT_L1_1, LiY_p18_3 ...
# -----------------------------
samples = X.columns.astype(str)

group = pd.Series([s.split("_")[0] for s in samples], index=samples)            # WT / LiY
timepoint = pd.Series([s.split("_")[1].upper() for s in samples], index=samples) # L1 / P18

group_palette = {"WT": "#4C78A8", "LiY": "#E45756"}      # 你也可换成自己喜欢的
tp_palette = {"L1": "#72B7B2", "P18": "#F2CF5B"}

col_colors = pd.DataFrame({
    "Group": group.map(group_palette),
    "Timepoint": timepoint.map(tp_palette),
}, index=samples)

# -----------------------------
# 4) 画 clustermap（层次聚类热图）
#    - 不显示基因名：yticklabels=False
#    - 显示样本名：xticklabels=True
# -----------------------------
sns.set(context="notebook", style="white")

g = sns.clustermap(
    X,
    method="average", metric="euclidean",
    col_cluster=True, row_cluster=False,
    col_colors=col_colors,
    yticklabels=False,           # ✅ 不显示gene names
    xticklabels=True,            # ✅ 显示sample names
    figsize=(14, 10),
    cmap="vlag",
)

# 调整样本名角度/大小
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=8)

# -----------------------------
# 5) 加图例：Group & Timepoint
# -----------------------------
handles_group = [Patch(facecolor=group_palette[k], label=f"Group: {k}") for k in group_palette]
handles_tp = [Patch(facecolor=tp_palette[k], label=f"Timepoint: {k}") for k in tp_palette]

g.ax_heatmap.legend(
    handles=handles_group + handles_tp,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    frameon=False,
    title="Annotations"
)

# 保存
# plt.savefig("heatmap_all_genes_not_all_samples.png", dpi=300, bbox_inches="tight")
plt.savefig("heatmap_all_genes_all_samples.png", dpi=300, bbox_inches="tight")
