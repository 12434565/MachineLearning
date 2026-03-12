import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------
# 1) 读数据
# -----------------------------
df = pd.read_csv("TPM_LiY_WT_excludeSpecials.tsv", sep="\t", index_col=0)
df = df.drop(columns=["WT_p18_1"])

# 差异分析结果（gene 为 index，至少有 logFC）
# de = pd.read_csv("DE_results.tsv", sep="\t", index_col=0)
# 去掉全 0 的基因（强烈推荐）
df = df.loc[df.sum(axis=1) > 0]

# -----------------------------
# 2) 选“平均表达量最高的 30 个基因”
# -----------------------------
top30_genes = df.mean(axis=1).sort_values(ascending=False).head(30).index
df_top30 = df.loc[top30_genes]

# -----------------------------
# 3) log1p + gene-wise Z-score
# -----------------------------
X = np.log1p(df_top30)
X = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1).replace(0, np.nan), axis=0)
X = X.fillna(0)

# -----------------------------
# 4) 样本注释（Group / Timepoint）
# -----------------------------
samples = X.columns.astype(str)

group = pd.Series([s.split("_")[0] for s in samples], index=samples)
timepoint = pd.Series([s.split("_")[1].upper() for s in samples], index=samples)

group_palette = {"WT": "#4C78A8", "LiY": "#E45756"}
tp_palette = {"L1": "#72B7B2", "P18": "#F2CF5B"}

col_colors = pd.DataFrame({
    "Group": group.map(group_palette),
    "Timepoint": timepoint.map(tp_palette),
}, index=samples)

# -----------------------------
# 5) 画 clustermap
# -----------------------------
sns.set(context="notebook", style="white")

g = sns.clustermap(
    X,
    method="average",
    metric="euclidean",
    row_cluster=True,
    col_cluster=True,
    col_colors=col_colors,
    yticklabels=False,     # ✅ 不显示 gene names
    xticklabels=True,      # ✅ 显示 sample names
    figsize=(12, 10),
    cmap="vlag"
)

plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)

# -----------------------------
# 6) 图例
# -----------------------------
handles_group = [Patch(facecolor=group_palette[k], label=f"Group: {k}") for k in group_palette]
handles_tp = [Patch(facecolor=tp_palette[k], label=f"Timepoint: {k}") for k in tp_palette]

g.ax_heatmap.legend(
    handles=handles_group + handles_tp,
    title="Annotations",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0)
)

plt.savefig("heatmap_notallsamples_top30_highest_expression.png", dpi=300, bbox_inches="tight")