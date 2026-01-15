import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def pca_self(df, number, label_mode="none"):
    """
    label_mode:
      - "none": 不加文字
      - "all": 所有点都加样本名
      - "outlier": 只给离群点加样本名（推荐）
    """

    # ---------- 解析样本名 ----------
    samples = pd.Index(df.index.astype(str))
    genotype = samples.str.split("_").str[0]       # WT / LiY
    stage = samples.str.split("_").str[1].str.upper()  # L1 / p18 -> L1 / P18

    # ---------- 映射：颜色=基因型，形状=阶段 ----------
    color_map = {"WT": "#1f77b4", "LiY": "#d62728"}
    marker_map = {"L1": "o", "P18": "^"}

    # ---------- PCA ----------
    X_scaled = StandardScaler().fit_transform(df)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # 保存结果（带编号，避免覆盖）
    pca_df = pd.DataFrame(
        X_pca,
        index=df.index,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )
    pca_df.to_csv(f"pca_result_{number}.csv")

    # ---------- 画图：按组画，才能同时有颜色+形状 ----------
    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100

    plt.figure(figsize=(7, 6))

    # 逐组绘制（WT/LiY × L1/P18）
    for g in ["WT", "LiY"]:
        for s in ["L1", "P18"]:
            idx = (genotype == g) & (stage == s)
            if idx.sum() == 0:
                continue

            plt.scatter(
                X_pca[idx, 0],
                X_pca[idx, 1],
                c=color_map.get(g, "gray"),
                marker=marker_map.get(s, "o"),
                s=80,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
                label=f"{g} {s}"
            )

    # ---------- 加文字标签（可选） ----------
    if label_mode == "all":
        for i, name in enumerate(df.index):
            plt.text(X_pca[i, 0], X_pca[i, 1], str(name), fontsize=7)

    elif label_mode == "outlier":
        # 简单离群定义：PC1或PC2超过99分位（你可以改成 97.5 等）
        outlier = (
            (X_pca[:, 0] > np.percentile(X_pca[:, 0], 99)) |
            (X_pca[:, 1] > np.percentile(X_pca[:, 1], 99))
        )
        for i in np.where(outlier)[0]:
            plt.text(X_pca[i, 0], X_pca[i, 1], str(df.index[i]), fontsize=8)

    plt.xlabel(f"PC1 ({pc1:.1f}%)")
    plt.ylabel(f"PC2 ({pc2:.1f}%)")
    plt.title("PCA plot (color=genotype, shape=stage)")
    plt.legend(title="Group", frameon=False)
    plt.tight_layout()

    plt.savefig(f"pca_{number}.png", dpi=300, bbox_inches="tight")
    plt.close()
    return X_pca


# ---------- 读数据 ----------
df = pd.read_csv("TPM_LiY_WT_excludeSpecials.tsv", sep="\t", index_col=0)
print(df.shape)
print(df.iloc[:10, :10])

# gene×sample -> sample×gene
df = df.T

# 1) 全部样本
pca_self(df, 1, label_mode="outlier")  # 推荐只标离群点；想全标改成 "all"

# 2) 去掉 WT_p18_1
df2 = df.drop(index=["WT_p18_1"], errors="ignore")
pca_self(df2, 2, label_mode="outlier")