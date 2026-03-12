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
    print("PCA done")

    # # ---------- 画图：按组画，才能同时有颜色+形状 ----------
    # pc1 = pca.explained_variance_ratio_[0] * 100
    # pc2 = pca.explained_variance_ratio_[1] * 100
    #
    # plt.figure(figsize=(7, 6))
    #
    # # 逐组绘制（WT/LiY × L1/P18）
    # for g in ["WT", "LiY"]:
    #     for s in ["L1", "P18"]:
    #         idx = (genotype == g) & (stage == s)
    #         if idx.sum() == 0:
    #             continue
    #
    #         plt.scatter(
    #             X_pca[idx, 0],
    #             X_pca[idx, 1],
    #             c=color_map.get(g, "gray"),
    #             marker=marker_map.get(s, "o"),
    #             s=80,
    #             alpha=0.8,
    #             edgecolors="black",
    #             linewidths=0.5,
    #             label=f"{g} {s}"
    #         )
    #
    # # ---------- 加文字标签（可选） ----------
    # if label_mode == "all":
    #     for i, name in enumerate(df.index):
    #         plt.text(X_pca[i, 0], X_pca[i, 1], str(name), fontsize=7)
    #
    # elif label_mode == "outlier":
    #     # 简单离群定义：PC1或PC2超过99分位（你可以改成 97.5 等）
    #     outlier = (
    #         (X_pca[:, 0] > np.percentile(X_pca[:, 0], 99)) |
    #         (X_pca[:, 1] > np.percentile(X_pca[:, 1], 99))
    #     )
    #     for i in np.where(outlier)[0]:
    #         plt.text(X_pca[i, 0], X_pca[i, 1], str(df.index[i]), fontsize=8)
    #
    # plt.xlabel(f"PC1 ({pc1:.1f}%)")
    # plt.ylabel(f"PC2 ({pc2:.1f}%)")
    # plt.title("PCA plot (color=genotype, shape=stage)")
    # plt.legend(title="Group", frameon=False)
    # plt.tight_layout()
    #
    # plt.savefig(f"pca_{number}.png", dpi=300, bbox_inches="tight")
    # plt.close()
    return X_pca


# ---------- 读数据 ----------
df = pd.read_csv("TPM_LiY_WT_excludeSpecials.tsv", sep="\t", index_col=0)
print(df.shape)
print(df.iloc[:10, :10])

# gene×sample -> sample×gene
df = df.T


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from adjustText import adjust_text

def draw_ellipse(ax, x, y, color, n_std=2.0):
    """
    x, y: cluster 在 PC1 / PC2 上的坐标
    n_std: 椭圆大小，2.0~2.5 比较好看
    """
    if x.size < 2:
        return

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)

    # 按特征值大小排序
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        edgecolor=color,
        alpha=0.9
    )

    ax.add_patch(ellipse)
def kmeans_on_pca(X_pca, sample_names, number, n_clusters=4, use_n_pcs=2):
    """
    X_pca: pca_self 返回的 PCA 得分矩阵 (n_samples, n_pcs_total)
    sample_names: df.index（样本名）
    number: 用于输出文件编号
    n_clusters: 聚类数，默认 4
    use_n_pcs: 用多少个PC参与kmeans（推荐>=2，比如 2/3/5/10）
    """
    cluster_colors = [
        "#383838",  # Cluster 0
        "#5CA7C7",  # Cluster 1
        "#D4352D",  # Cluster 2
        "#FBCE6A"  # Cluster 3
    ]
    X_use = X_pca[:, :use_n_pcs]

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(X_use)

    # 保存每个样本的聚类标签
    out = pd.DataFrame({"sample": sample_names, "cluster": clusters})
    out.to_csv(f"kmeans_clusters_{number}.csv", index=False)

    # 可选：轮廓系数（越大越好；>0.5通常较好，仅作参考）
    sil = None
    if n_clusters >= 2 and len(sample_names) > n_clusters:
        sil = silhouette_score(X_use, clusters)
        with open(f"kmeans_silhouette_{number}.txt", "w") as f:
            f.write(f"silhouette_score (k={n_clusters}, use_n_pcs={use_n_pcs}) = {sil:.4f}\n")

    # 画图：用PC1/PC2展示聚类（就算kmeans用更多PC也可以只画前两维）
    # fig, ax = plt.subplots(figsize=(7, 6))
    plt.figure(figsize=(7, 6))

    for k in range(n_clusters):
        idx = clusters == k
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            s=80,
            alpha=0.6,
            edgecolors="black",
            color=cluster_colors[k],
            linewidths=0.5,
            label=f"Cluster {k}"
        )

    # 画聚类中心
    centers = km.cluster_centers_
    for k in range(n_clusters):
        plt.scatter(
            centers[k, 0],
            centers[k, 1],
            marker="*",
            s=200,  # ⭐ 星星大小（推荐 120–200）
            color=cluster_colors[k],  # 和 cluster 颜色一致
            linewidths=0,
            label=f"Center {k}"
        )
    # # ③ ⭐ 漂浮虚线外框（椭圆）
    # for k in range(n_clusters):
    #     idx = clusters == k
    #     draw_ellipse(
    #         ax,
    #         X_pca[idx, 0],
    #         X_pca[idx, 1],
    #         color=plt.cm.tab10(k),
    #         n_std=2.2  # 可调：2.0 小一点，2.5 更松
    #     )
    for i, name in enumerate(sample_names):

        plt.annotate(
            str(name),
            xy=(X_pca[i, 0], X_pca[i, 1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7,
            color=cluster_colors[clusters[i]],
            alpha=1,
            bbox=dict(
                facecolor="white",
                alpha=0.6,
                edgecolor="none",
                pad=0.2
            )
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"k-means on PCA (k={n_clusters})")
    for k in range(n_clusters):
        idx = clusters == k
        pts = X_pca[idx, :2]  # 只在 PC1 / PC2 上画外框

        if pts.shape[0] < 3:
            continue  # 点太少画不了凸包

        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]

        # 闭合曲线
        hull_pts = np.vstack([hull_pts, hull_pts[0]])

        plt.plot(
            hull_pts[:, 0],
            hull_pts[:, 1],
            linestyle="--",
            linewidth=1.5,
            color=cluster_colors[k],
            alpha=0.8
        )
    plt.legend(frameon=False)
    plt.tight_layout()
    from adjustText import adjust_text

    plt.savefig(f"kmeans_pca_{number}.png", dpi=300)
    plt.close()


    return clusters, sil


df2 = df.drop(index=["WT_p18_1"], errors="ignore")
X_pca2 = pca_self(df2, 2, label_mode="all")
kmeans_on_pca(X_pca2, df2.index, number=2, n_clusters=4, use_n_pcs=2)