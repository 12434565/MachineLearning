import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rows = []

def add(n, temp1_low, temp1_high, avg_low, avg_high, y_mean, y_sd=0.6):
    # 让 temp_1 成为最强信号：target 主要由 temp_1 分段决定
    for _ in range(n):
        temp_1 = rng.uniform(temp1_low, temp1_high)
        average = rng.uniform(avg_low, avg_high)
        temp_2 = temp_1 + rng.normal(0, 1.0)   # 弱化 temp_2 信息量（别让它抢分裂）
        friend = rng.uniform(20, 80)           # 纯噪声
        target = rng.normal(y_mean, y_sd)      # 叶子 value 附近小噪声
        rows.append([temp_1, average, temp_2, friend, target])

# ★关键：在分裂阈值附近留“空档”，例如 temp_1=59.5 附近不生成样本
# 左子树：temp_1 <= 59.5  → 我们生成 temp_1 在 [30,58]（不碰 59.5 附近）
add(8,  30, 40, 40, 46, 41.0)   # temp_1<=44.5 & average<=46.8
add(9,  30, 40, 40, 46, 45.0)

add(29, 46, 55, 40, 46, 51.9)   # temp_1<=55.5 & average<=46.8
add(17, 46, 55, 40, 46, 58.2)

# 右子树：temp_1 > 59.5 → 生成 temp_1 在 [61,74]（跳过 59.5~61）
add(19, 61, 66, 50, 60, 60.7)   # average<=60.8 & temp_1<=67.5
add(23, 61, 66, 55, 60, 66.3)

add(42, 68, 74, 61, 75, 73.0)   # average<=75.6 & temp_1>67.5
add(15, 68, 74, 61, 75, 80.6)

df = pd.DataFrame(rows, columns=["temp_1","average","temp_2","friend","target"])
df.to_csv("forced_like_target_tree.csv", index=False)
print("Saved forced_like_target_tree.csv", df.shape)