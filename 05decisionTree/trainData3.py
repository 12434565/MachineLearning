import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(42)
rows = []

def add(n, temp1_low, temp1_high, avg_low, avg_high, y_mean, y_sd=0.6):
    for _ in range(n):
        temp_1 = rng.uniform(temp1_low, temp1_high)
        average = rng.uniform(avg_low, avg_high)
        temp_2 = temp_1 + rng.normal(0, 1.0)
        friend = rng.uniform(20, 80)
        value = rng.normal(y_mean, y_sd)   # ← 改名为 value
        rows.append([temp_1, average, temp_2, friend, value])

# 左子树
add(8,  30, 40, 40, 46, 41.0)
add(9,  30, 40, 40, 46, 45.0)

add(29, 46, 55, 40, 46, 51.9)
add(17, 46, 55, 40, 46, 58.2)

# 右子树
add(19, 61, 66, 50, 60, 60.7)
add(23, 61, 66, 55, 60, 66.3)

add(42, 68, 74, 61, 75, 73.0)
add(15, 68, 74, 61, 75, 80.6)

# 构建 DataFrame
df = pd.DataFrame(
    rows,
    columns=["temp_1", "average", "temp_2", "friend", "value"]
)

# 特征 & 标签
X = df[["temp_1", "average", "temp_2", "friend"]]
y = df["value"]

# 训练模型
model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# 保存数据
df.to_csv("forced_like_target_tree_with_value.csv", index=False)
print("Saved forced_like_target_tree_with_value.csv", df.shape)
print(df.head())