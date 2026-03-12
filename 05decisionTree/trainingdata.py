import numpy as np
import pandas as pd

np.random.seed(42)

rows = []

def make_block(n, temp1_range, avg_range, target_mean):
    for _ in range(n):
        temp_1 = np.random.uniform(*temp1_range)
        average = np.random.uniform(*avg_range)
        temp_2 = temp_1 + np.random.normal(0, 5)
        friend = np.random.uniform(20, 80)
        target = np.random.normal(target_mean, 2)
        rows.append([temp_1, average, temp_2, friend, target])

# --- Left subtree ---
make_block(8,  (30, 40), (40, 45), 41)   # leaf 41
make_block(9,  (30, 40), (40, 45), 45)   # leaf 45
make_block(29, (45, 55), (40, 46), 52)   # leaf 51.9
make_block(17, (45, 55), (40, 46), 58)   # leaf 58.2

# --- Right subtree ---
make_block(19, (60, 65), (50, 60), 60)   # leaf 60.7
make_block(23, (60, 65), (55, 60), 66)   # leaf 66.3
make_block(42, (65, 75), (60, 75), 73)   # leaf 73
make_block(15, (65, 75), (60, 75), 80)   # leaf 80.6

df = pd.DataFrame(rows, columns=[
    "temp_1", "average", "temp_2", "friend", "target"
])

df.to_csv("generated_tree_data.csv", index=False)

print("CSV file generated: generated_tree_data.csv")
print(df.head())