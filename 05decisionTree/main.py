import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 必须在 import pyplot 之前

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# 1) Load your training data (must contain these columns + the target)
# Example file: temps.csv
# Columns: temp_2, temp_1, average, friend, target

# df = pd.read_csv("generated_tree_data.csv")
# df = pd.read_csv("generated_tree_data.csv")
df = pd.read_csv("forced_like_target_tree_with_value.csv")

X = df[["temp_2", "temp_1", "average", "friend"]]
y = df["value"]  # <-- change to your real target column name

# 2) Train a decision tree regressor
# Use MSE criterion (squared_error) like your plot
# Depth chosen to resemble the shown tree (adjust if needed)
model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3,
    random_state=42
)
model.fit(X, y)

# 3) Visualize the trained tree
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.savefig("tree.png", dpi=200, bbox_inches="tight")
print("Saved to tree.png")

# 4) Predict for the next day sample
x_new = pd.DataFrame([{
    "temp_2": 39,
    "temp_1": 35,
    "average": 44,
    "friend": 30
}])

pred = model.predict(x_new)[0]
print("Prediction:", pred)

# 5) Which variables were used along THIS prediction path?
node_indicator = model.decision_path(x_new)
leaf_id = model.apply(x_new)[0]

features = X.columns
used_features = set()
for node_id in node_indicator.indices:
    if node_id == leaf_id:
        continue
    feat_idx = model.tree_.feature[node_id]
    if feat_idx >= 0:
        used_features.add(features[feat_idx])

print("Used variables on this path:", sorted(used_features))
