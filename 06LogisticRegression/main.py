import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

df = pd.read_csv("05_babies.csv")
df["LBW"] = (df["bwt"] < 100).astype(int)

X = df[["age", "weight", "height", "smoke", "parity", "gestation"]]
y = df["LBW"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # 填补缺失值
    ("scaler", StandardScaler()),                    # 标准化（可提高收敛速度）
    ("lr", LogisticRegression(max_iter=1000, class_weight='balanced')),
    # ("lr", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)

y_prob = pipe.predict_proba(X_test)[:, 1]
# y_pred = (y_prob >= 0.5).astype(int)
y_pred = (y_prob >= 0.45).astype(int)
# y_pred = (y_prob >= 0.1).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
auc = roc_auc_score(y_test, y_prob)

print("Confusion matrix:\n", cm)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("AUROC:", auc)

# 5-fold CV (用 pipeline 才不会泄漏)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print("CV AUC:", cv_auc)
print("Mean CV AUC:", cv_auc.mean())

