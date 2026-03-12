import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

# ---------- read csv safely ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "a.csv")
df = pd.read_csv(csv_path)

# debugger-style checks
assert len(df) == 14
assert set(df["PlayTennis"].unique()) == {"Yes", "No"}

X = df[["Outlook", "Temperature", "Humidity", "Wind"]]
y = df["PlayTennis"].values

# ---------- encode categorical -> integers ----------
enc = OrdinalEncoder(dtype=np.int64)
X_enc = enc.fit_transform(X)
assert X_enc.min() >= 0

# ---------- train NB (unsmoothed) ----------
nb = CategoricalNB(alpha=0.0, fit_prior=True)
nb.fit(X_enc, y)

# ---------- priors computed from the table ----------
# sklearn stores counts per class in class_count_
priors = nb.class_count_ / nb.class_count_.sum()
priors = dict(zip(nb.classes_, priors))
priors_rounded = {k: round(float(v), 2) for k, v in priors.items()}
print("Priors (rounded to 2 decimals):", priors_rounded)

# ---------- query ----------
x_query = pd.DataFrame([{
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong",
}])

xq_enc = enc.transform(x_query)
pred = nb.predict(xq_enc)[0]
proba = nb.predict_proba(xq_enc)[0]

print("Classes:", nb.classes_)
print("Predicted probabilities:", dict(zip(nb.classes_, proba)))
print("Prediction:", pred)