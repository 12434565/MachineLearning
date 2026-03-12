import pandas as pd

# --- data ---
df = pd.read_csv("a.csv")


# sanity check (debug-style check)
assert len(df) == 14
assert set(df["PlayTennis"].unique()) == {"Yes", "No"}

# --- priors ---
priors = df["PlayTennis"].value_counts(normalize=True).to_dict()
priors_rounded = {k: round(v, 2) for k, v in priors.items()}

# --- query ---
x = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}

def score_unnormalized_unsmoothed(class_label: str) -> float:
    sub = df[df["PlayTennis"] == class_label]
    score = priors[class_label]  # prior (not rounded for math)
    for feat, val in x.items():
        p = (sub[feat] == val).mean()  # unsmoothed MLE
        score *= p
    return score

scores = {c: score_unnormalized_unsmoothed(c) for c in ["Yes", "No"]}
prediction = max(scores, key=scores.get)

print("Priors (rounded to 2 decimals):", priors_rounded)
print("Unnormalized scores:", scores)
print("Prediction:", prediction)