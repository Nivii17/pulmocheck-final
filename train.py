import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

n = 1000
r = lambda a, b: np.random.randint(a, b, n)
df = pd.DataFrame({"patient_id": range(1, n + 1), "gender": r(0, 2), "age": r(20, 80), "smoker": r(0, 2),
                   "drinks": r(0, 2), "cough_days": r(0, 30), "oxygen": r(85, 100),
                   "history": np.random.choice(["None", "Asthma", "Bronchitis", "Pneumonia"], n, p=[.6, .15, .15, .1])})
df["lung_issue"] = (df.history != "None").astype(int)
df["risk"] = (((df.smoker == 1) & (df.cough_days > 10)) | (df.oxygen < 92) | ((df.lung_issue == 1) & (df.smoker == 1))).astype(int)
df.to_csv("patients.csv", index=False)

X, y = df.drop(columns=["patient_id", "history", "risk"]), df["risk"]
print("Accuracy (5-fold):", cross_val_score(RandomForestClassifier(), X, y, cv=5).mean())
joblib.dump(RandomForestClassifier().fit(X, y), "model.pkl")