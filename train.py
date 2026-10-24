import pandas as pd, joblib, sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("cancer_patient_data_sets.csv").drop(columns=["index"])
df.to_sql("patients", sqlite3.connect("pulmo.db"), if_exists="replace", index=False)

X, y = df.drop(columns=["Patient Id", "Level"]), df["Level"]
print("Accuracy (5-fold):", cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5).mean())
joblib.dump(RandomForestClassifier(random_state=42).fit(X, y), "model.pkl")
