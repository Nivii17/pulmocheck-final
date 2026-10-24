from flask import Flask, request
import joblib, pandas as pd, sqlite3
app = Flask(__name__)
model = joblib.load("model.pkl")
con = sqlite3.connect("pulmo.db", check_same_thread=False)

HISTORY = ["Smoking", "Passive Smoker", "Alcohol use", "chronic Lung Disease", "Genetic Risk", "Air Pollution",
           "OccuPational Hazards", "Dust Allergy", "Balanced Diet", "Obesity"]
SYMPTOMS = ["Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss", "Shortness of Breath", "Wheezing",
            "Swallowing Difficulty", "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring"]

def show(p, cols):
    return "<ul>" + "".join(f"<li><b>{c}: {p[c]}</b></li>" if p[c] >= 6 else f"<li>{c}: {p[c]}</li>" for c in cols) + "</ul>"

@app.route("/")
def home():
    pid = request.args.get("id", "").strip().upper()
    page = '<h2>PulmoCheck</h2><form>Patient ID (P1 - P1000): <input name="id"><button>Predict</button></form>'
    if pid:
        row = pd.read_sql('SELECT * FROM patients WHERE "Patient Id" = ?', con, params=(pid,))
        if row.empty:
            return page + "<p>Patient not found</p>"
        p = row.iloc[0]
        X = p[model.feature_names_in_].astype(float).to_frame().T
        risk, conf = model.predict(X)[0], model.predict_proba(X).max()
        why = [c for c in HISTORY + SYMPTOMS if p[c] >= 6]
        page += f"<p>Patient {pid} | Age: {p.Age} | Gender: {'Male' if p.Gender == 1 else 'Female'}</p>"
        page += f"<h3>{risk} risk ({conf:.0%} confidence)</h3><p><b>Main risk factors:</b> {', '.join(why) or 'None'}</p>"
        page += f"<h4>Patient history</h4>{show(p, HISTORY)}<h4>Current symptoms</h4>{show(p, SYMPTOMS)}"
        page += "<small>Scores run from 1 (low) to 9 (high). Bold means 6 or more.</small>"
    return page

if __name__ == "__main__":
    app.run()