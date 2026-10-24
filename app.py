from flask import Flask, request
import joblib, pandas as pd, sqlite3
app = Flask(__name__)
model = joblib.load("model.pkl")
con = sqlite3.connect("pulmo.db", check_same_thread=False)

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
        why = [f"{c} (score {p[c]})" for c in model.feature_names_in_ if c not in ("Age", "Gender") and p[c] >= 6]
        page += f"<p>Patient {pid} | Age: {p.Age} | Gender: {'Male' if p.Gender == 1 else 'Female'}</p>"
        page += f"<h3>{risk} risk ({conf:.0%} confidence)</h3><b>High-scoring factors:</b><ul>{''.join(f'<li>{w}</li>' for w in why) or '<li>None</li>'}</ul>"
    return page

if __name__ == "__main__":
    app.run()
