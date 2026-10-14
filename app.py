from flask import Flask, request
import joblib, pandas as pd
app = Flask(__name__)
model = joblib.load("model.pkl")
df = pd.read_csv("patients.csv").set_index("patient_id")

@app.route("/")
def home():
    pid = request.args.get("id")
    page = '<h2>PulmoCheck</h2><form>Patient ID (1-1000): <input name="id"><button>Predict</button></form>'
    if pid and pid.isdigit() and int(pid) in df.index:
        p = df.loc[int(pid)]
        risk = model.predict(p[model.feature_names_in_].astype(float).to_frame().T)[0]
        yn = lambda v: "Yes" if v else "No"
        why = [t for c, t in [(p.smoker, "Smokes"), (p.drinks, "Drinks alcohol"), (p.lung_issue, f"Past lung issue: {p.history}"),
                              (p.cough_days > 10, f"Cough for {p.cough_days} days"), (p.oxygen < 92, f"Low oxygen ({p.oxygen}%)")] if c]
        page += f"<p>Gender: {'Male' if p.gender else 'Female'} | Age: {p.age} | Smoker: {yn(p.smoker)} | Drinks: {yn(p.drinks)}<br>History: {p.history} | Cough: {p.cough_days} days | Oxygen: {p.oxygen}%</p>"
        page += f"<h3>{'High risk' if risk else 'Low risk'}</h3><b>Risk factors:</b><ul>{''.join(f'<li>{w}</li>' for w in why) or '<li>None</li>'}</ul>"
    elif pid:
        page += "<p>Patient not found</p>"
    return page

app.run()