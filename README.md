# PulmoCheck: Respiratory Disease Risk Prediction

A web app that predicts respiratory disease risk for a stored patient. Enter a patient ID and it shows the patient's details, the predicted risk (High or Low), and the risk factors behind it.

## Features
- Patient lookup by ID from 1,000 stored records
- Random Forest classifier with 5-fold cross-validation
- Patient details: gender, age, smoking, drinking, medical history, cough duration, oxygen level
- Risk factors list: smoking, drinking, past lung issues, long cough, low oxygen

## How it works
1. `train.py` creates the patient records (`patients.csv`), evaluates a Random Forest with 5-fold cross-validation, and saves the trained model.
2. `app.py` is a Flask web app. It finds the patient by ID, feeds that patient's data to the model, and shows the prediction with the risk factors.

## Tech
Python, Flask, scikit-learn, Pandas

## Run it
```
pip install flask pandas numpy scikit-learn
python train.py
python app.py
```
Open `http://127.0.0.1:5000/` and enter a patient ID between 1 and 1000.

## Note
The demo dataset is synthetic, so the accuracy is not a real-world result. Replace it with a real patient dataset for real results.