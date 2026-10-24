# PulmoCheck: Lung Disease Risk Prediction

A small web app that predicts a patient's lung risk level (Low, Medium or High). Enter a patient ID and it shows the patient's age and gender, the predicted risk with a confidence score, and the symptoms and risk factors that scored high.

## Features
- Patient lookup by ID (P1 to P1000) using a SQL query on a SQLite database
- Random Forest classifier evaluated with 5-fold cross-validation
- Risk level prediction with confidence percentage
- High-scoring factors list (any symptom or risk factor with a score of 6 or more)

## How it works
1. `train.py` loads the patient CSV, stores it in a SQLite table (`pulmo.db`), evaluates a Random Forest with 5-fold cross-validation, and saves the trained model (`model.pkl`).
2. `app.py` is a Flask web app. It fetches the patient with a SQL `SELECT`, feeds the record to the model, and shows the prediction and risk factors.

## Dataset
[Cancer Patients and Air Pollution: A New Link](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/data) (Kaggle): 1,000 patient records with 23 features (age, gender, air pollution, smoking, alcohol use, chronic lung disease, genetic risk, symptoms such as wheezing and dry cough) and a `Level` label (Low, Medium, High).

## Tech
Python, Flask, scikit-learn, Pandas, SQLite

## Run it
```
pip install flask pandas scikit-learn
python train.py
python app.py
```
Open `http://127.0.0.1:5000/` and enter a patient ID such as `P1`, `P10` or `P100`.

## Results
5-fold cross-validated accuracy on this dataset is about 1.0 (100%).

## Limitations
- The dataset is very clean and its classes are easy to separate, so this accuracy is not a real-world result.
- The label is a lung cancer risk level, not a clinical diagnosis. This is a learning project and is not for medical use.
- The dataset does not document the Gender codes; the app assumes 1 = Male and 2 = Female.
