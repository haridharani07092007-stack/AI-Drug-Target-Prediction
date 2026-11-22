
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

from feature_extraction import extract_features

app = Flask(__name__)

MODEL_PATH = "models/drug_model.pkl"

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/")
def home():

    return render_template(
        "index.html",
        result=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return render_template(
            "index.html",
            result=None,
            error="Model not available. Train and approve it first."
        )

    compound = request.form.get("compound", "")
    smiles = request.form.get("smiles", "")

    features = extract_features(smiles)

    if features is None:
        return render_template(
            "index.html",
            result=None,
            error="Invalid SMILES structure."
        )

    probability = float(
        model.predict_proba([features])[0][1]
    )

    prediction = (
        "Active" if probability >= 0.5 else "Inactive"
    )

    result = {
        "compound": compound,
        "prediction": prediction,
        "probability": round(probability * 100, 2)
    }

    return render_template(
        "index.html",
        result=result,
        error=None
    )


@app.route("/screening")
def screening():

    file_path = "results/screening_results.csv"

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        records = data.to_dict(orient="records")
    else:
        records = []

    return render_template(
        "screening.html",
        records=records
    )


if __name__ == "__main__":
    app.run(debug=True)
app.run(host="0.0.0.0", port=5000, debug=False)