
import mlflow
import mlflow.sklearn
import json
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

# Load trained model
model = joblib.load("models/drug_model.pkl")

# Load test results
with open("results/metrics.json") as file:
    metrics = json.load(file)

mlflow.set_experiment("AI_Drug_Target_Prediction")

with mlflow.start_run():

    mlflow.log_param("algorithm", "Random Forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("fingerprint_bits", 2048)
    mlflow.log_param("target", "EGFR")

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    mlflow.sklearn.log_model(
        model,
        artifact_path="drug_prediction_model"
    )

    mlflow.log_artifact("results/metrics.json")

print("MLflow tracking completed!")
