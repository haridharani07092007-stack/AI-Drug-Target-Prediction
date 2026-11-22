
import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from preprocessing import prepare_data
from feature_extraction import extract_features

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading dataset...")

df = prepare_data("data/raw/drug_data.csv")

# Convert SMILES into fingerprints
X = []
y = []

for _, row in df.iterrows():

    features = extract_features(row["canonical_smiles"])

    if features is not None:
        X.append(features)
        y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("Feature matrix:", X.shape)

# Check both classes are available
if len(np.unique(y)) != 2:
    raise ValueError("Dataset must contain active and inactive compounds.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# Generate predictions
predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_test, predictions, zero_division=0
    )),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "pr_auc": float(average_precision_score(y_test, probabilities))
}

print("\nMODEL EVALUATION")
for name, value in metrics.items():
    print(name, ":", round(value, 4))

# Save confusion matrix
cm = confusion_matrix(y_test, predictions)

pd.DataFrame(
    cm,
    index=["Actual Inactive", "Actual Active"],
    columns=["Predicted Inactive", "Predicted Active"]
).to_csv("results/confusion_matrix.csv")

# Save classification report
with open("results/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, predictions))

# Automated validation
approved = (
    metrics["f1_score"] >= 0.80
    and metrics["roc_auc"] >= 0.75
)

print("\nMODEL VALIDATION")

if approved:
    print("MODEL APPROVED")
    joblib.dump(model, "models/drug_model.pkl")
    print("Model saved successfully!")
else:
    print("MODEL NOT APPROVED")
    print("Improve the model before deployment.")

# Save metrics
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nTraining process completed!")
