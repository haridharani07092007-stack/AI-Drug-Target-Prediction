
import os
import pandas as pd
import joblib

from feature_extraction import extract_features

# Check whether model exists
model_path = "models/drug_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Train and approve the model first."
    )

model = joblib.load(model_path)

# Load candidate compounds
data = pd.read_csv("data/candidates.csv")

results = []

for _, row in data.iterrows():

    name = row["Compound"]
    smiles = row["SMILES"]

    features = extract_features(smiles)

    if features is None:
        print("Invalid SMILES:", name)
        continue

    probability = float(
        model.predict_proba([features])[0][1]
    )

    prediction = (
        "Active" if probability >= 0.5 else "Inactive"
    )

    results.append({
        "Compound": name,
        "Prediction": prediction,
        "Active_Probability": round(probability, 4)
    })

# Create result table
output = pd.DataFrame(results)

if not output.empty:

    output = output.sort_values(
        by="Active_Probability",
        ascending=False
    )

    os.makedirs("results", exist_ok=True)

    output.to_csv(
        "results/screening_results.csv",
        index=False
    )

    print("\nVIRTUAL SCREENING RESULTS")
    print(output.to_string(index=False))

    print("\nScreening completed successfully!")

else:
    print("No valid compounds found.")
