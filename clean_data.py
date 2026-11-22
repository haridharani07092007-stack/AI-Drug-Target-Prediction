
import os
from preprocessing import prepare_data

os.makedirs("data/processed", exist_ok=True)

df = prepare_data("data/raw/drug_data.csv")

df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

print("Cleaned dataset saved successfully!")
