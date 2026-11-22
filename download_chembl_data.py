
import os
import requests
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

params = {
    "target_chembl_id": "CHEMBL203",
    "pchembl_value__isnull": "false",
    "limit": 500,
    "offset": 0,
    "only": "molecule_chembl_id,canonical_smiles,standard_type,standard_value,standard_units,pchembl_value"
}

print("Downloading EGFR dataset...")

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

records = response.json()["activities"]

df = pd.DataFrame(records)

columns = [
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "molecule_chembl_id"
]

df = df[columns]
df = df.dropna(subset=["canonical_smiles", "pchembl_value"])

df.to_csv("data/raw/drug_data.csv", index=False)

print("Download completed!")
print("Total records:", len(df))
print("File saved: data/raw/drug_data.csv")
print(df.head())
