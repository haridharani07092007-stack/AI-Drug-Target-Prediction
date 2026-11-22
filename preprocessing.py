
import pandas as pd
import os

def prepare_data(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    print("Original dataset:", df.shape)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Check required columns
    required = ["canonical_smiles", "pchembl_value"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep required data
    df = df[["canonical_smiles", "pchembl_value"]].copy()

    # Remove missing values
    df = df.dropna()

    # Convert activity values to numbers
    df["pchembl_value"] = pd.to_numeric(
        df["pchembl_value"], errors="coerce"
    )

    df = df.dropna()

    # Remove duplicate compounds
    df = df.drop_duplicates(subset=["canonical_smiles"])

    # Assign activity labels
    # 1 = Active, 0 = Inactive
    df["label"] = df["pchembl_value"].apply(
        lambda x: 1 if x >= 6 else (0 if x <= 5 else None)
    )

    # Remove ambiguous activity values
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    print("Cleaned dataset:", df.shape)
    print("Activity distribution:")
    print(df["label"].value_counts())

    return df
