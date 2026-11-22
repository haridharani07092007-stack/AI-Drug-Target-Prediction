from feature_extraction import extract_features

smiles = "CCO"

features = extract_features(smiles)

if features is not None:
    print("Feature extraction successful!")
    print("Feature length:", len(features))
    print("Features:", features)
else:
    print("Invalid chemical structure")