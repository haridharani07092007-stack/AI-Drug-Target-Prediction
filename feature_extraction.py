
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def extract_features(smiles):

    # Convert SMILES into molecule
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    # Generate Morgan fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    # Convert fingerprint into NumPy array
    features = np.zeros((2048,), dtype=np.int8)

    for index in fingerprint.GetOnBits():
        features[index] = 1

    return features
