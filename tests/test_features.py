
import sys
from pathlib import Path

# Add the main project folder to Python's search path
project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

from feature_extraction import extract_features


def test_valid_smiles():
    features = extract_features("CCO")

    assert features is not None
    assert len(features) == 2048


def test_invalid_smiles():
    features = extract_features("INVALID_SMILES")

    assert features is None
