"""Verification script to test ResNet-18 classification on a real dataset sample."""

from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.classification import DocumentClassifier
from src.config import settings

if __name__ == "__main__":
    sample_image = next((settings.RAW_DATA_PATH / "handwritten_prescription").glob("*.jpg"))
    print(f"Testing image: {sample_image}")

    classifier = DocumentClassifier()
    result = classifier.predict(sample_image)
    print("\nClassification Result:")
    pprint(result)
