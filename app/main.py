"""FastAPI ASGI entry point and backward-compatible pipeline CLI."""

import argparse
import json
import sys
from pathlib import Path

from app.api import app
from src.config import settings
from src.pipeline.pipeline import MedicalDocumentPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Healthcare Document AI - Classification and Extraction Pipeline"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the medical document image (JPG, PNG, WebP).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.CLASSIFICATION_CONFIDENCE_THRESHOLD,
        help=f"Classification confidence threshold (default: {settings.CLASSIFICATION_CONFIDENCE_THRESHOLD})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)

    if not image_path.is_file():
        print(f"\n[ERROR] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" HEALTHCARE DOCUMENT AI PIPELINE")
    print("=" * 60)
    print(f" Document Image : {image_path}")
    print(f" ResNet Path    : {settings.RESNET_V3_MODEL_PATH}")
    print(f" Extraction     : Ollama ({settings.OLLAMA_MODEL})")
    print(" Mode           : ResNet medical gate followed by Qwen extraction")
    print("=" * 60 + "\n")

    pipeline = MedicalDocumentPipeline(
        resnet_model_path=settings.RESNET_V3_MODEL_PATH,
        classification_threshold=args.threshold,
        extractor_backend="ollama",
    )

    result = pipeline.process(image=image_path)

    classification = result["classification"]
    print("--- 1. DOCUMENT CLASSIFICATION ---")
    print(f"Predicted Type : {classification['document_type']}")
    print(f"Confidence     : {classification['confidence'] * 100:.2f}%")
    print("Probabilities  :")
    for class_name, prob in classification["probabilities"].items():
        print(f"  - {class_name:<20}: {prob * 100:.2f}%")

    if result.get("qwen_invoked"):
        print("\n--- 2. EXTRACTED MEDICAL JSON ---")
        if result["extracted_json"]:
            print(json.dumps(result["extracted_json"], indent=2))
        else:
            print("[NO JSON EXTRACTED / EXTRACTION FAILED]")

        print("\n--- 3. VALIDATION & AUDIT ---")
        val = result["validation"]
        print(f"Valid JSON     : {val['valid']}")
        print(f"Issues Count   : {val['issue_count']}")
        if val["issues"]:
            for issue in val["issues"]:
                print(f"  * [{issue.get('type')}] {issue.get('field', '$')}: {issue.get('message')}")

    print("\n--- 4. REVIEW STATUS ---")
    print(f"Status  : {result['review']['status']}")
    if result["review"]["reasons"]:
        print("Reasons : " + "; ".join(result["review"]["reasons"]))

    print("\n" + "=" * 60)
    print(" Pipeline execution finished.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
