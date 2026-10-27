"""Predict document types from images and PDFs with the trained ResNet-18 v2.

Examples:
    python predict_document.py path/to/document.jpg
    python predict_document.py path/to/document.pdf
    python predict_document.py test_new_documents/
    python predict_document.py test_new_documents/ --list
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

from src.preprocessing.image_preprocessing import load_document_pages


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "models" / "medical_document_resnet18_v2.pth"
DEFAULT_RESULTS_DIR = ROOT / "results"
CLASS_NAMES = ["Discharge_Summary", "Lab_Report", "Prescription", "Non_Medical"]
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}
CONFIDENCE_THRESHOLD = 0.70
SUPPORTED_TEXT = ".jpg, .jpeg, .png, .webp, .pdf"

# Same evaluation transform as training: direct resize to 224x224, RGB,
# tensor conversion, and ImageNet normalization. Random augmentation is train-only.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CheckpointMismatchError(ValueError):
    """Raised when checkpoint metadata or weights do not match ResNet-18 v2."""


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise CheckpointMismatchError("Checkpoint must be a metadata dictionary.")
    required = {"model_state_dict", "classes", "class_to_idx", "architecture", "architecture_kwargs"}
    missing = required - checkpoint.keys()
    if missing:
        raise CheckpointMismatchError(f"Checkpoint is missing metadata: {sorted(missing)}")
    if checkpoint["architecture"] != "torchvision.models.resnet18":
        raise CheckpointMismatchError(
            "Architecture mismatch: expected torchvision.models.resnet18, got "
            f"{checkpoint['architecture']!r}."
        )
    if checkpoint["classes"] != CLASS_NAMES or checkpoint["class_to_idx"] != CLASS_TO_IDX:
        raise CheckpointMismatchError("Checkpoint class names/order do not match the required four-class mapping.")
    if checkpoint["architecture_kwargs"].get("num_classes") != len(CLASS_NAMES):
        raise CheckpointMismatchError("Checkpoint metadata does not specify a four-class ResNet-18 head.")
    return checkpoint


class DocumentPredictor:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        threshold: float = CONFIDENCE_THRESHOLD,
        device: str = "auto",
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        self.threshold = float(threshold)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable; using CPU.", file=sys.stderr)
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        checkpoint = _load_checkpoint(Path(checkpoint_path), self.device)
        # The checkpoint stores the complete ImageNet-initialized, fine-tuned
        # state. weights=None avoids an unnecessary download; strict loading
        # below verifies the exact ResNet-18 topology and four-class output head.
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (RuntimeError, KeyError) as exc:
            raise CheckpointMismatchError(
                f"Checkpoint weights do not match ResNet-18: {exc}"
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    def _predict_pages(self, pages: list[Image.Image]) -> tuple[str, float, list[dict[str, Any]]]:
        if not pages:
            raise ValueError("Document contains no pages.")
        page_results = []
        probabilities_per_page = []
        for page_number, page in enumerate(pages, start=1):
            tensor = INFERENCE_TRANSFORM(page.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            probabilities_per_page.append(probabilities)
            confidence, index = torch.max(probabilities, dim=0)
            page_results.append({
                "page": page_number,
                "predicted_class": CLASS_NAMES[int(index.item())],
                "confidence": round(float(confidence.item()), 4),
                "class_probabilities": {
                    name: round(float(probabilities[i].item()), 4)
                    for i, name in enumerate(CLASS_NAMES)
                },
            })

        # Deterministic PDF aggregation: arithmetic mean of page-level softmaxes.
        mean_probabilities = torch.stack(probabilities_per_page).mean(dim=0)
        confidence, index = torch.max(mean_probabilities, dim=0)
        return CLASS_NAMES[int(index.item())], float(confidence.item()), page_results

    def predict(self, path: Path, filename: str | None = None) -> dict[str, Any]:
        path = Path(path)
        file_type = "pdf" if path.suffix.lower() == ".pdf" else "image"
        name = filename or path.name
        result: dict[str, Any] = {
            "filename": name,
            "file_type": file_type,
            "predicted_class": None,
            "confidence": None,
            "model_version": "v2",
            "accepted": False,
        }
        try:
            # Use the project's renderer for both raster images and PDFs.
            pages = load_document_pages(path)
            model_prediction, confidence, page_results = self._predict_pages(pages)
        except (OSError, ValueError, RuntimeError, UnidentifiedImageError) as exc:
            result["error"] = f"Could not read document: {exc}"
            if file_type == "pdf":
                result["page_count"] = 0
                result["per_page_predictions"] = []
            return result

        result["confidence"] = round(confidence, 4)
        result["model_prediction"] = model_prediction
        if file_type == "pdf":
            result["page_count"] = len(pages)
            result["per_page_predictions"] = page_results
        if confidence < self.threshold:
            result["predicted_class"] = "Unknown/Low_Confidence"
            result["accepted"] = False
        else:
            result["predicted_class"] = model_prediction
            result["accepted"] = model_prediction in {
                "Lab_Report", "Prescription", "Discharge_Summary"
            }
        return result


def _discover(folder: Path) -> tuple[list[Path], list[Path]]:
    supported, unsupported = [], []
    for path in sorted(folder.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file():
            continue
        (supported if path.suffix.lower() in SUPPORTED_EXTENSIONS else unsupported).append(path)
    return supported, unsupported


def _write_results(results: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "predictions.json"
    csv_path = output_dir / "predictions.csv"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = [
        "filename", "file_type", "predicted_class", "confidence", "accepted",
        "model_version", "page_count", "model_prediction", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in columns})
    return csv_path, json_path


def _relative_name(path: Path, folder: Path) -> str:
    return path.relative_to(folder).as_posix()


def _print_list(files: list[Path], folder: Path, unsupported: list[Path]) -> None:
    if not files:
        print(f"No supported documents found in {folder}.")
        print(f"Supported extensions: {SUPPORTED_TEXT}")
    else:
        print(f"Supported files found: {len(files)}")
        for index, path in enumerate(files, start=1):
            print(f"{index}. {_relative_name(path, folder)}")
    if unsupported:
        print(f"Ignored unsupported files: {len(unsupported)}")
        for path in unsupported:
            print(f"  {_relative_name(path, folder)}")


def _print_results(results: list[dict[str, Any]]) -> None:
    print(f"{'Filename':<40} {'Type':<8} {'Prediction':<30} {'Confidence':>12} {'Accepted':>10}")
    print("-" * 105)
    for result in results:
        confidence = result.get("confidence")
        confidence_text = f"{confidence:.2%}" if confidence is not None else "—"
        print(
            f"{result['filename']:<40} {result['file_type']:<8} "
            f"{result['predicted_class'] or 'Invalid document':<30} "
            f"{confidence_text:>12} {str(result['accepted']):>10}"
        )
        for page in result.get("per_page_predictions", []):
            print(f"  page {page['page']}: {page['predicted_class']} ({page['confidence']:.2%})")
        if result.get("error"):
            print(f"  Error: {result['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify real document images and PDFs with ResNet-18 v2.")
    parser.add_argument("input", type=Path, help="An image/PDF file or a folder to scan recursively.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--list", action="store_true", help="List supported folder inputs without loading the model.")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input path does not exist: {args.input}", file=sys.stderr)
        return 2

    if args.input.is_file():
        if args.input.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Unsupported file type: {args.input.suffix or '(no extension)'}. Supported: {SUPPORTED_TEXT}", file=sys.stderr)
            return 2
        files = [args.input]
        folder = args.input.parent
        unsupported: list[Path] = []
        is_folder = False
    elif args.input.is_dir():
        files, unsupported = _discover(args.input)
        folder = args.input
        is_folder = True
    else:
        print(f"Input is neither a regular file nor a folder: {args.input}", file=sys.stderr)
        return 2

    if args.list:
        if not is_folder:
            print(f"Supported files found: 1\n1. {args.input.name}")
        else:
            _print_list(files, folder, unsupported)
        return 0

    if is_folder and not files:
        print(f"No supported documents found in {folder}.")
        print(f"Supported extensions: {SUPPORTED_TEXT}")
        if unsupported:
            print(f"Ignored unsupported files: {len(unsupported)}")
            for path in unsupported:
                print(f"  {_relative_name(path, folder)}")
        return 0
    if is_folder:
        print(f"Supported files found: {len(files)}")
        if unsupported:
            print(f"Ignored unsupported files: {len(unsupported)}")

    try:
        predictor = DocumentPredictor(args.checkpoint, args.threshold, args.device)
    except (FileNotFoundError, CheckpointMismatchError, ValueError, RuntimeError) as exc:
        print(f"Model initialization failed: {exc}", file=sys.stderr)
        return 2

    results = [
        predictor.predict(
            path,
            filename=_relative_name(path, folder) if is_folder else path.name,
        )
        for path in files
    ]
    if not is_folder:
        result = results[0]
        _print_results(results)
        print("\nJSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result.get("error") else 0

    _print_results(results)
    csv_path, json_path = _write_results(results, args.results_dir)
    print(f"\nCSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
