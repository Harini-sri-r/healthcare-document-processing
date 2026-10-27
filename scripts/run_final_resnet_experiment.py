"""Run the final four-class ResNet-18 experiment on the local dataset.

The dataset is read-only. Exact duplicate files are excluded from the in-memory
training index only; no image is deleted, renamed, or rewritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
REPORTS = PROJECT_ROOT / "reports"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "medical_document_resnet18_v3_final.pth"
V2_MODEL = PROJECT_ROOT / "models" / "medical_document_resnet18_v2.pth"

CLASSES = ["Discharge_Summary", "Lab_Report", "Prescription", "Non_Medical"]
CLASS_DIRS = {
    "Discharge_Summary": "discharge_summaries",
    "Lab_Report": "lab_reports",
    "Prescription": "handwritten_prescription",
    "Non_Medical": "non_medical",
}
EXPECTED = {
    "Discharge_Summary": 500,
    "Lab_Report": 516,
    "Prescription": 129,
    "Non_Medical": 700,
}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SEED = 42
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def audit_dataset() -> tuple[dict[str, list[Path]], dict[str, int]]:
    paths_by_class: dict[str, list[Path]] = {}
    counts: dict[str, int] = {}
    for label in CLASSES:
        folder = DATA_ROOT / CLASS_DIRS[label]
        if not folder.is_dir():
            raise FileNotFoundError(f"Required dataset folder is missing: {folder}")
        paths = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        paths_by_class[label] = paths
        counts[label] = len(paths)

    print("FINAL DATASET AUDIT", flush=True)
    print("===================", flush=True)
    for label in CLASSES:
        print(f"{label}: {counts[label]}", flush=True)
    print(f"Total: {sum(counts.values())}", flush=True)
    if counts != EXPECTED:
        raise RuntimeError(
            "Dataset count mismatch; training stopped before model initialization. "
            f"Expected {EXPECTED}, found {counts}."
        )

    invalid: list[dict[str, str]] = []
    for label, paths in paths_by_class.items():
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                invalid.append({"file": str(path.relative_to(PROJECT_ROOT)), "error": str(exc)})
    if invalid:
        raise RuntimeError(
            f"Found {len(invalid)} unreadable images; training stopped. "
            f"Examples: {invalid[:5]}"
        )
    print("PIL image integrity: all counted image files opened successfully.", flush=True)
    return paths_by_class, counts


def deduplicate(paths_by_class: dict[str, list[Path]], counts: dict[str, int]):
    records: list[dict[str, str]] = []
    seen: dict[str, dict[str, str]] = {}
    duplicate_rows: list[dict[str, str]] = []
    print("Calculating SHA-256 for every image...", flush=True)
    total = sum(counts.values())
    done = 0
    for label in CLASSES:
        for path in paths_by_class[label]:
            digest = sha256_file(path)
            prior = seen.get(digest)
            relative = str(path.relative_to(PROJECT_ROOT))
            if prior is not None:
                if prior["label"] != label:
                    raise RuntimeError(
                        "An exact duplicate has conflicting labels; review dataset labels "
                        f"before training: {prior['path']} ({prior['label']}) and {relative} ({label})."
                    )
                duplicate_rows.append({
                    "duplicate_file": relative,
                    "kept_file": prior["path"],
                    "sha256": digest,
                    "label": label,
                })
            else:
                item = {"path": str(path), "relative_path": relative, "label": label, "sha256": digest}
                seen[digest] = item
                records.append(item)
            done += 1
            if done % 250 == 0 or done == total:
                print(f"  Hashed {done}/{total} images", flush=True)

    result = {
        "images_before_deduplication": total,
        "duplicate_occurrences": len(duplicate_rows),
        "unique_images_used": len(records),
        "duplicate_files": duplicate_rows,
    }
    write_json(REPORTS / "final_resnet_deduplication.json", result)
    print(f"Images before deduplication: {total}", flush=True)
    print(f"Duplicate occurrences: {len(duplicate_rows)}", flush=True)
    print(f"Unique images used: {len(records)}", flush=True)
    return records, result


def stratified_split(records: list[dict[str, str]]):
    frame = pd.DataFrame(records)
    train, holdout = train_test_split(
        frame, test_size=0.20, random_state=SEED, stratify=frame["label"]
    )
    validation, test = train_test_split(
        holdout, test_size=0.50, random_state=SEED, stratify=holdout["label"]
    )
    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    counts = {
        split: {label: int((df["label"] == label).sum()) for label in CLASSES}
        for split, df in splits.items()
    }
    table = []
    for label in CLASSES:
        table.append({
            "class": label,
            "total": int((frame["label"] == label).sum()),
            "train": counts["train"][label],
            "validation": counts["validation"][label],
            "test": counts["test"][label],
        })
    print("\nClass | Total | Train | Validation | Test", flush=True)
    for row in table:
        print(
            f"{row['class']} | {row['total']} | {row['train']} | "
            f"{row['validation']} | {row['test']}", flush=True
        )
    write_json(REPORTS / "final_resnet_split.json", {
        "seed": SEED,
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "class_order": CLASSES,
        "counts": counts,
        "table": table,
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CLASSES))
    width = 0.25
    for i, split in enumerate(("train", "validation", "test")):
        ax.bar(x + (i - 1) * width, [counts[split][c] for c in CLASSES], width, label=split.title())
    ax.set_xticks(x, CLASSES, rotation=15, ha="right")
    ax.set_ylabel("Unique images")
    ax.set_title("Stratified Dataset Split Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "dataset_split_distribution.png", dpi=300)
    plt.close(fig)
    return splits, counts, table


class ImageRecords(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASSES.index(row["label"])


def save_dataset_distribution(counts: dict[str, int]) -> None:
    display_names = ["Discharge Summary", "Lab Report", "Prescription", "Non-Medical"]
    values = [counts[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(display_names, values, color=["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"])
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_ylabel("Image count")
    ax.set_title("Final Dataset Class Distribution")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(REPORTS / "resnet_dataset_distribution.png", dpi=300)
    plt.close(fig)


def plot_training(history: list[dict[str, float]]) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(REPORTS / "final_resnet_training_history.csv", index=False)
    for kind, train_col, val_col, ylabel, title, filename in (
        ("accuracy", "train_accuracy", "validation_accuracy", "Accuracy", "Training and Validation Accuracy", "resnet_training_accuracy.png"),
        ("loss", "train_loss", "validation_loss", "Loss", "Training and Validation Loss", "resnet_training_loss.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(frame["epoch"], frame[train_col], marker="o", linewidth=2, label="Train")
        ax.plot(frame["epoch"], frame[val_col], marker="o", linewidth=2, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(frame["epoch"])
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORTS / filename, dpi=300)
        plt.close(fig)


def save_test_artifacts(y_true: list[int], y_pred: list[int]):
    labels = list(range(len(CLASSES)))
    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "test_sample_count": len(y_true),
    }
    write_json(REPORTS / "final_resnet_test_metrics.json", metrics)
    print("\nTEST RESULTS (held-out test set only)", flush=True)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}", flush=True)

    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=CLASSES, zero_division=0, digits=4
    )
    (REPORTS / "final_resnet_classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(report_dict).T.to_csv(REPORTS / "final_resnet_classification_report.csv", index_label="class")
    print("\nClassification report:\n" + report_text, flush=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        REPORTS / "final_resnet_confusion_matrix.csv", index_label="true\\predicted"
    )
    print("Confusion matrix class order:", CLASSES, flush=True)
    print(cm, flush=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(labels, CLASSES, rotation=25, ha="right")
    ax.set_yticks(labels, CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Held-Out Test Confusion Matrix")
    cutoff = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cutoff else "black", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS / "resnet_confusion_matrix.png", dpi=300)
    plt.close(fig)

    per_class = {label: report_dict[label] for label in CLASSES}
    values = {
        metric: [per_class[label][metric] for label in CLASSES]
        for metric in ("precision", "recall", "f1-score")
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CLASSES))
    width = 0.24
    for i, (metric, color) in enumerate(zip(values, ("#4472C4", "#ED7D31", "#70AD47"))):
        ax.bar(x + (i - 1) * width, values[metric], width, label=metric.replace("-score", "").title(), color=color)
    ax.set_xticks(x, CLASSES, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Held-Out Test Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "resnet_per_class_metrics.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axis("off")
    ax.text(0.5, 0.9, "ResNet-18 — Final Test Results", ha="center", fontsize=22, weight="bold")
    for i, (name, value) in enumerate((
        ("Accuracy", accuracy), ("Macro Precision", macro_p),
        ("Macro Recall", macro_r), ("Macro F1", macro_f1),
    )):
        ax.text(0.15, 0.72 - i * 0.15, f"{name}: {value * 100:.2f}%", fontsize=18)
    fig.tight_layout()
    fig.savefig(REPORTS / "resnet_final_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return metrics, report_dict, cm.tolist()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_OUTPUT.resolve() == V2_MODEL.resolve():
        raise RuntimeError("Safety stop: v3 output resolves to the protected v2 checkpoint path.")
    set_seeds(SEED)
    paths_by_class, counts = audit_dataset()
    records, dedup = deduplicate(paths_by_class, counts)
    splits, split_counts, split_table = stratified_split(records)
    save_dataset_distribution(counts)

    # Show completed checks and exact split prior to downloading model weights or training.
    print("\nPre-training checks complete. Count audit, PIL checks, hashes, split report and charts are saved.", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU run)'}", flush=True)
    print("Beginning 10 visible training epochs...", flush=True)

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    workers = 0
    train_loader = DataLoader(ImageRecords(splits["train"], train_transform), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(ImageRecords(splits["validation"], eval_transform), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(ImageRecords(splits["test"], eval_transform), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())

    weights = compute_class_weight(
        class_weight="balanced", classes=np.array(CLASSES), y=splits["train"]["label"].to_numpy()
    )
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    class_weight_map = {label: float(weight) for label, weight in zip(CLASSES, weights)}
    print("Class weights (training set only):", json.dumps(class_weight_map), flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    config = {
        "architecture": "torchvision.models.resnet18",
        "weights": "torchvision.models.ResNet18_Weights.DEFAULT",
        "classes": CLASSES,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "image_size": list(IMAGE_SIZE),
        "loss": "weighted CrossEntropyLoss; weights computed from training split only",
        "device": str(device),
        "pytorch_version": torch.__version__,
    }
    best_accuracy = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_seen = 0
        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            n = targets.size(0)
            train_loss_sum += loss.item() * n
            train_correct += (logits.argmax(1) == targets).sum().item()
            train_seen += n
            if batch_idx % 25 == 0 or batch_idx == len(train_loader):
                print(f"  Epoch {epoch}/{EPOCHS}: train batch {batch_idx}/{len(train_loader)}", flush=True)

        train_loss = train_loss_sum / train_seen
        train_accuracy = train_correct / train_seen
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                n = targets.size(0)
                val_loss_sum += loss.item() * n
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_seen += n
        validation_loss = val_loss_sum / val_seen
        validation_accuracy = val_correct / val_seen
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        print(f"Epoch {epoch}/{EPOCHS}", flush=True)
        print(f"Train Loss: {train_loss:.6f}", flush=True)
        print(f"Train Accuracy: {train_accuracy:.6f}", flush=True)
        print(f"Validation Loss: {validation_loss:.6f}", flush=True)
        print(f"Validation Accuracy: {validation_accuracy:.6f}", flush=True)
        print(f"Epoch elapsed: {(time.time() - epoch_started) / 60:.1f} min", flush=True)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            temporary = MODEL_OUTPUT.with_suffix(MODEL_OUTPUT.suffix + ".tmp")
            torch.save({"state_dict": state, "class_mapping": {c: i for i, c in enumerate(CLASSES)},
                        "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
                        "configuration": config}, temporary)
            os.replace(temporary, MODEL_OUTPUT)
            write_json(REPORTS / "final_resnet_best_model.json", {
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "model_configuration": config,
                "class_mapping": {c: i for i, c in enumerate(CLASSES)},
            })
        # Keep history artifacts current if execution is interrupted.
        pd.DataFrame(history).to_csv(REPORTS / "final_resnet_training_history.csv", index=False)

    plot_training(history)
    print("\nLoading best validation-accuracy checkpoint for the held-out test evaluation.", flush=True)
    checkpoint = torch.load(MODEL_OUTPUT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device, non_blocking=True))
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    test_metrics, report_dict, cm = save_test_artifacts(y_true, y_pred)

    full_report = {
        "dataset": {
            "root": str(DATA_ROOT),
            "class_directories": CLASS_DIRS,
            "counts_before_deduplication": counts,
            "total_before_deduplication": sum(counts.values()),
            "unique_images_after_sha256_deduplication": len(records),
            "lab_external_provenance_note": (
                "The 90 lab_external images have source IDs, PDF names, pages and matching hashes in the project expansion report; "
                "original GDC PDFs were not locally available for independent page comparison."
            ),
        },
        "deduplication": {k: v for k, v in dedup.items() if k != "duplicate_files"},
        "split": {"seed": SEED, "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                  "counts": split_counts, "table": split_table},
        "class_weights_train_only": class_weight_map,
        "model_configuration": config,
        "training_history": history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test_metrics": test_metrics,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "confusion_matrix_class_order": CLASSES,
        "checkpoint": str(MODEL_OUTPUT.relative_to(PROJECT_ROOT)),
    }
    write_json(REPORTS / "final_resnet_experiment_report.json", full_report)

    print("\n==================================================", flush=True)
    print("FINAL RESNET-18 EXPERIMENT COMPLETE", flush=True)
    print("==================================================", flush=True)
    print("Dataset: 1845 expected images", flush=True)
    print(f"Best Epoch: {best_epoch}", flush=True)
    print(f"Best Validation Accuracy: {best_accuracy:.6f}", flush=True)
    print("TEST RESULTS", flush=True)
    print("------------", flush=True)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"{key.replace('_', ' ').title()}: {test_metrics[key]:.6f}", flush=True)
    pngs = sorted(p.name for p in REPORTS.glob("*.png") if p.name in {
        "dataset_split_distribution.png", "resnet_training_accuracy.png", "resnet_training_loss.png",
        "resnet_confusion_matrix.png", "resnet_per_class_metrics.png", "resnet_dataset_distribution.png",
        "resnet_final_results.png",
    })
    report_files = sorted(p.name for p in REPORTS.iterdir() if p.is_file() and (
        p.name.startswith("final_resnet_") or p.name in {"dataset_split_distribution.png"}
    ))
    print("Generated PPT images:", flush=True)
    for name in pngs:
        print(f"  reports/{name}", flush=True)
    print("Generated reports:", flush=True)
    for name in report_files:
        print(f"  reports/{name}", flush=True)
    print("Model:", MODEL_OUTPUT.relative_to(PROJECT_ROOT), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"EXPERIMENT STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
