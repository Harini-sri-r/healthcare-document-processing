"""Train/evaluate a separate four-class ResNet-18 checkpoint.

Expected folders below data/raw: handwritten_prescription, discharge_summaries,
lab_reports, non_medical. The original ResNet checkpoint is only read as an
optional architecture reference and is never modified. Run with --help.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import settings


CLASSES = ["Discharge_Summary", "Lab_Report", "Prescription", "Non_Medical"]
CLASS_DIRS = {
    "Discharge_Summary": "discharge_summaries",
    "Lab_Report": "lab_reports",
    "Prescription": "handwritten_prescription",
    "Non_Medical": "non_medical",
}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def duplicate_key(path: Path) -> str:
    """Exact file-content identity; visual/layout similarity is not a duplicate."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_and_split(root: Path, seed: int, ratios=(0.70, 0.15, 0.15)):
    rng = random.Random(seed)
    counts = {}
    class_paths = {}
    for label in CLASSES:
        folder = root / CLASS_DIRS[label]
        items = sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS) if folder.is_dir() else []
        class_paths[label] = items
        counts[label] = len(items)

    # Report and fail before decoding potentially large images.
    print("DATASET COUNTS", flush=True)
    for label in CLASSES:
        print(f"{label}: {counts[label]}", flush=True)
    print(f"Total: {sum(counts.values())}", flush=True)
    missing = [label for label in CLASSES if not counts[label]]
    if missing:
        raise ValueError(
            "Training blocked: missing training images for " + ", ".join(missing)
            + ". Add real images to data/raw/non_medical and rerun."
        )

    print("\nOriginal files:", flush=True)
    for label in CLASSES:
        print(f"{label}: {counts[label]}", flush=True)

    # Keep the first deterministic file for each exact byte hash. This only
    # changes the split candidate list; source files are never deleted.
    class_paths_by_hash: dict[str, dict[str, Path]] = {}
    hash_labels: dict[str, str] = {}
    effective_counts: dict[str, int] = {}
    for label in CLASSES:
        unique: dict[str, Path] = {}
        for path in class_paths[label]:
            digest = duplicate_key(path)
            prior_label = hash_labels.get(digest)
            if prior_label is not None and prior_label != label:
                raise ValueError(
                    f"Identical file content has conflicting labels: {prior_label} and {label}. "
                    "Review the labels before splitting."
                )
            hash_labels[digest] = label
            unique.setdefault(digest, path)
        class_paths_by_hash[label] = unique
        effective_counts[label] = len(unique)

    print("\nAfter exact duplicate removal:", flush=True)
    for label in CLASSES:
        print(f"{label}: {effective_counts[label]}", flush=True)

    # Near-duplicate removal is deliberately disabled: document layout
    # similarity is not reliable evidence that two pages are the same image.
    print("\nAfter near-duplicate filtering (disabled; exact hashes only):", flush=True)
    for label in CLASSES:
        print(f"{label}: {effective_counts[label]}", flush=True)

    for label in CLASSES:
        removed = counts[label] - effective_counts[label]
        removal_fraction = removed / counts[label] if counts[label] else 0.0
        if removal_fraction > 0.20:
            raise ValueError(
                f"Safety stop: exact duplicate filtering removed {removed}/{counts[label]} "
                f"({removal_fraction:.1%}) of {label}. Review the dataset and duplicate "
                "diagnostics; no files were deleted."
            )
        if effective_counts[label] < 3:
            raise ValueError(
                f"{label} has fewer than 3 distinct exact-content images after deduplication; "
                "cannot create train/validation/test splits. No files were deleted."
            )

    class_groups: dict[str, list[list[tuple[str, Path]]]] = defaultdict(list)
    for label in CLASSES:
        class_groups[label] = [[(label, path)] for path in class_paths_by_hash[label].values()]

    partitions = {name: [] for name in ("train", "validation", "test")}
    for label in CLASSES:
        label_groups = class_groups[label]
        rng.shuffle(label_groups)
        n = len(label_groups)
        train_end = max(1, int(n * ratios[0]))
        val_end = max(train_end + 1, int(n * (ratios[0] + ratios[1])))
        if val_end >= n:
            raise ValueError(f"{label} needs at least 3 distinct images to split train/validation/test; found {n}.")
        for split, selected in (("train", label_groups[:train_end]), ("validation", label_groups[train_end:val_end]), ("test", label_groups[val_end:])):
            for group in selected:
                partitions[split].extend(group)
    split_counts = {
        split: {label: sum(1 for record_label, _ in records if record_label == label)
                for label in CLASSES}
        for split, records in partitions.items()
    }
    print("\nSplit counts:", flush=True)
    for split in ("train", "validation", "test"):
        print(split.upper(), flush=True)
        for label in CLASSES:
            print(f"{label}: {split_counts[split][label]}", flush=True)
    return partitions, effective_counts


class ImageRecords(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        label, path = self.records[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), CLASSES.index(label)


def metrics(confusion):
    result = {}
    total = int(confusion.sum())
    for index, label in enumerate(CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[index, :].sum())}
    accuracy = int(confusion.diagonal().sum()) / total if total else 0.0
    macro_precision = sum(item["precision"] for item in result.values()) / len(CLASSES)
    macro_recall = sum(item["recall"] for item in result.values()) / len(CLASSES)
    macro_f1 = sum(item["f1"] for item in result.values()) / len(CLASSES)
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "test_sample_count": total,
        "per_class": result,
        "confusion_matrix": confusion.tolist(),
    }


def run(args):
    partitions, dataset_counts = collect_and_split(args.data_root, args.seed)
    if args.split_only:
        print("\nSplit-only mode: training was not started.", flush=True)
        return
    if args.output.resolve() == settings.RESNET_MODEL_PATH.resolve():
        raise ValueError("Refusing to overwrite the original 3-class checkpoint.")
    if str(args.device or settings.DEVICE).startswith("cpu"):
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting inter-op threads only before parallel work.
            pass
        print(f"CPU threads: intra_op={torch.get_num_threads()}, inter_op=1", flush=True)
    torch.manual_seed(args.seed)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(ImageRecords(partitions["train"], train_tf), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ImageRecords(partitions["validation"], eval_tf), batch_size=args.batch_size)
    test_loader = DataLoader(ImageRecords(partitions["test"], eval_tf), batch_size=args.batch_size)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    device = torch.device(args.device or settings.DEVICE)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    epoch_history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_count += labels.size(0)
        train_loss /= max(1, train_count)
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                batch_loss = criterion(model(images), labels)
                val_loss += batch_loss.item() * labels.size(0)
                val_count += labels.size(0)
        val_loss /= max(1, val_count)
        epoch_history.append({"epoch": epoch + 1, "training_loss": train_loss, "validation_loss": val_loss})
        print(f"epoch={epoch + 1} training_loss={train_loss:.6f} validation_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")

    model.load_state_dict(best_state)
    model.eval()
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in test_loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            for truth, guess in zip(labels, predicted):
                confusion[truth, guess] += 1
    report = metrics(confusion)
    report["dataset_counts"] = dataset_counts
    report["split_counts"] = {split: dict(Counter(label for label, _ in records)) for split, records in partitions.items()}
    report["epoch_losses"] = epoch_history
    report["training"] = {"seed": args.seed, "epochs": args.epochs, "best_epoch": best_epoch,
                          "best_validation_loss": best_loss, "pretrained": "ImageNet ResNet18_Weights.DEFAULT",
                          "evaluation_label": "held-out dataset performance"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "classes": CLASSES,
        "class_to_idx": {label: index for index, label in enumerate(CLASSES)},
        "architecture": "torchvision.models.resnet18",
        "architecture_kwargs": {"num_classes": len(CLASSES)},
        "training_metadata": report["training"],
        "dataset_counts": dataset_counts,
        "split_counts": report["split_counts"],
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary_output)
    os.replace(temporary_output, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"checkpoint={args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=settings.RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=settings.RESNET_V2_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=settings.REPORTS_PATH / "resnet18_v2_evaluation.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="PyTorch CPU intra-op thread count (default: 4).")
    parser.add_argument("--split-only", action="store_true",
                        help="Create and report the dataset split, then exit without training.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
