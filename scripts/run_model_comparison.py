"""Frozen-split image classifier benchmark. Run with --models resnet50 first.

The benchmark reads the existing split manifest and image files only. It writes
all new artifacts under reports/model_comparison and models/model_comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(os.environ.get("MODEL_COMPARISON_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
MANIFEST = ROOT / "data/splits/final_reproducible_v1/split_manifest.csv"
SPLIT_ROOT = ROOT / "data/splits/final_reproducible_v1"
OUT = Path(os.environ.get("MODEL_COMPARISON_OUTPUT_ROOT", ROOT / "reports/model_comparison"))
CKPT = OUT / "checkpoints"
CLASSES = ["Discharge_Summary", "Lab_Report", "Prescription", "Non_Medical"]
SEED, EPOCHS, BATCH_SIZE, LR, SIZE = 42, 10, 16, 1e-4, 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ARCHS = {
    "resnet50": ("resnet50", models.ResNet50_Weights.DEFAULT),
    "efficientnet_b0": ("efficientnet_b0", models.EfficientNet_B0_Weights.DEFAULT),
    "mobilenet_v3_large": ("mobilenet_v3_large", models.MobileNet_V3_Large_Weights.DEFAULT),
}


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def records_from_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Frozen split manifest missing: {MANIFEST}")
    df = pd.read_csv(MANIFEST).fillna("")
    required = {"split", "class", "filename", "split_path", "sha256"}
    if not required.issubset(df.columns): raise ValueError(f"Manifest missing {required-set(df.columns)}")
    if set(df["class"]) != set(CLASSES): raise ValueError("Manifest class set differs from required four classes")
    for _, row in df.iterrows():
        p = ROOT / row.split_path
        if not p.is_file(): raise FileNotFoundError(f"Manifest image missing: {p}")
        if sha(p) != row.sha256: raise ValueError(f"SHA-256 mismatch: {p}")
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "validation", "test")}


class ManifestImages(Dataset):
    def __init__(self, frame, transform): self.frame, self.transform = frame, transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(ROOT / row.split_path) as im: x = self.transform(im.convert("RGB"))
        return x, CLASSES.index(row["class"]), row.filename


def build(name, weights):
    ctor = getattr(models, ARCHS[name][0])
    model = ctor(weights=weights)
    if name == "resnet50": model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    elif name == "efficientnet_b0": model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    else: model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    return model


def run_one(name, device, records):
    seed_everything()
    train_t = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.05),
        transforms.RandomAffine(degrees=0, translate=(.05, .05)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_t = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    loaders = {s: DataLoader(ManifestImages(records[s], train_t if s == "train" else eval_t), batch_size=BATCH_SIZE,
        shuffle=s == "train", num_workers=0, pin_memory=device.type == "cuda") for s in records}
    weights = ARCHS[name][1]
    model = build(name, weights).to(device)
    y_train = records["train"]["class"].tolist()
    counts = np.array([y_train.count(c) for c in CLASSES], dtype=np.float32)
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start = time.perf_counter(); best_acc, best_epoch, history = -1., 0, []
    model_dir, out_dir = CKPT / name, OUT / name
    model_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train(); loss_sum = correct = seen = 0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
            z = model(x); loss = criterion(z, y); loss.backward(); opt.step()
            loss_sum += loss.item() * y.size(0); correct += (z.argmax(1) == y).sum().item(); seen += y.size(0)
        train_loss, train_acc = loss_sum / seen, correct / seen
        model.eval(); val_loss = val_correct = val_seen = 0
        with torch.no_grad():
            for x, y, _ in loaders["validation"]:
                x, y = x.to(device), y.to(device); z = model(x); loss = criterion(z, y)
                val_loss += loss.item() * y.size(0); val_correct += (z.argmax(1) == y).sum().item(); val_seen += y.size(0)
        val_loss /= val_seen; val_acc = val_correct / val_seen
        history.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc, validation_loss=val_loss, validation_accuracy=val_acc))
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"state_dict": model.state_dict(), "class_order": CLASSES, "best_epoch": epoch}, model_dir / "best.pth")
        print(f"{name} epoch {epoch}/{EPOCHS}: val_acc={val_acc:.4f} best={best_acc:.4f}", flush=True)
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(model_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    y_true, y_pred, filenames = [], [], []
    if device.type == "cuda": torch.cuda.synchronize()
    infer_start = time.perf_counter()
    with torch.no_grad():
        for x, y, files in loaders["test"]:
            z = model(x.to(device)); y_pred.extend(z.argmax(1).cpu().tolist()); y_true.extend(y.tolist()); filenames.extend(files)
    if device.type == "cuda": torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - infer_start
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report.csv", index_label="class")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "confusion_matrix.csv", index_label="true\\predicted")
    fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(4), CLASSES, rotation=25, ha="right"); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{name} test confusion matrix")
    for i in range(4):
        for j in range(4): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=200); plt.close(fig)
    pd.DataFrame({"filename": filenames, "true_class": [CLASSES[i] for i in y_true], "predicted_class": [CLASSES[i] for i in y_pred]}).to_csv(out_dir / "predictions.csv", index=False)
    per_ms = inference_seconds / len(y_true) * 1000
    batch = torch.randn(1, 3, SIZE, SIZE, device=device)
    row = {"model": name, "status": "measured", "test_accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"], "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"],
        "prescription_precision": report["Prescription"]["precision"], "prescription_recall": report["Prescription"]["recall"], "prescription_f1": report["Prescription"]["f1-score"],
        "best_epoch": best_epoch, "best_validation_accuracy": best_acc, "training_seconds": training_seconds,
        "test_inference_total_seconds": inference_seconds, "inference_ms_per_image": per_ms, "model_size_mb_state_dict_params": model_bytes / 1e6,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "checkpoint": str(model_dir / "best.pth")}
    (out_dir / "run_metadata.json").write_text(json.dumps({**row, "seed": SEED, "pretrained_weights": str(weights), "input_dimensions": [3, SIZE, SIZE],
       "optimizer": "Adam", "learning_rate": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
       "augmentation": "Resize 224; RandomRotation(5); ColorJitter(.15,.15,.15,.05); RandomAffine translate .05; ImageNet normalization",
       "pytorch": torch.__version__, "torchvision": torchvision.__version__, "sklearn": sklearn.__version__, "test_images": len(y_true)}, indent=2), encoding="utf-8")
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", choices=["resnet50", "efficientnet_b0", "mobilenet_v3_large"], default=["resnet50"])
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    records = records_from_manifest()
    print("Verified frozen split counts:", {s: {c: int((d["class"] == c).sum()) for c in CLASSES} for s, d in records.items()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_path = OUT / "model_comparison.csv"
    old = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    results = [r for r in old if r.get("model") not in args.models]
    for name in args.models: results.append(run_one(name, device, records))
    # User-provided baseline preserved exactly and explicitly not recomputed here.
    results.append({"model": "ResNet-18 (provided baseline)", "status": "user-provided; not rerun", "test_accuracy": .9837,
        "macro_precision": .9869, "macro_recall": .9736, "macro_f1": .9798, "prescription_precision": None,
        "prescription_recall": None, "prescription_f1": None, "training_seconds": None, "inference_ms_per_image": None,
        "model_size_mb_state_dict_params": None, "peak_gpu_memory_mb": None, "device": None, "checkpoint": None})
    pd.DataFrame(results).drop_duplicates(subset=["model"], keep="last").to_csv(results_path, index=False)
    summary = pd.DataFrame(results).drop_duplicates(subset=["model"], keep="last")
    ppt_cols = ["model", "status", "test_accuracy", "macro_precision", "macro_recall", "macro_f1",
                "prescription_recall", "training_seconds", "inference_ms_per_image", "model_size_mb_state_dict_params", "peak_gpu_memory_mb"]
    summary[[c for c in ppt_cols if c in summary]].to_csv(OUT / "ppt_ready_comparison_table.csv", index=False)
    (OUT / "comparison_interpretation.md").write_text(
        "# Model comparison interpretation\n\n"
        "Only rows marked `measured` are experiments from this comparison run. The ResNet-18 row is the supplied baseline and was not rerun; runtime, model size, and per-class results are therefore blank. "
        "Compare measured architectures using macro metrics, Prescription recall/F1, inference latency, and resource use together. Any claim of a winner must be based on completed measured rows; a one-seed run is descriptive and does not establish statistical significance.\n",
        encoding="utf-8")
    print("Updated", results_path)


if __name__ == "__main__": main()
