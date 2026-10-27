"""Train and report FINAL REPRODUCIBLE RESNET-18 EXPERIMENT v1.

The old v3 experiment artifacts are never read as results or overwritten here.
The frozen split manifest is checked against the current readable dataset before
any training starts. Exact duplicate files remain in data/raw and are excluded
from split candidates by whole-file SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix,
    precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits" / "final_reproducible_v1"
REPORTS = ROOT / "reports"
CHECKPOINT = ROOT / "models" / "medical_document_resnet18_reproducible_v1.pth"
CLASSES = ["Discharge_Summary", "Lab_Report", "Prescription", "Non_Medical"]
DIRS = {"Discharge_Summary":"discharge_summaries", "Lab_Report":"lab_reports",
        "Prescription":"handwritten_prescription", "Non_Medical":"non_medical"}
DISPLAY = {"Discharge_Summary":"Discharge Summary", "Lab_Report":"Lab Report",
           "Prescription":"Prescription", "Non_Medical":"Non-Medical"}
EXPECTED = dict(zip(CLASSES, [500, 516, 129, 700]))
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SEED, EPOCHS, BATCH, LR, SIZE = 42, 10, 16, 1e-4, (224, 224)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def audit_and_manifest():
    raw, hash_first, duplicate_rows = [], {}, []
    counts = {}
    for cls in CLASSES:
        paths = sorted(p for p in (DATA / DIRS[cls]).rglob("*")
                       if p.is_file() and p.suffix.lower() in EXTS)
        counts[cls] = len(paths)
        for p in paths:
            try:
                with Image.open(p) as im: im.verify()
            except Exception as e:
                raise RuntimeError(f"Unreadable image {p}: {e}")
            digest = sha(p)
            row = {"class": cls, "filename": p.name,
                   "relative_path": p.relative_to(ROOT).as_posix(), "sha256": digest}
            if digest in hash_first:
                prior = hash_first[digest]
                if prior["class"] != cls:
                    raise RuntimeError(f"Duplicate bytes have conflicting labels: {prior} / {row}")
                duplicate_rows.append({**row, "duplicate_of": prior["relative_path"]})
            else:
                hash_first[digest] = row
                raw.append(row)
    if counts != EXPECTED:
        raise RuntimeError(f"Dataset counts differ from requested verified totals: {counts}")
    if len(raw) != 1835 or len(duplicate_rows) != 10:
        raise RuntimeError(f"Expected 1835 unique and 10 duplicate occurrences; got {len(raw)} and {len(duplicate_rows)}")

    # Recreate the deterministic split in memory and require it to match the
    # permanent manifest before trusting the materialized files.
    frame = pd.DataFrame(raw)
    tr, hold = train_test_split(frame, test_size=.20, random_state=SEED, stratify=frame["class"])
    va, te = train_test_split(hold, test_size=.50, random_state=SEED, stratify=hold["class"])
    expected_frames = {"train":tr, "validation":va, "test":te}
    expected_keys = set()
    for split, part in expected_frames.items():
        expected_keys.update((split, r["class"], r["relative_path"], r["sha256"])
                             for r in part.to_dict("records"))
    manifest_path = SPLITS / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str)
    actual_keys = set(map(tuple, manifest[["split","class","relative_path","sha256"]].itertuples(index=False, name=None)))
    if actual_keys != expected_keys or len(manifest) != 1835:
        raise RuntimeError("Permanent split manifest does not match the deterministic current dataset split")
    # Check each source and frozen copy, plus the requested exact directory layout.
    for row in manifest.to_dict("records"):
        src = ROOT / Path(row["relative_path"])
        frozen = SPLITS / row["split"] / row["class"] / row["filename"]
        if not src.is_file() or sha(src) != row["sha256"]:
            raise RuntimeError(f"Manifest source changed or missing: {src}")
        if not frozen.is_file() or sha(frozen) != row["sha256"]:
            raise RuntimeError(f"Frozen split copy missing or changed: {frozen}")
    if manifest.sha256.duplicated().any() or manifest.relative_path.duplicated().any():
        raise RuntimeError("Manifest contains cross-split duplicate hashes or repeated source files")
    for split in ("train", "validation", "test"):
        for cls in CLASSES:
            folder = SPLITS / split / cls
            if not folder.is_dir(): raise RuntimeError(f"Missing split directory: {folder}")
            if len(list(folder.iterdir())) != int(((manifest.split == split) & (manifest["class"] == cls)).sum()):
                raise RuntimeError(f"Unexpected files in {folder}")
    return raw, duplicate_rows, counts, manifest

class Records(Dataset):
    def __init__(self, frame, transform): self.frame, self.transform = frame.reset_index(drop=True), transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        r = self.frame.iloc[i]
        with Image.open(SPLITS / r["split"] / r["class"] / r["filename"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, CLASSES.index(r["class"])

def evaluate(model, loader, criterion, device):
    model.eval(); loss_sum = 0.; y, pred = [], []
    with torch.no_grad():
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            logits = model(x); loss_sum += criterion(logits, target).item() * len(target)
            y.extend(target.cpu().tolist()); pred.extend(logits.argmax(1).cpu().tolist())
    labels = list(range(len(CLASSES)))
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted", zero_division=0)
    pc_p, pc_r, pc_f, support = precision_recall_fscore_support(y, pred, labels=labels, average=None, zero_division=0)
    return {"Dataset_Size":len(y), "Loss":loss_sum/len(y), "Accuracy":accuracy_score(y,pred),
            "Macro_Precision":mp,"Macro_Recall":mr,"Macro_F1":mf,
            "Weighted_Precision":wp,"Weighted_Recall":wr,"Weighted_F1":wf,
            "per_class":{c:{"Precision":float(pc_p[i]),"Recall":float(pc_r[i]),"F1":float(pc_f[i]),"Support":int(support[i])} for i,c in enumerate(CLASSES)},
            "y_true":y,"y_pred":pred}

def main():
    REPORTS.mkdir(exist_ok=True); CHECKPOINT.parent.mkdir(exist_ok=True)
    if CHECKPOINT.exists(): raise FileExistsError(f"Refusing to overwrite existing new-experiment checkpoint: {CHECKPOINT}")
    seed_everything()
    raw, duplicates, raw_counts, manifest = audit_and_manifest()
    split_counts = {s:{c:int(((manifest.split==s)&(manifest["class"]==c)).sum()) for c in CLASSES}
                    for s in ("train","validation","test")}
    split_sizes = {s:sum(split_counts[s].values()) for s in split_counts}
    pd.DataFrame(raw).to_csv(REPORTS / "reproducible_v1_dataset_sha256.csv", index=False)
    save_json(REPORTS / "reproducible_v1_duplicates.json", {"duplicate_occurrences":10,"duplicates":duplicates})
    proof = ["FINAL REPRODUCIBLE RESNET-18 EXPERIMENT", "", "Dataset:",
      "1,845 readable images", "1,835 unique images", "10 duplicate occurrences", "",
      "Classes:"] + [f"{DISPLAY[c]} = {raw_counts[c]}" for c in CLASSES] + ["",
      "SHA-256 hashes were calculated over every readable image file. The 10 duplicate occurrences remain in data/raw but were excluded from the split so the same exact image cannot appear across train/validation/test.", "",
      "Exact deterministic split counts (unique images):"]
    proof += [f"{s.title()} = {split_sizes[s]} ({', '.join(f'{DISPLAY[c]} {split_counts[s][c]}' for c in CLASSES)})" for s in split_counts]
    proof += ["", "The exact membership of the previous experiment's 1468/183/184 split could not be recovered. Therefore, this experiment uses a newly generated, deterministic stratified split with seed 42."]
    (REPORTS / "reproducible_v1_dataset_used.txt").write_text("\n".join(proof)+"\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = transforms.Compose([transforms.Resize(SIZE), transforms.RandomRotation(5),
      transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.05),
      transforms.RandomAffine(degrees=0, translate=(.05,.05)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    eval_tf = transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    loaders = {}
    for split, tf, shuffle in (("train",train_tf,True),("validation",eval_tf,False),("test",eval_tf,False)):
        df = manifest[manifest.split==split]
        loaders[split] = DataLoader(Records(df,tf), batch_size=BATCH, shuffle=shuffle, num_workers=0,
                                    pin_memory=torch.cuda.is_available())
    weights = compute_class_weight("balanced", classes=np.array(CLASSES), y=manifest[manifest.split=="train"]["class"].to_numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features,len(CLASSES)); model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    config = {"experiment":"FINAL REPRODUCIBLE RESNET-18 EXPERIMENT","architecture":"ResNet-18",
      "initialization":"ImageNet pretrained torchvision ResNet18_Weights.DEFAULT","classes":CLASSES,
      "input_size":list(SIZE),"optimizer":"Adam","learning_rate":LR,"epochs":EPOCHS,"batch_size":BATCH,
      "seed":SEED,"loss":"class-weighted CrossEntropyLoss; balanced weights computed from training data",
      "class_weights":{c:float(w) for c,w in zip(CLASSES,weights)},
      "augmentation":["Resize(224,224)","RandomRotation(5 degrees)","ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.05)","RandomAffine(translate=(.05,.05))"],
      "evaluation_transform":["Resize(224,224)","ToTensor","ImageNet normalization"],"imagenet_mean":MEAN,"imagenet_std":STD,
      "device":str(device),"torch_version":torch.__version__,"sklearn_version":sklearn.__version__}
    hist=[]; best_acc=-1.; best_epoch=0
    for epoch in range(1,EPOCHS+1):
        start=time.time(); model.train(); total_loss=0.; correct=seen=0
        for x,y in loaders["train"]:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
            total_loss += loss.item()*len(y); correct += (logits.argmax(1)==y).sum().item(); seen += len(y)
        train_loss=total_loss/seen; train_acc=correct/seen
        val=evaluate(model,loaders["validation"],criterion,device)
        row={"Epoch":epoch,"Train Loss":train_loss,"Train Accuracy":train_acc,
             "Validation Loss":val["Loss"],"Validation Accuracy":val["Accuracy"]}
        hist.append(row); pd.DataFrame(hist).to_csv(REPORTS/"reproducible_v1_training_history.csv",index=False)
        print(f"Epoch {epoch}/{EPOCHS} Train Loss {train_loss:.6f} Train Accuracy {train_acc:.6f} Validation Loss {val['Loss']:.6f} Validation Accuracy {val['Accuracy']:.6f} elapsed {(time.time()-start)/60:.1f} min",flush=True)
        if val["Accuracy"] > best_acc:
            best_acc,best_epoch=val["Accuracy"],epoch
            torch.save({"state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
              "class_mapping":{c:i for i,c in enumerate(CLASSES)},"best_epoch":epoch,
              "best_validation_accuracy":best_acc,"configuration":config},CHECKPOINT)
    # Curves
    histdf=pd.DataFrame(hist)
    for filename, cols, ylabel, title in (
      ("reproducible_v1_training_accuracy.png",("Train Accuracy","Validation Accuracy"),"Accuracy","Training and Validation Accuracy"),
      ("reproducible_v1_training_loss.png",("Train Loss","Validation Loss"),"Loss","Training and Validation Loss")):
        fig,ax=plt.subplots(figsize=(8,5)); ax.plot(histdf.Epoch,histdf[cols[0]],marker="o",label="Training")
        ax.plot(histdf.Epoch,histdf[cols[1]],marker="o",label="Validation"); ax.set(xlabel="Epoch",ylabel=ylabel,title=title)
        ax.set_xticks(histdf.Epoch); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(REPORTS/filename,dpi=300); plt.close(fig)
    checkpoint=torch.load(CHECKPOINT,map_location=device,weights_only=False); model.load_state_dict(checkpoint["state_dict"])
    result={}; metrics_rows=[]
    for split, title in (("train","Training"),("validation","Validation"),("test","Testing")):
        ev=evaluate(model,loaders[split],criterion,device)
        result[title]={k:v for k,v in ev.items() if k not in ("y_true","y_pred")}
        metrics_rows.append({"Split":title,**{k:ev[k] for k in ("Dataset_Size","Loss","Accuracy","Macro_Precision","Macro_Recall","Macro_F1","Weighted_Precision","Weighted_Recall","Weighted_F1")}})
        cm=confusion_matrix(ev["y_true"],ev["y_pred"],labels=range(len(CLASSES)))
        pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(REPORTS/f"reproducible_v1_{split}_confusion_matrix.csv",index_label="Actual / Predicted")
        fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(cm,cmap="Blues"); fig.colorbar(im,ax=ax)
        ax.set_xticks(range(4),CLASSES,rotation=25,ha="right"); ax.set_yticks(range(4),CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{title} Confusion Matrix")
        for i in range(4):
            for j in range(4): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
        fig.tight_layout(); fig.savefig(REPORTS/f"reproducible_v1_{split}_confusion_matrix.png",dpi=300); plt.close(fig)
    (REPORTS/"reproducible_v1_metrics.csv").write_text(pd.DataFrame(metrics_rows).to_csv(index=False),encoding="utf-8")
    result.update({"experiment":"FINAL REPRODUCIBLE RESNET-18 EXPERIMENT","dataset":{"readable":1845,"unique":1835,"duplicate_occurrences":10,"class_counts":raw_counts},
      "split":{"seed":SEED,"proportions":{"train":.8,"validation":.1,"test":.1},"counts":split_counts,"sizes":split_sizes},
      "best_epoch":best_epoch,"best_validation_accuracy":best_acc,"configuration":config,
      "checkpoint":CHECKPOINT.relative_to(ROOT).as_posix(),"manifest":(SPLITS/"split_manifest.csv").relative_to(ROOT).as_posix()})
    save_json(REPORTS/"reproducible_v1_metrics.json",result)
    # Full self-contained methodology report.
    lines=["FINAL REPRODUCIBLE RESNET-18 EXPERIMENT", "",
      "1. Dataset", "Current readable dataset: 1,845 images (Discharge Summary 500; Lab Report 516; Prescription 129; Non-Medical 700). SHA-256 identifies 1,835 unique images and 10 duplicate occurrences.",
      "", "2. Dataset cleaning", "All counted images passed PIL image verification. data/raw/non_medical/.gitkeep was excluded because it is not an image.",
      "", "3. Duplicate handling", "Whole-file SHA-256 duplicates were retained in data/raw and excluded from split candidates. No duplicate hash appears across splits.",
      "", "4. Train/validation/test split", f"Stratified split, seed {SEED}; exact totals: training {split_sizes['train']}, validation {split_sizes['validation']}, testing {split_sizes['test']}. Per-class: {json.dumps(split_counts)}.",
      "The exact membership of the previous experiment's 1468/183/184 split could not be recovered. This is a newly generated deterministic split.",
      "", "5. Model architecture", "Torchvision ResNet-18 initialized with ImageNet pretrained weights; final fully connected layer outputs four classes.",
      "", "6. Hyperparameters", json.dumps(config,indent=2), "", "7. Training results", json.dumps(hist,indent=2)]
    for title, section in (("Validation results","Validation"),("Testing results","Testing"),("Training results on selected checkpoint","Training")):
        lines += ["", f"8. {title}" if title=="Validation results" else f"9. {title}" if title=="Testing results" else title, json.dumps(result[section],indent=2)]
    lines += ["", "10. Per-class metrics"]
    for s in ("Training","Validation","Testing"): lines += [s, json.dumps(result[s]["per_class"],indent=2)]
    lines += ["", "11. Confusion matrix", "CSV and PNG confusion matrices are stored separately for training, validation and testing under reports/reproducible_v1_*_confusion_matrix.*.",
      "", "12. Best epoch", f"Epoch {best_epoch}; validation accuracy {best_acc:.6f}.",
      "", "13. Limitations", "This experiment uses a newly reconstructed split because previous split membership was unavailable. Metrics are specific to this dataset and split; duplicate matching only detects byte-identical files.",
      "", "14. Reproducibility information", f"Seed 42; frozen manifest {SPLITS.relative_to(ROOT).as_posix()}/split_manifest.csv; dataset SHA-256 inventory reports/reproducible_v1_dataset_sha256.csv; checkpoint {CHECKPOINT.relative_to(ROOT).as_posix()}.",
      "", "These results belong to the new reproducible split and must not be represented as the original experiment's results."]
    (REPORTS/"reproducible_v1_complete_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\nFINAL REPRODUCIBLE EXPERIMENT",flush=True)
    print(f"Dataset: Readable: 1845 Unique: 1835 Duplicates: 10",flush=True)
    print(f"Split: Train: {split_sizes['train']} Validation: {split_sizes['validation']} Test: {split_sizes['test']}",flush=True)
    print(f"Best Epoch: {best_epoch}",flush=True)
    for title in ("Training","Validation","Testing"):
        m=result[title]; print(f"\n{title.upper()}\nLoss: {m['Loss']:.6f}\nAccuracy: {m['Accuracy']:.6f}\nMacro Precision: {m['Macro_Precision']:.6f}\nMacro Recall: {m['Macro_Recall']:.6f}\nMacro F1: {m['Macro_F1']:.6f}",flush=True)
    print(f"\nCheckpoint: {CHECKPOINT.relative_to(ROOT).as_posix()}\nManifest: {(SPLITS/'split_manifest.csv').relative_to(ROOT).as_posix()}\nComplete report: {(REPORTS/'reproducible_v1_complete_report.txt').relative_to(ROOT).as_posix()}",flush=True)
    print("These results belong to the new reproducible split and must not be represented as the original experiment's results.",flush=True)

if __name__ == "__main__": main()
