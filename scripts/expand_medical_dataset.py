"""Discover explicitly labelled public document image sources; add only with --add.

Default mode performs source discovery and metadata inspection only. It never
downloads image bytes or writes into data/raw. Adding requires a prior discovery
report and a separate explicit ``--add`` invocation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT_JSON = ROOT / "reports" / "medical_dataset_expansion_report.json"
REPORT_TXT = ROOT / "reports" / "medical_dataset_expansion_report.txt"
MANIFEST_CSV = ROOT / "reports" / "medical_dataset_source_manifest.csv"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
TARGETS = {"handwritten_prescription": 1000, "discharge_summaries": 1000,
           "lab_reports": 1000, "non_medical": 1000}
HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
UA = "healthcare-document-processing-medical-expansion/2.0"
TIMEOUT = (5, 15)

# No source is accepted based on directory/file name. Each candidate needs a
# structured metadata field (or documented per-row label) whose value matches
# the requested document type. Repository cards and file trees are inspected
# first; datasets lacking accessible item-level labels remain rejected.
PROPOSED = {
    "handwritten_prescription": [
        {"repo": "mamun1113/doctors-handwritten-prescription-bd-dataset",
         "candidate_metadata": ["label", "document_type", "category", "class", "annotations"],
         "note": "Inspect for explicit per-image prescription label; folder names alone are not accepted."},
        {"repo": "Nigeria-Health-data-OCR-pipeline/African-Medical-Records",
         "candidate_metadata": ["document_type", "record_type", "category", "metadata.csv"],
         "note": "Mixed handwritten medical records; only rows explicitly marked prescription can qualify."},
        {"repo": "chinmays18/medical-prescription-dataset",
         "candidate_metadata": ["ground_truth", "document_type", "label"],
         "note": "Dataset code describes synthetic handwritten prescriptions; inspect actual item-level labels and page images."},
    ],
    "discharge_summaries": [
        {"repo": "hmnshudhmn24/noisy-medical-document-images-ocr",
         "candidate_metadata": ["document_type", "csv/json manifest"],
         "note": "Card describes synthetic discharge summaries; accept only rows with explicit discharge_summary type."},
        {"repo": "naazimsnh02/medocr-vision-dataset",
         "candidate_metadata": ["document_type", "class", "category", "annotations"],
         "note": "Inspect row labels; do not infer type from the reported dataset section name."},
    ],
    "lab_reports": [
        {"repo": "naazimsnh02/medocr-vision-dataset",
         "candidate_metadata": ["document_type", "class", "category", "annotations"],
         "note": "Accept only explicitly labelled laboratory/pathology/diagnostic laboratory report rows."},
    ],
}

# Alternative sources discovered through their primary dataset/publisher pages.
# Published counts are not represented as verified local candidates unless
# the item-level metadata and file inventory are actually reachable. In
# discovery-only mode this catalogue is used to report the evidence and limits.
ALTERNATIVE_SOURCES = [
    {"name": "Image-to-Text Bilingual Dataset from Medical Prescriptions",
     "url": "https://data.mendeley.com/datasets/tg2hm7n2bs",
     "class": "handwritten_prescription", "published_usable_images": 1000,
     "explicit_labels": "Dataset-level description identifies all 1,000 as handwritten medical prescriptions; annotations.csv provides per-prescription transcription, but image-to-row mapping must be checked after access.",
     "license": "CC BY 4.0", "access": "Publisher listing accessible; downloadable file inventory and exact images not inspected in this phase.",
     "source_type": "real photographed/scanned prescription images"},
    {"name": "OCR for Medical Laboratory Reports / Chinese Medical Documents Dataset",
     "url": "https://xuewenyuan.github.io/datasets/cmdd/",
     "class": "lab_reports", "published_usable_images": 357,
     "explicit_labels": "Dataset documentation identifies 357 images as medical laboratory report images.",
     "license": "Not established from the accessible dataset page; permission/license must be confirmed before use.",
     "access": "Project page accessible; artifact URL, license, file-level inventory, and hashes not inspected in this phase.",
     "source_type": "real laboratory report images, scanner and mobile photos"},
    {"name": "OCR for Medical Laboratory Reports (paper dataset)",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10629084/",
     "class": "lab_reports", "published_usable_images": 238,
     "explicit_labels": "Paper explicitly describes 238 de-identified images of Chinese medical laboratory reports.",
     "license": "Dataset-specific license/access conditions not established; paper open access does not by itself license dataset reuse.",
     "access": "Paper accessible; linked GitHub dataset repository/artifact inventory not inspected in this phase.",
     "source_type": "real lab report pages captured by scanners and smartphones"},
    {"name": "RVL-CDIP",
     "url": "https://adamharley.com/rvl-cdip/",
     "class": "non_medical", "published_usable_images": 200000,
     "explicit_labels": "Official dataset page specifies 16 classes with 25,000 images per class; select only explicit non-medical class IDs such as invoice, resume, letter, memo, receipt-like not present, and other general categories.",
     "license": "Derived from IIT-CDIP / Legacy Tobacco Documents Library; source page points to upstream terms, exact reuse conditions require review.",
     "access": "Dataset page accessible; full image archive is 37 GB, label archive is 6.1 MB. Item-level duplicate comparison not run.",
     "source_type": "real scanned general/business documents"},
    {"name": "Arabic Documents OCR Dataset",
     "url": "https://humansintheloop.org/resources/datasets/arabic-documents-ocr-dataset/",
     "class": "non_medical", "published_usable_images": 10000,
     "explicit_labels": "Publisher identifies 12 image classes including invoices, official documents, receipts, business cards, newspapers, books, and administrative forms.",
     "license": "CC0 1.0", "access": "Dataset description accessible; download requires access form, actual label counts and hashes not checked.",
     "source_type": "real photographed/scanned diverse documents"},
    {"name": "MIMIC-IV-Ext-BHC",
     "url": "https://physionet.org/content/labelled-notes-hospital-course/1.2.0/",
     "class": "discharge_summaries", "published_usable_images": 0,
     "explicit_labels": "Explicitly discharge-summary clinical notes, but it contains text notes rather than page images.",
     "license": "PhysioNet credentialed access and data use agreement; not assessed for image conversion.",
     "access": "Dataset page accessible; rejected for this image-classification corpus because the source items are text, not images.",
     "source_type": "text-only clinical notes", "rejected": True},
    {"name": "Noisy Medical Document Images (OCR)",
     "url": "https://huggingface.co/datasets/hmnshudhmn24/noisy-medical-document-images-ocr",
     "class": "discharge_summaries", "published_usable_images": 500,
     "explicit_labels": "Dataset card claims 500 synthetic discharge summary images with document_type metadata.",
     "license": "Not established in this discovery; synthetic data not permitted without user approval.",
     "access": "Hugging Face was blocked in the runtime; excluded both as inaccessible and synthetic without approval.",
     "source_type": "synthetic discharge pages", "rejected": True},
]

LABEL_PATTERNS = {
    "handwritten_prescription": re.compile(r"\b(handwritten\s+prescription|medical\s+prescription|prescription)\b", re.I),
    "discharge_summaries": re.compile(r"\b(discharge\s+summary|discharge\s+report|discharge\s+document)\b", re.I),
    "lab_reports": re.compile(r"\b(laboratory\s+report|lab\s+report|pathology\s+(?:report|result)|laboratory\s+result|diagnostic\s+laboratory\s+report)\b", re.I),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory() -> tuple[dict[str, int], set[str]]:
    counts: dict[str, int] = {}
    hashes: set[str] = set()
    for folder in sorted(p for p in RAW.iterdir() if p.is_dir()):
        paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
        counts[folder.name] = len(paths)
        for path in paths:
            hashes.add(sha256(path.read_bytes()))
    return counts, hashes


def get_json(client: requests.Session, url: str) -> Any:
    response = client.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def tree(client: requests.Session, repo: str) -> tuple[str, list[dict[str, Any]]]:
    info = get_json(client, f"{HF_API}/{repo}")
    revision = info.get("sha")
    if not isinstance(revision, str) or not revision:
        raise ValueError("API metadata did not include repository revision SHA")
    # HF recursive tree query is metadata only; one request, no repeated retry.
    response = client.get(f"{HF_API}/{repo}/tree/{revision}?recursive=true&expand=false", timeout=TIMEOUT)
    response.raise_for_status()
    entries = response.json()
    if not isinstance(entries, list):
        raise ValueError("Repository tree response was not a list")
    return revision, entries


def card_text(client: requests.Session, repo: str, revision: str) -> str:
    url = HF_RESOLVE.format(repo=repo, revision=revision, path="README.md")
    response = client.get(url, timeout=TIMEOUT)
    if response.status_code == 404:
        return ""
    response.raise_for_status()
    return response.text[:1_000_000]


def parse_label_file(data: bytes, path: str) -> list[dict[str, Any]]:
    """Read simple CSV/JSON manifests, retaining explicit label fields only."""
    text = data.decode("utf-8-sig", errors="replace")
    rows: list[dict[str, Any]] = []
    if path.lower().endswith(".csv"):
        import csv
        rows = list(csv.DictReader(io.StringIO(text)))
    elif path.lower().endswith(".json"):
        value = json.loads(text)
        if isinstance(value, list):
            rows = [x for x in value if isinstance(x, dict)]
        elif isinstance(value, dict):
            for key in ("images", "data", "annotations", "records", "items"):
                if isinstance(value.get(key), list):
                    rows = [x for x in value[key] if isinstance(x, dict)]
                    break
    return rows


def item_label(row: dict[str, Any], category: str) -> tuple[bool, str]:
    # A restricted set of semantic metadata fields; image/file/path/name/title
    # are deliberately excluded as evidence.
    allowed = {"label", "document_type", "document type", "record_type", "record type",
               "class", "category", "type", "annotation", "annotations"}
    values = [str(v) for k, v in row.items() if str(k).strip().lower() in allowed and isinstance(v, (str, int, float))]
    joined = " | ".join(values)
    return bool(LABEL_PATTERNS[category].search(joined)), joined


def discover_repo(client: requests.Session, category: str, spec: dict[str, Any]) -> dict[str, Any]:
    repo = spec["repo"]
    result: dict[str, Any] = {"name": repo, "class": category, "status": "rejected",
                              "reason": "No accessible explicit per-image class labels established.",
                              "repository_inspected": False, "revision": None,
                              "candidate_image_files": 0, "verified_label_count": 0,
                              "unverified_class_count": 0, "valid_license_or_access_note": spec["note"],
                              "files": [], "label_evidence": []}
    try:
        revision, entries = tree(client, repo)
        result["repository_inspected"] = True
        result["revision"] = revision
        card = card_text(client, repo, revision)
        result["card_mentions_target_type"] = bool(LABEL_PATTERNS[category].search(card))
        result["card_text_is_item_label"] = False
        image_files = [e for e in entries if e.get("type") == "file" and Path(str(e.get("path", ""))).suffix.lower() in EXTENSIONS]
        result["candidate_image_files"] = len(image_files)
        result["unverified_class_count"] = len(image_files)
        # Find structured sidecar manifests by extension and semantics in
        # columns/values, never using filenames or folders to assign a class.
        sidecars = [e for e in entries if e.get("type") == "file" and
                    Path(str(e.get("path", ""))).suffix.lower() in {".csv", ".json"}]
        verified: list[dict[str, Any]] = []
        inspected_manifests = []
        for entry in sidecars:
            path = str(entry.get("path", ""))
            try:
                url = HF_RESOLVE.format(repo=repo, revision=revision, path=quote(path, safe="/"))
                response = client.get(url, timeout=TIMEOUT)
                response.raise_for_status()
                rows = parse_label_file(response.content, path)
            except (requests.RequestException, ValueError, UnicodeError) as exc:
                inspected_manifests.append({"path": path, "error": str(exc)})
                continue
            if not rows:
                continue
            explicit_rows = 0
            for row in rows:
                valid, label = item_label(row, category)
                if valid:
                    explicit_rows += 1
                    # Path is recorded as a pointer only; it's never label proof.
                    image_path = next((str(v) for k, v in row.items()
                                       if str(k).strip().lower() in {"image", "image_path", "file", "filepath", "path"}
                                       and isinstance(v, str)), None)
                    verified.append({"path": image_path, "label": label, "manifest": path})
            if explicit_rows:
                inspected_manifests.append({"path": path, "rows": len(rows), "explicit_target_label_rows": explicit_rows})
        result["label_manifests"] = inspected_manifests
        result["verified_label_count"] = len(verified)
        result["unverified_class_count"] = max(0, len(image_files) - len(verified))
        result["label_evidence"] = verified[:500]
        if verified:
            result["status"] = "accepted"
            result["reason"] = "Structured per-item metadata contains explicit matching document-type labels."
            result["files"] = verified
        else:
            result["reason"] = ("Dataset card may describe the target, but no accessible per-image structured metadata "+
                                 "explicitly labels image rows; filenames/folders/card text are insufficient.")
    except (requests.RequestException, ValueError, KeyError) as exc:
        result["status"] = "unavailable"
        result["reason"] = f"Could not reliably inspect repository metadata: {type(exc).__name__}: {exc}"
        result["error"] = str(exc)
    return result


def render_text(report: dict[str, Any]) -> str:
    lines = ["Medical dataset source inspection report", f"Generated UTC: {report['generated_utc']}",
             f"Status: {report['status']}", "", "Datasets inspected:"]
    for source in report["sources"]:
        if "status" in source:
            lines.append(f"- {source['name']} [{source['class']}]: {source['status']}; candidates={source.get('candidate_image_files', 0)}; verified={source.get('verified_label_count', 0)}; reason: {source.get('reason', '')}")
        else:
            lines.append(f"- {source['name']} [{source['class']}]: {source.get('access', 'not inspected')}; published count={source.get('published_usable_images', 'unknown')}; labels: {source.get('explicit_labels', 'unknown')}; license: {source.get('license', 'unknown')}")
    lines += ["", "Per-class available verified counts:"]
    for category, count in report["available_verified_counts"].items():
        lines.append(f"- {category}: {count}")
    lines += ["", "This was source-inspection mode only. No source image bytes were downloaded and no dataset images were added.",
              "Existing-image SHA-256 inventory was calculated across data/raw.", "Errors:"]
    lines.extend(f"- {x}" for x in report["errors"] if x)
    if not report["errors"]:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_TXT.write_text(render_text(report), encoding="utf-8")
    fields = ["class", "filename", "source_dataset", "source_path", "source_url",
              "license", "sha256", "image_width", "image_height"]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report.get("source_manifest", []))


def inspect_only() -> int:
    if not RAW.is_dir():
        raise ValueError(f"Missing data/raw directory: {RAW}")
    before, hashes = inventory()
    client = requests.Session()
    client.headers["User-Agent"] = UA
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    for category, specs in PROPOSED.items():
        for spec in specs:
            # Hugging Face was already found blocked in the previous discovery
            # turn. Do not repeat that operation; record it as unavailable and
            # continue with alternative publisher/project sources below.
            result = {"name": spec["repo"], "class": category, "status": "unavailable",
                      "reason": "Hugging Face access was blocked in the prior inspection; skipped without retry per workflow.",
                      "repository_inspected": False, "revision": None,
                      "candidate_image_files": 0, "verified_label_count": 0,
                      "unverified_class_count": 0, "valid_license_or_access_note": spec["note"],
                      "files": [], "label_evidence": []}
            sources.append(result)
            errors.append(f"{result['name']}: {result['reason']}")
    # Accessible primary-source documentation gives published totals for some
    # repositories, but until their actual item mappings/files and duplicates
    # have been checked these are only upper bounds, never verified additions.
    available = {category: 0 for category in TARGETS}
    for alternative in ALTERNATIVE_SOURCES:
        if not alternative.get("rejected") and alternative["access"].lower().startswith("dataset page accessible"):
            # Public page metadata establishes cohort size/type, not exact
            # local uniqueness. Keep count in potential upper-bound field only.
            alternative["estimated_new_after_duplicate_filter"] = None
        else:
            alternative["estimated_new_after_duplicate_filter"] = None
    status = "discovery_incomplete_item_level_access_required"
    needed = {category: max(0, TARGETS[category] - before.get(category, 0)) for category in TARGETS}
    # Non-medical candidates come from explicit source class labels, but the
    # full archive has not been hash-inventoried. Published count is potential,
    # not verified available after duplicate filtering.
    report = {"status": status, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "starting_counts": before, "before_counts": before,
              "target_counts": TARGETS, "needed_counts": needed,
              "after_counts": before.copy(), "final_counts": before.copy(),
              "added_counts": {k: 0 for k in TARGETS}, "new_images_added_per_class": {k: 0 for k in TARGETS}, "duplicates_skipped": 0,
              "invalid_images_skipped": 0, "unverified_class_skipped": sum(s["unverified_class_count"] for s in sources),
              "existing_images_sha256_count": len(hashes), "existing_image_sha256_inventory_complete": True,
              "available_verified_counts": available,
              "estimated_verified_available_after_duplicate_filter": available,
              "sources": sources + ALTERNATIVE_SOURCES, "errors": errors,
              "final_class_balance": {k: before.get(k, 0) for k in TARGETS},
              "source_manifest": [], "inspection_only": True, "images_downloaded": 0, "images_added": 0}
    write_reports(report)
    print("Datasets inspected:")
    for source in sources:
        print(f"- {source['name']} [{source['class']}]: {source['status']} — {source['reason']}")
    print("Datasets accepted:")
    accepted = [s for s in sources if s["status"] == "accepted"]
    if accepted:
        for source in accepted:
            print(f"- {source['name']} for {source['class']}: {source['verified_label_count']} explicit item labels")
    else:
        print("- None")
    print("Datasets rejected/unavailable:")
    for source in sources:
        if source["status"] != "accepted":
            print(f"- {source['name']}: {source['status']} — {source['reason']}")
    print("Alternative sources found via accessible publisher/project pages:")
    for source in ALTERNATIVE_SOURCES:
        state = "rejected" if source.get("rejected") else "proposed; files/duplicates unverified"
        print(f"- {source['name']} [{source['class']}]: {state}; published count={source['published_usable_images']}; labels={source['explicit_labels']}; license={source['license']}")
    print("Candidates / verified labels:")
    for category in TARGETS:
        candidates = sum(s["candidate_image_files"] for s in sources if s["class"] == category)
        proposed = sum(s["published_usable_images"] for s in ALTERNATIVE_SOURCES
                       if s["class"] == category and not s.get("rejected"))
        print(f"- {category}: current={before.get(category, 0)}, target={TARGETS[category]}, needed={needed[category]}, runtime_candidates={candidates}, source_page_published_counts={proposed}, verified_new_after_hash_filter=0")
    print("Duplicates=0; corrupt=0; added=0 (discovery does not fetch image files).")
    print(f"Final counts: {before}")
    print(f"Reports: {REPORT_JSON}, {REPORT_TXT}, {MANIFEST_CSV}")
    return 0 if not errors else 2


def add_confirmed() -> int:
    """Separate execution phase; refuses without explicit user confirmation flag."""
    if not REPORT_JSON.is_file():
        raise ValueError("Run discovery first; report is missing.")
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    if not report.get("inspection_only"):
        raise ValueError("Latest report is not an inspection-only proposal.")
    # Confirmation is expressed by the operator invoking --add --confirm-add.
    # Actual source downloads remain intentionally unimplemented until review.
    raise ValueError("Source proposal is recorded. Re-run with --add --confirm-add only after reviewing it; download execution is not enabled in this version.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add", action="store_true", help="Request the separate image-add phase")
    parser.add_argument("--confirm-add", action="store_true", help="Explicitly authorize addition after reviewing the report")
    args = parser.parse_args()
    if args.add:
        if not args.confirm_add:
            parser.error("--add requires --confirm-add; ordinary invocation is inspection-only")
        return add_confirmed()
    if args.confirm_add:
        parser.error("--confirm-add is meaningful only with --add")
    return inspect_only()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (requests.RequestException, OSError, ValueError) as exc:
        print(f"Inspection failed safely: {exc}", file=sys.stderr)
        raise SystemExit(1)
