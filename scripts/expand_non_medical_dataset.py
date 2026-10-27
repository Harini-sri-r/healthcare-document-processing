"""Add a deterministic, deduplicated subset of public non-medical documents.

RVL-CDIP-derived images are fetched through the Hugging Face Dataset Viewer,
one image at a time. AeroBench certificate PDFs are fetched individually and
their first pages are rendered as real document images. No full dataset archive
is downloaded. Existing files in data/raw/non_medical are treated as read-only.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pymupdf
import requests
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "non_medical"
REPORT_PATH = PROJECT_ROOT / "reports" / "non_medical_dataset_expansion_report.json"
SEED = 42
RVL_DATASET = "HAMMALE/rvl_cdip_OCR"
RVL_CONFIG = "default"
RVL_SPLIT = "train"
RVL_ROWS_URL = "https://datasets-server.huggingface.co/rows"
RVL_REPO_API = f"https://huggingface.co/api/datasets/{RVL_DATASET}"
AEROBENCH_DATASET = "FathinDos/aerobench"
AEROBENCH_API = f"https://huggingface.co/api/datasets/{AEROBENCH_DATASET}"
AEROBENCH_FILES_API = f"{AEROBENCH_API}/tree/{{revision}}/data/filled_forms?recursive=true&expand=false"
AEROBENCH_RESOLVE = f"https://huggingface.co/datasets/{AEROBENCH_DATASET}/resolve/{{revision}}/"
TARGETS = {
    "resumes_cvs": 50,
    "invoices": 50,
    "certificates": 50,
    "academic_general": 50,
    "letters_business": 50,
    "other_general": 50,
}
# Source-provided category metadata is used for selection; filenames are only
# output names and are never consulted to assign the document category.
RVL_GROUPS: dict[str, dict[str, int]] = {
    "resumes_cvs": {"resume": 50},
    "invoices": {"invoice": 50},
    "academic_general": {"presentation": 25, "news article": 25},
    "letters_business": {"letter": 50},
    "other_general": {"advertisement": 13, "memo": 13, "questionnaire": 12, "budget": 12},
}
PAGE_SIZE = 100
MAX_RETRIES = 4
USER_AGENT = "healthcare-document-processing-nonmedical-expansion/1.0"


class DownloadError(RuntimeError):
    """A public source could not provide the requested data."""


def session() -> requests.Session:
    value = requests.Session()
    value.headers.update({"User-Agent": USER_AGENT})
    return value


def get_json(client: requests.Session, url: str, *, params: dict[str, Any] | None = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(url, params=params, timeout=(20, 90))
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = max(10.0, float(retry_after)) if retry_after else 30.0
                except ValueError:
                    delay = 30.0
                print(f"Dataset source rate-limited the request; retrying in {delay:.0f}s.", flush=True)
                time.sleep(delay)
                last_error = requests.HTTPError("429 Too Many Requests")
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise DownloadError(f"Could not access public dataset metadata at {url}: {last_error}")


def fetch_bytes(client: requests.Session, url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with client.get(url, timeout=(20, 120), stream=True) as response:
                response.raise_for_status()
                return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise DownloadError(f"Could not download an individual public source file: {last_error}")


def image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def existing_file_digests() -> set[str]:
    """Read existing images without changing any existing file."""
    digests: set[str] = set()
    for path in OUTPUT_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}:
            continue
        try:
            digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
            with Image.open(path) as image:
                image.load()
                digests.add(image_digest(image))
        except (OSError, UnidentifiedImageError, ValueError):
            # Existing user data is preserved byte-for-byte; it is not counted
            # as a duplicate hash when it cannot be decoded.
            continue
    return digests


def valid_category_count(category: str) -> int:
    """Count readable images already named for a document category."""
    count = 0
    for path in OUTPUT_DIR.glob(f"public_{category}_*.jpg"):
        try:
            with Image.open(path) as image:
                image.load()
            count += 1
        except (OSError, UnidentifiedImageError, ValueError):
            # Preserve pre-existing files; they simply do not satisfy the target.
            continue
    return count


def verify_image_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.convert("RGB")


def source_revision(client: requests.Session, api_url: str) -> str:
    payload = get_json(client, api_url)
    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha:
        raise DownloadError(f"Source API did not provide a revision SHA for {api_url}.")
    return sha


def scan_rvl_candidates(client: requests.Session) -> dict[str, list[dict[str, Any]]]:
    """Read category metadata and image URLs, never fetching unselected images."""
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        payload = get_json(
            client,
            RVL_ROWS_URL,
            params={"dataset": RVL_DATASET, "config": RVL_CONFIG, "split": RVL_SPLIT,
                    "offset": offset, "length": PAGE_SIZE},
        )
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            if total is None:
                raise DownloadError("RVL-CDIP Dataset Viewer returned no rows.")
            break
        total = int(payload.get("num_rows_total", 0))
        for item in rows:
            row = item.get("row", {})
            category = row.get("category")
            image = row.get("image")
            image_url = image.get("src") if isinstance(image, dict) else None
            if isinstance(category, str) and isinstance(image_url, str):
                candidates[category].append({
                    "category": category,
                    "image_url": image_url,
                    "row_index": int(item.get("row_idx", offset)),
                })
        offset += len(rows)
        print(f"Scanned RVL-CDIP metadata: {min(offset, total)}/{total} rows", flush=True)
        # The Dataset Viewer applies per-client request limits. Throttle below
        # its observed limit instead of retrying rapid requests into a 429.
        time.sleep(3.0)
    if not candidates:
        raise DownloadError("RVL-CDIP Dataset Viewer did not expose image URLs and category metadata.")
    return candidates


def aero_pdf_files(client: requests.Session, revision: str) -> list[dict[str, Any]]:
    listing = get_json(client, AEROBENCH_FILES_API.format(revision=revision))
    files = [entry for entry in listing if entry.get("type") == "file"
             and str(entry.get("path", "")).lower().endswith(".pdf")]
    if not files:
        raise DownloadError("AeroBench did not expose individual filled-form certificate PDFs.")
    return files


def save_image(image: Image.Image, category: str, index: int, known: set[str],
               manifest: list[dict[str, str]], *, source: str,
               source_category: str) -> tuple[bool, str]:
    source_digest = image_digest(image)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=95, optimize=True)
    image_bytes = encoded.getvalue()
    file_digest = hashlib.sha256(image_bytes).hexdigest()
    if source_digest in known or file_digest in known:
        return False, source_digest
    destination = OUTPUT_DIR / f"public_{category}_{index:03d}.jpg"
    while destination.exists():
        index += 1
        destination = OUTPUT_DIR / f"public_{category}_{index:03d}.jpg"
    # Never overwrite or change pre-existing user data.
    if destination.exists():
        raise DownloadError(f"Refusing to overwrite existing file: {destination}")
    destination.write_bytes(image_bytes)
    try:
        with Image.open(destination) as check:
            check.verify()
        with Image.open(destination) as check:
            check.load()
            saved_digest = image_digest(check)
    except (OSError, UnidentifiedImageError, ValueError):
        destination.unlink(missing_ok=True)
        raise
    # JPEG is lossy, so preserve both the decoded source fingerprint and the
    # saved-file fingerprint for exact duplicate checks during this run.
    known.add(source_digest)
    known.add(saved_digest)
    known.add(hashlib.sha256(destination.read_bytes()).hexdigest())
    manifest.append({"file": destination.name, "category": category, "source": source,
                     "source_category": source_category, "sha256_pixels": saved_digest})
    return True, source_digest


def download_rvl(client: requests.Session, candidates: dict[str, list[dict[str, Any]]],
                 known: set[str], rng: random.Random, manifest: list[dict[str, str]]) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    invalid = duplicates = 0
    for group, target_count in TARGETS.items():
        if group == "certificates":
            continue
        group_counts: Counter[str] = Counter()
        already_present = valid_category_count(group)
        remaining = max(0, target_count - already_present)
        accepted = 0
        attempts = 0
        for source_category, wanted in RVL_GROUPS[group].items():
            wanted = min(wanted, remaining)
            if wanted <= 0:
                continue
            available = list(candidates.get(source_category, []))
            if len(available) < wanted:
                raise DownloadError(
                    f"RVL-CDIP category {source_category!r} has only {len(available)} accessible rows; "
                    f"need {wanted}. Refusing to fabricate or relabel data."
                )
            rng.shuffle(available)
            accepted_from_source_category = 0
            for item in available:
                if accepted_from_source_category >= wanted:
                    break
                attempts += 1
                try:
                    source_bytes = fetch_bytes(client, item["image_url"])
                    image = verify_image_bytes(source_bytes)
                    # The category comes from the dataset row, never its path.
                    was_saved, _digest = save_image(
                        image, group, accepted + 1, known, manifest,
                        source=RVL_DATASET, source_category=source_category,
                    )
                    if not was_saved:
                        duplicates += 1
                        continue
                    accepted += 1
                    accepted_from_source_category += 1
                    group_counts[source_category] += 1
                except (DownloadError, requests.RequestException) as exc:
                    if isinstance(exc, DownloadError):
                        raise
                    raise DownloadError(f"RVL-CDIP image source failed: {exc}") from exc
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    invalid += 1
                    print(f"Skipping unreadable RVL-CDIP image row {item['row_index']}: {exc}", flush=True)
            if accepted_from_source_category != wanted:
                raise DownloadError(
                    f"Could not fill source category {source_category}: saved "
                    f"{accepted_from_source_category}/{wanted} after {attempts} rows."
                )
            remaining -= accepted_from_source_category
        counts[group] = accepted
        if accepted + already_present != target_count:
            raise DownloadError(
                f"Could not fill {group}: existing={already_present}, added={accepted}, "
                f"target={target_count} after {attempts} selected rows."
            )
        print(f"Downloaded {group}: {accepted} new; {already_present + accepted} total", flush=True)
    return counts, invalid, duplicates


def download_certificates(client: requests.Session, revision: str, known: set[str], rng: random.Random,
                          manifest: list[dict[str, str]]) -> tuple[int, int, int]:
    files = aero_pdf_files(client, revision)
    if len(files) < TARGETS["certificates"]:
        raise DownloadError(f"AeroBench exposes only {len(files)} certificate PDFs; 50 are required.")
    rng.shuffle(files)
    already_present = valid_category_count("certificates")
    remaining = max(0, TARGETS["certificates"] - already_present)
    count = invalid = duplicates = 0
    attempts = 0
    for entry in files:
        if count >= remaining:
            break
        path = str(entry["path"])
        relative_path = path if path.startswith("data/") else f"data/filled_forms/{path}"
        url = AEROBENCH_RESOLVE.format(revision=revision) + quote(relative_path, safe="/")
        attempts += 1
        try:
            pdf_data = fetch_bytes(client, url)
            document = pymupdf.open(stream=pdf_data, filetype="pdf")
            try:
                if document.page_count < 1:
                    raise ValueError("PDF has no pages")
                pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7), alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            finally:
                document.close()
            was_saved, _digest = save_image(
                image, "certificates", count + 1, known, manifest,
                source=AEROBENCH_DATASET, source_category="real_release_certificate",
            )
            if not was_saved:
                duplicates += 1
                continue
            count += 1
        except DownloadError:
            raise
        except requests.RequestException as exc:
            raise DownloadError(f"AeroBench certificate source failed: {exc}") from exc
        except (OSError, ValueError, RuntimeError, UnidentifiedImageError) as exc:
            invalid += 1
            print(f"Skipping invalid AeroBench certificate PDF {path}: {exc}", flush=True)
    if count + already_present != TARGETS["certificates"]:
        raise DownloadError(f"Could not fill certificates: existing={already_present}, added={count}/50 after {attempts} files.")
    print(f"Downloaded certificates: {count}", flush=True)
    return count, invalid, duplicates


def final_inventory() -> tuple[dict[str, int], int, int]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
    category_counts = Counter()
    total_valid = invalid_existing = 0
    for path in OUTPUT_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, UnidentifiedImageError, ValueError):
            invalid_existing += 1
            continue
        total_valid += 1
        name = path.name.lower()
        if name.startswith("cord_receipt_"):
            category_counts["receipts"] += 1
        elif name.startswith("public_resumes_cvs_"):
            category_counts["resumes_cvs"] += 1
        elif name.startswith("public_invoices_"):
            category_counts["invoices"] += 1
        elif name.startswith("public_certificates_"):
            category_counts["certificates"] += 1
        elif name.startswith("public_academic_general_"):
            category_counts["academic_general"] += 1
        elif name.startswith("public_letters_business_"):
            category_counts["letters_business"] += 1
        elif name.startswith("public_other_general_"):
            category_counts["other_general"] += 1
        else:
            category_counts["uncategorized"] += 1
    return dict(category_counts), total_valid, invalid_existing


def write_report(manifest: list[dict[str, str]], rvl_revision: str, aero_revision: str,
                 errors: dict[str, int], final_counts: dict[str, int],
                 total_valid: int, invalid_existing: int) -> None:
    previous = {}
    if REPORT_PATH.is_file():
        try:
            previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    by_file = {item["file"]: item for item in previous.get("files", [])}
    by_file.update({item["file"]: item for item in manifest})
    cumulative_manifest = list(by_file.values())
    category_counts = Counter(item["category"] for item in cumulative_manifest)
    source_counts = Counter(item["source"] for item in cumulative_manifest)
    source_category_counts = Counter(item["source_category"] for item in cumulative_manifest)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "directory": str(OUTPUT_DIR),
        "added_images": len(cumulative_manifest),
        "final_valid_image_count": total_valid,
        "final_category_counts": final_counts,
        "invalid_existing_files_excluded": invalid_existing,
        "category_counts_added": dict(sorted(category_counts.items())),
        "source_counts_added": dict(source_counts),
        "source_category_counts_added": dict(sorted(source_category_counts.items())),
        "invalid_images_skipped": previous.get("invalid_images_skipped", 0) + errors["invalid"],
        "duplicates_skipped": previous.get("duplicates_skipped", 0) + errors["duplicates"],
        "sources": [
            {"name": RVL_DATASET, "url": f"https://huggingface.co/datasets/{RVL_DATASET}",
             "revision": rvl_revision, "complete_dataset_size": "1.44 GB", "license_note": "Dataset card describes a public research dataset; underlying RVL-CDIP records originate from the UCSF Legacy Tobacco Documents Library. Selected records are document classes explicitly labeled resume, invoice, letter, presentation, news, advertisement, memo, questionnaire, or budget."},
            {"name": AEROBENCH_DATASET, "url": f"https://huggingface.co/datasets/{AEROBENCH_DATASET}",
             "revision": aero_revision, "complete_filled_forms_size": "117 MB", "license": "CC BY 4.0", "license_note": "Dataset card identifies the selected filled EASA Form 1 / FAA 8130-3 items as real public release certificates."},
        ],
        "files": cumulative_manifest,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = session()
    try:
        rvl_revision = source_revision(client, RVL_REPO_API)
        aero_revision = source_revision(client, AEROBENCH_API)
        candidates = scan_rvl_candidates(client)
        known = existing_file_digests()
        rng = random.Random(SEED)
        manifest: list[dict[str, str]] = []
        rvl_counts, invalid_rvl, dup_rvl = download_rvl(client, candidates, known, rng, manifest)
        certificate_count, invalid_cert, dup_cert = download_certificates(
            client, aero_revision, known, rng, manifest
        )
        source_category_counts: Counter[str] = Counter(item["source_category"] for item in manifest)
        invalid = invalid_rvl + invalid_cert
        duplicates = dup_rvl + dup_cert
        final_counts, total_valid, invalid_existing = final_inventory()
        write_report(manifest, rvl_revision, aero_revision,
                     {"invalid": invalid, "duplicates": duplicates},
                     final_counts, total_valid, invalid_existing)
    except (DownloadError, requests.RequestException, OSError, ValueError) as exc:
        print(f"Non-medical dataset expansion failed: {exc}", file=sys.stderr)
        return 1

    final_counts, total_valid, invalid_existing = final_inventory()
    all_images = [path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in
                  {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}]
    disk_bytes = sum(path.stat().st_size for path in all_images)
    existing_before_added = len(all_images) - len(manifest)
    print("\nAdded category counts:")
    for category in TARGETS:
        print(f"{category}: {sum(1 for item in manifest if item['category'] == category)}")
    source_counts = Counter(item["source"] for item in manifest)
    source_category_counts = Counter(item["source_category"] for item in manifest)
    print("\nAdded counts by source:")
    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count}")
    print("\nAdded counts by source category:")
    for category, count in sorted(source_category_counts.items()):
        print(f"{category}: {count}")
    print(f"\nExisting images preserved: {existing_before_added}")
    print(f"Images added: {len(manifest)}")
    print("\nFinal valid category counts:")
    for category in ("receipts", *TARGETS.keys()):
        print(f"{category}: {final_counts.get(category, 0)}")
    print(f"Final valid non-medical image count: {total_valid}")
    print(f"Invalid downloaded images skipped: {invalid}")
    print(f"Existing invalid files excluded from count: {invalid_existing}")
    print(f"Duplicates skipped: {duplicates}")
    print(f"Total disk usage: {disk_bytes / (1024 * 1024):.2f} MiB")
    print(f"Directory: {OUTPUT_DIR}")
    print(f"Dataset report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
