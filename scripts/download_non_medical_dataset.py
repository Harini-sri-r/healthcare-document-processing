"""Fetch a reproducible subset of real CORD receipt images by HTTP byte ranges.

The source image ZIP is 168 MB for 800 training receipts. This downloader reads
the small JSONL index and ZIP central directory, then requests only the byte
ranges for the selected 400 JPEG entries. It never downloads or caches the ZIP.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
import struct
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import settings


DATASET_NAME = "CORD (receipt images), Heliosoph mirror of NAVER Clova CORD"
DATASET_URL = "https://huggingface.co/datasets/Heliosoph/CORD"
REVISION = "51b7e932f2a07a883d77487b34aa5a33f76fa713"
BASE_URL = f"https://huggingface.co/datasets/Heliosoph/CORD/resolve/{REVISION}"
IMAGE_ZIP_URL = f"{BASE_URL}/cord-train-images.zip"
INDEX_URL = f"{BASE_URL}/cord-train.jsonl.gz"
SEED = 42
TARGET_IMAGES = 400
WORKERS = 8
USER_AGENT = "healthcare-document-processing-cord-subset/1.0"

_thread_local = threading.local()


class SourceError(RuntimeError):
    pass


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = session
    return session


def _range(url: str, start: int, end: int) -> bytes:
    response = _session().get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        stream=True,
        timeout=(20, 90),
    )
    try:
        if response.status_code != 206:
            raise SourceError(
                f"Source does not support selective HTTP ranges for {url.rsplit('/', 1)[-1]} "
                f"(HTTP {response.status_code}; expected 206 Partial Content). "
                "Refusing to download the full archive."
            )
        content_range = response.headers.get("Content-Range", "")
        expected_prefix = f"bytes {start}-{end}/"
        if not content_range.startswith(expected_prefix):
            raise SourceError(f"Unexpected Content-Range header: {content_range!r}")
        return response.content
    finally:
        response.close()


def _file_size(url: str) -> int:
    response = _session().head(url, allow_redirects=True, timeout=(20, 60))
    try:
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        if value is None:
            raise SourceError("Source did not report Content-Length for its image archive.")
        return int(value)
    finally:
        response.close()


def _archive_directory(url: str, size: int) -> dict[str, dict[str, int]]:
    tail_size = min(size, 65_557)
    tail_start = size - tail_size
    tail = _range(url, tail_start, size - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0 or len(tail) < eocd_pos + 22:
        raise SourceError("Could not locate the ZIP end-of-central-directory record.")
    (_sig, disk, cd_disk, disk_entries, total_entries, cd_size, cd_offset, _comment) = struct.unpack_from(
        "<4s4H2LH", tail, eocd_pos
    )
    if disk or cd_disk or disk_entries != total_entries:
        raise SourceError("Multi-disk ZIP archives are not supported.")

    # ZIP64 is unnecessary for this source archive, but handle it if the server
    # changes the archive format while preserving the pinned revision.
    if total_entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        locator_pos = eocd_pos - 20
        if locator_pos < 0 or tail[locator_pos:locator_pos + 4] != b"PK\x06\x07":
            raise SourceError("ZIP64 archive is missing its locator record.")
        _sig, _disk, zip64_offset, _disks = struct.unpack_from("<4sLQL", tail, locator_pos)
        zip64_head = _range(url, zip64_offset, zip64_offset + 55)
        if zip64_head[:4] != b"PK\x06\x06":
            raise SourceError("Invalid ZIP64 end-of-central-directory record.")
        (_zsig, _record_size, _made, _needed, _disk, _cd_disk,
         _disk_count, total_entries, cd_size, cd_offset) = struct.unpack_from("<4sQ2H2L4Q", zip64_head)

    central = _range(url, cd_offset, cd_offset + cd_size - 1)
    entries: dict[str, dict[str, int]] = {}
    offset = 0
    while offset < len(central):
        if central[offset:offset + 4] != b"PK\x01\x02":
            raise SourceError("Invalid ZIP central-directory entry encountered.")
        fields = struct.unpack_from("<4s6H3L5H2L", central, offset)
        (_sig, _made, _needed, flags, method, _time, _date, crc,
         compressed_size, uncompressed_size, filename_size, extra_size,
         comment_size, _start_disk, _internal, _external, local_offset) = fields
        name_start = offset + 46
        filename_bytes = central[name_start:name_start + filename_size]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        filename = filename_bytes.decode(encoding)
        extra_start = name_start + filename_size
        extra = central[extra_start:extra_start + extra_size]
        if (compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF
                or local_offset == 0xFFFFFFFF):
            cursor = 0
            while cursor + 4 <= len(extra):
                tag, length = struct.unpack_from("<HH", extra, cursor)
                value_start = cursor + 4
                if tag == 0x0001:
                    position = value_start
                    if uncompressed_size == 0xFFFFFFFF:
                        uncompressed_size = struct.unpack_from("<Q", extra, position)[0]
                        position += 8
                    if compressed_size == 0xFFFFFFFF:
                        compressed_size = struct.unpack_from("<Q", extra, position)[0]
                        position += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack_from("<Q", extra, position)[0]
                    break
                cursor = value_start + length
        entries[filename] = {
            "flags": flags,
            "method": method,
            "crc": crc,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        }
        offset = extra_start + extra_size + comment_size
    if not entries or len(entries) > total_entries:
        raise SourceError("ZIP central directory did not contain the expected image entries.")
    return entries


def _load_index() -> list[dict]:
    response = _session().get(INDEX_URL, timeout=(20, 90))
    response.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as compressed:
        records = [json.loads(line) for line in compressed if line.strip()]
    if not records or not all("image_file" in record for record in records):
        raise SourceError("CORD training index is empty or missing image_file entries.")
    return records


def _read_zip_member(url: str, entry: dict[str, int]) -> bytes:
    header = _range(url, entry["local_offset"], entry["local_offset"] + 29)
    if header[:4] != b"PK\x03\x04":
        raise SourceError("Invalid local ZIP file header for a selected receipt.")
    (_sig, _needed, flags, method, _mtime, _mdate, _crc, _compressed,
     _uncompressed, name_len, extra_len) = struct.unpack_from("<4s5H3L2H", header)
    if flags & 0x1:
        raise SourceError("Encrypted source ZIP entries are not supported.")
    data_start = entry["local_offset"] + 30 + name_len + extra_len
    compressed = _range(url, data_start, data_start + entry["compressed_size"] - 1)
    if method == 0:
        image_bytes = compressed
    elif method == 8:
        image_bytes = zlib.decompress(compressed, -15)
    else:
        raise SourceError(f"Unsupported ZIP compression method {method}.")
    if len(image_bytes) != entry["uncompressed_size"]:
        raise SourceError("Selected ZIP image did not match its declared uncompressed size.")
    if (zlib.crc32(image_bytes) & 0xFFFFFFFF) != entry["crc"]:
        raise SourceError("Selected ZIP image failed its CRC check.")
    return image_bytes


def _existing_images(output_dir: Path) -> tuple[set[str], int]:
    hashes: set[str] = set()
    receipt_count = 0
    for path in output_dir.glob("cord_receipt_*.jpg"):
        try:
            if path.stat().st_size == 0:
                path.unlink(missing_ok=True)
                continue
            with Image.open(path) as image:
                image.load()
                rgb = image.convert("RGB")
            hashes.add(hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest())
            receipt_count += 1
        except (OSError, UnidentifiedImageError):
            path.unlink(missing_ok=True)
    return hashes, receipt_count


def _decode_and_save(source_bytes: bytes, destination: Path) -> str:
    with Image.open(io.BytesIO(source_bytes)) as source:
        source.load()
        image = source.convert("RGB")
    image_hash = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
    image.save(destination, format="JPEG", quality=95, optimize=True)
    with Image.open(destination) as saved:
        saved.verify()
    if destination.stat().st_size <= 0:
        destination.unlink(missing_ok=True)
        raise UnidentifiedImageError("Saved image is empty.")
    return image_hash


def download_subset() -> dict[str, int | Path]:
    output_dir = settings.RAW_DATA_PATH / "non_medical"
    output_dir.mkdir(parents=True, exist_ok=True)
    duplicate_hashes, already_downloaded = _existing_images(output_dir)
    invalid = 0
    duplicates = 0
    archive_size = _file_size(IMAGE_ZIP_URL)
    entries = _archive_directory(IMAGE_ZIP_URL, archive_size)
    records = _load_index()
    rng = random.Random(SEED)
    selected = rng.sample(records, min(TARGET_IMAGES, len(records)))
    selected = [record for record in selected if already_downloaded < TARGET_IMAGES]
    start_index = already_downloaded + 1
    work = []
    for record in selected:
        entry = entries.get(record["image_file"])
        if entry is None:
            invalid += 1
            continue
        work.append((record["image_file"], entry))

    print(f"Source: {DATASET_NAME}")
    print(f"Pinned source revision: {REVISION}")
    print(f"Image archive size: {archive_size / (1024 * 1024):.2f} MiB (not downloaded as a whole)")
    print(f"Selected source images: {len(work)}; fetching individual ZIP byte ranges...")
    next_index = start_index
    # Each transfer reads one local header and one compressed image payload.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_read_zip_member, IMAGE_ZIP_URL, entry): name for name, entry in work}
        for future in as_completed(futures):
            name = futures[future]
            try:
                image_bytes = future.result()
                destination = output_dir / f"cord_receipt_{next_index:04d}.jpg"
                image_hash = _decode_and_save(image_bytes, destination)
                if image_hash in duplicate_hashes:
                    destination.unlink(missing_ok=True)
                    duplicates += 1
                    continue
                duplicate_hashes.add(image_hash)
                next_index += 1
            except (requests.RequestException, SourceError, OSError,
                    UnidentifiedImageError, ValueError, zlib.error) as exc:
                invalid += 1
                print(f"Skipping invalid source image {name}: {exc}", flush=True)
                # No corrupted temporary output is retained.
                (output_dir / f"cord_receipt_{next_index:04d}.jpg").unlink(missing_ok=True)
            downloaded = next_index - start_index
            if downloaded % 25 == 0 and downloaded:
                print(f"Progress: {downloaded}/{len(work)} new valid images", flush=True)

    final_count = sum(1 for _ in output_dir.glob("cord_receipt_*.jpg"))
    return {
        "downloaded": max(0, final_count - already_downloaded),
        "total": final_count,
        "invalid": invalid,
        "duplicates": duplicates,
        "output_dir": output_dir,
        "archive_size": archive_size,
    }


def validate_output(output_dir: Path) -> tuple[int, int]:
    invalid = 0
    duplicates = 0
    seen: set[str] = set()
    for path in output_dir.glob("cord_receipt_*.jpg"):
        try:
            if path.stat().st_size <= 0:
                raise UnidentifiedImageError("empty image")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                image_hash = hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()
            if image_hash in seen:
                duplicates += 1
            else:
                seen.add(image_hash)
        except (OSError, UnidentifiedImageError, ValueError):
            path.unlink(missing_ok=True)
            invalid += 1
    return invalid, duplicates


def main() -> int:
    try:
        result = download_subset()
    except (SourceError, requests.RequestException, OSError, ValueError) as exc:
        print(f"CORD subset download failed: {exc}", file=sys.stderr)
        return 1

    invalid_after_check, duplicate_after_check = validate_output(result["output_dir"])
    storage_bytes = sum(path.stat().st_size for path in result["output_dir"].glob("cord_receipt_*.jpg"))
    print("\nCategory counts:")
    print(f"receipt: {result['total']}")
    print(f"Total images: {result['total']}")
    print(f"Total disk usage: {storage_bytes / (1024 * 1024):.2f} MiB")
    print(f"Invalid images: {result['invalid'] + invalid_after_check}")
    print(f"Duplicates skipped: {result['duplicates'] + duplicate_after_check}")
    print(f"Directory: {result['output_dir']}")
    print(f"Attribution: CORD authors / NAVER Clova; CC BY 4.0; source {DATASET_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
