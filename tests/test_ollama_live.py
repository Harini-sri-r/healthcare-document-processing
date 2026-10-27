"""Live integration test for OllamaExtractor — requires Ollama server running.

Prerequisites:
    1. ollama serve   (in a separate terminal)
    2. ollama pull qwen2.5vl:3b

Run manually:
    .venv/Scripts/python.exe tests/test_ollama_live.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

if __name__ != "__main__":
    pytest.skip("Manual live Ollama test; run directly when explicitly validating a real document.", allow_module_level=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.extraction.json_validator import JSONValidator
from src.extraction.ollama_extractor import OllamaExtractor


# Use a real lab report image from the dataset
IMAGE_PATH = (
    settings.RAW_DATA_PATH
    / "lab_reports"
    / "AHD-0425-PA-0007561_JITENDRA TRIVEDI DS_28-04-2025_1019-21_AM.pdf_page_9.png"
)


def main():
    print("\n" + "=" * 60)
    print(" OLLAMA LIVE INTEGRATION TEST")
    print("=" * 60)

    # 1. Initialize extractor
    extractor = OllamaExtractor()
    print(f" Model       : {extractor.model}")
    print(f" Ollama URL  : {extractor.base_url}")

    # 2. Check connection
    print("\n--- Checking Ollama connection ---")
    conn = extractor.check_connection()
    print(f" Reachable       : {conn['reachable']}")
    print(f" Model Available : {conn['model_available']}")
    if conn["error"]:
        print(f" Error           : {conn['error']}")
        print("\n[FAILED] Cannot proceed without a running Ollama server and model.")
        print("Run:  ollama serve")
        print(f"Run:  ollama pull {extractor.model}")
        sys.exit(1)

    # 3. Verify image
    if not IMAGE_PATH.is_file():
        print(f"\n[FAILED] Image not found: {IMAGE_PATH}")
        sys.exit(1)
    print(f"\n Image Path : {IMAGE_PATH}")

    # 4. Run extraction
    print("\n--- Sending image to Ollama ---")
    print(" (This may take 30-120 seconds on CPU...)")
    result = extractor.extract(IMAGE_PATH)

    # 5. Report results
    print(f"\n--- Extraction Result ---")
    print(f" Success    : {result['success']}")
    print(f" Attempts   : {result['attempts']}")

    if result["success"] and result["data"]:
        print("\n========== EXTRACTED JSON ==========\n")
        print(json.dumps(result["data"], indent=2, ensure_ascii=False))

        # 6. Validate
        print("\n--- JSON Validation ---")
        validator = JSONValidator()
        validation = validator.validate(result)
        print(f" Valid          : {validation['valid']}")
        print(f" Issue Count   : {validation['issue_count']}")
        print(f" Requires Review: {validation['requires_review']}")
        if validation["issues"]:
            for issue in validation["issues"][:10]:
                print(f"   * [{issue.get('type')}] {issue.get('field', '$')}: {issue.get('message')}")
    else:
        print(f"\n[FAILED] Extraction error: {result.get('error')}")
        if result.get("history"):
            for h in result["history"]:
                print(f"  Attempt {h['attempt']}: valid_json={h['valid_json']}, error={h.get('error')}")

    print("\n" + "=" * 60)
    print(" Live integration test complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
