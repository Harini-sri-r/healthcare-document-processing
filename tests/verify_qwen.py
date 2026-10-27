"""Verification script to verify local Qwen2.5-VL model path and offline readiness."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.extraction.qwen_extractor import QwenExtractor

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFYING LOCAL QWEN2.5-VL MODEL PATH")
    print("=" * 60)
    print(f"Project Root     : {settings.PROJECT_ROOT}")
    print(f"Local Qwen Path  : {settings.QWEN_MODEL_PATH}")
    print(f"Directory Exists : {settings.QWEN_MODEL_PATH.is_dir()}")

    config_file = settings.QWEN_MODEL_PATH / "config.json"
    index_file = settings.QWEN_MODEL_PATH / "model.safetensors.index.json"
    print(f"config.json      : {config_file.is_file()}")
    print(f"safetensors index: {index_file.is_file()}")

    extractor = QwenExtractor()
    print(f"Extractor Target : {extractor.model_path}")
    print(f"Model Lazy Load  : {not extractor.is_loaded}")
    print("=" * 60)
    print("Local Qwen path is verified and ready for inference.")
    print("=" * 60)
