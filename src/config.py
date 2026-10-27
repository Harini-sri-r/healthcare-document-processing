from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Central configuration for paths, model parameters, and runtime settings."""

    # Project Root: resolved relative to this configuration file
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # Directory Paths
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATA_PATH: Path = PROJECT_ROOT / "data"
    RAW_DATA_PATH: Path = DATA_PATH / "raw"
    PROCESSED_DATA_PATH: Path = DATA_PATH / "processed"
    REPORTS_PATH: Path = PROJECT_ROOT / "reports"

    # Model Paths
    RESNET_MODEL_PATH: 
    Path = Path(
        os.getenv(
            "MEDICAL_DOCUMENT_RESNET_MODEL_PATH",
            str(MODELS_DIR / "medical_document_resnet18.pth"),
        )
    )
    RESNET_V3_MODEL_PATH: Path = Path(
        os.getenv(
            "MEDICAL_DOCUMENT_RESNET_V3_MODEL_PATH",
            str(MODELS_DIR / "medical_document_resnet18_v3_final.pth"),
        )
    )
    RESNET_V2_MODEL_PATH: Path = Path(
        os.getenv(
            "MEDICAL_DOCUMENT_RESNET_V2_MODEL_PATH",
            str(MODELS_DIR / "medical_document_resnet18_v2.pth"),
        )
    )
    QWEN_MODEL_PATH: Path = Path(
        os.getenv(
            "MEDICAL_DOCUMENT_QWEN_MODEL_PATH",
            str(MODELS_DIR / "qwen2.5-vl-3b-instruct"),
        )
    )

    # Runtime & Device Settings
    DEVICE: str = os.getenv(
        "MEDICAL_DOCUMENT_DEVICE",
        "cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("MEDICAL_DOCUMENT_CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.80")
    )
    # Provisional gate threshold; not scientifically calibrated. Evaluate on
    # representative medical and non-medical images before production tuning.
    RESNET_MEDICAL_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("RESNET_MEDICAL_CONFIDENCE_THRESHOLD", "0.80")
    )

    # Qwen Vision-Language Extractor Parameters
    QWEN_MAX_RETRIES: int = int(os.getenv("MEDICAL_DOCUMENT_QWEN_MAX_RETRIES", "2"))
    QWEN_MAX_NEW_TOKENS: int = int(os.getenv("MEDICAL_DOCUMENT_QWEN_MAX_NEW_TOKENS", "1600"))

    # Ollama Local Inference Settings
    OLLAMA_BASE_URL: str = os.getenv("MEDICAL_DOCUMENT_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.getenv("MEDICAL_DOCUMENT_OLLAMA_MODEL", "qwen2.5vl:3b")
    # CPU-only Qwen vision extraction of a full document can exceed two minutes.
    OLLAMA_TIMEOUT: int = int(os.getenv("MEDICAL_DOCUMENT_OLLAMA_TIMEOUT", "600"))
    OLLAMA_EXTRACTION_MAX_DURATION: int = int(os.getenv("MEDICAL_DOCUMENT_OLLAMA_EXTRACTION_MAX_DURATION", "1200"))
    # A document image can contribute thousands of vision tokens.  8192 leaves
    # room for the dynamic extraction prompt and the JSON response.
    OLLAMA_NUM_CTX: int = int(os.getenv("MEDICAL_DOCUMENT_OLLAMA_NUM_CTX", "8192"))
    OLLAMA_NUM_PREDICT: int = int(
        os.getenv("MEDICAL_DOCUMENT_OLLAMA_NUM_PREDICT", "1600")
    )

    # Independent verification is opt-in and uses the local Ollama instance.
    SECOND_MODEL_VERIFICATION_ENABLED: bool = os.getenv(
        "MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_ENABLED", "false"
    ).lower() in {"1", "true", "yes", "on"}
    SECOND_MODEL_VERIFICATION_MODEL: str = os.getenv(
        "MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_MODEL", "granite3.2-vision:2b"
    )
    SECOND_MODEL_VERIFICATION_TIMEOUT: int = int(os.getenv(
        "MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_TIMEOUT", "600"
    ))
    SECOND_MODEL_VERIFICATION_NUM_PREDICT: int = int(os.getenv(
        "MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_NUM_PREDICT", "2000"
    ))

    # Logging
    LOG_LEVEL: str = os.getenv("MEDICAL_DOCUMENT_LOG_LEVEL", "INFO")


settings = Settings()

__all__ = ["settings", "Settings"]
