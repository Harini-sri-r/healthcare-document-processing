"""Information extraction components using local Vision-Language Models."""

from src.extraction.json_validator import (
    JSONValidator,
    clean_and_parse_json,
    normalize_empty_strings_to_none,
)
from src.extraction.ollama_extractor import OllamaExtractor
from src.extraction.prompts import build_medical_extraction_prompt
from src.extraction.qwen_extractor import QwenExtractor
from src.extraction.source_verifier import SourceVerifier

__all__ = [
    "QwenExtractor",
    "SourceVerifier",
    "OllamaExtractor",
    "build_medical_extraction_prompt",
    "clean_and_parse_json",
    "normalize_empty_strings_to_none",
    "JSONValidator",
]

