"""Source-grounded consistency checks for dynamically extracted document JSON."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from src.extraction.ollama_extractor import OllamaExtractor
from src.preprocessing.image_preprocessing import ImageInput, load_document_pages
from src.utils.logging_utils import get_logger

logger = get_logger("source_verifier")


def _parse_source_verifier_response(text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse one complete leading JSON object, allowing only non-JSON commentary after it.

    Unlike the extraction parser, this deliberately does not search arbitrary text
    for a JSON-looking substring. A second JSON value or an incomplete object is
    rejected so commentary cannot become verification evidence.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "Empty source-verifier response."
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.lstrip("\ufeff"), flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(cleaned.lstrip())
    except json.JSONDecodeError as exc:
        return None, f"Malformed source-verifier JSON: {exc.msg}."
    if not isinstance(value, dict):
        return None, "Source-verifier response must be one JSON object."
    trailing = cleaned.lstrip()[end:].strip()
    if trailing.startswith("```"):
        trailing = trailing[3:].strip()
    if trailing and (trailing.startswith("{") or trailing.startswith("[") or trailing.startswith("```") or not re.search(r"[A-Za-z]", trailing)):
        return None, "Source-verifier response contains a second JSON value or ambiguous trailing content."
    return value, None

SOURCE_VERIFICATION_PROMPT = """You are performing SOURCE-GROUNDED EXTRACTION VERIFICATION, also called SOURCE CONSISTENCY CHECKING.

Carefully inspect the original document image page or pages and compare the supplied extracted JSON with content visibly present in the source. Recursively inspect every field, object, and array that exists in the supplied JSON. The JSON structure is dynamic and may contain arbitrary field names and nesting.

Check whether extracted labels and values are visibly supported. Pay close attention to dates, times, numbers, units, names when visible, medication names/dosages/instructions when visible, laboratory values when visible, tables, repeated records, document type, and relationships between labels and values. Identify clearly supported information, conflicts with the source, unsupported information, values that cannot be confidently read, unreadable source content, ambiguous handwriting, and uncertain visual evidence.

Do not invent missing information. Do not correct or rewrite the supplied extraction. Do not provide replacement values. Do not diagnose the patient or provide medical advice. If any supplied value is conflicting, unsupported, unclear, ambiguous, or cannot be checked reliably, report it as an issue. Return one JSON object only, with this shape:
{"verification_status":"VERIFIED" or "REVIEW_REQUIRED","confidence":0.0,"issues":[{"path":"dynamic.path[0]","extracted_value":"value as supplied","source_observation":"brief observation about visible source","reason":"why it is or is not sufficiently supported"}]}

Use VERIFIED only when the extracted information is sufficiently supported by the visible source and there are no significant issues. Otherwise use REVIEW_REQUIRED. Confidence must be a number from 0 to 1. Use an empty issues array only when verified."""


class SourceVerifier:
    """Compare extracted JSON against original image pages with configured Ollama Qwen."""

    def __init__(self, extractor: Optional[OllamaExtractor] = None):
        self.extractor = extractor or OllamaExtractor()

    def verify(self, image: ImageInput, extracted_json: Dict[str, Any]) -> Dict[str, Any]:
        """Return a separate consistency result without changing ``extracted_json``."""
        try:
            pages = load_document_pages(image)
            if len(pages) > 20:
                return self._unavailable("PDF exceeds the 20-page verification limit.")
            request_text = (
                f"Verify this extracted JSON against the supplied original page image(s):\n"
                f"{json.dumps(extracted_json, ensure_ascii=False, separators=(',', ':'))}"
            )
            generated = self.extractor.generate_raw_text(pages, SOURCE_VERIFICATION_PROMPT + "\n\n" + request_text)
            if not generated.get("success"):
                return self._unavailable(generated.get("error") or "Ollama verification request failed.")
            parsed, error = _parse_source_verifier_response(generated.get("text") or "")
            if not isinstance(parsed, dict):
                return self._unavailable(error or "Verifier response was not a JSON object.")
            # A syntactically valid but incomplete/unrecognized response is a
            # verifier failure, not evidence that the extraction is correct.
            if not isinstance(parsed.get("issues"), list):
                return self._unavailable("Verifier response did not include a valid issues list.")
            if parsed.get("verification_status") not in {"VERIFIED", "REVIEW_REQUIRED"}:
                return self._unavailable("Verifier response did not include a recognized verification status.")
            return self._normalize_result(parsed)
        except Exception as exc:  # Verification must never interrupt document extraction.
            logger.warning("Source verification failed safely: %s", type(exc).__name__)
            return self._unavailable("Source verification could not be completed.")

    @staticmethod
    def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
        status = result.get("verification_status")
        issues = result.get("issues")
        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not 0.0 <= confidence <= 1.0:
            confidence = 0.0
        if status != "VERIFIED" or not isinstance(issues, list) or issues:
            status = "REVIEW_REQUIRED"
        # Keep API issue entries predictable and avoid malformed model output
        # producing unserializable or opaque review details.
        normalized_issues = []
        for issue in issues:
            if isinstance(issue, dict):
                normalized_issues.append({
                    "path": str(issue.get("path", "$")),
                    "extracted_value": issue.get("extracted_value"),
                    "source_observation": str(issue.get("source_observation", "")),
                    "reason": str(issue.get("reason", "Source support could not be established.")),
                })
            else:
                normalized_issues.append({
                    "path": "$", "extracted_value": None, "source_observation": "",
                    "reason": "Verifier returned a malformed issue entry.",
                })
        return {"status": status, "confidence": confidence, "issues": normalized_issues}

    @staticmethod
    def _unavailable(reason: str) -> Dict[str, Any]:
        return {
            "status": "UNAVAILABLE",
            "confidence": 0.0,
            "issues": [{"path": "$", "source_observation": "", "reason": reason}],
        }


__all__ = ["SourceVerifier", "SOURCE_VERIFICATION_PROMPT"]
