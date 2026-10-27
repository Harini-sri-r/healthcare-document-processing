"""Optional independent VLM comparison against source images and Qwen output."""
from __future__ import annotations

import json
import time
from typing import Any

from src.config import settings
from src.extraction.json_validator import clean_and_parse_json
from src.extraction.ollama_extractor import OllamaExtractor

FIELD_STATUSES = {"CONSISTENT", "DISCREPANCY_DETECTED", "UNREADABLE", "NOT_VERIFIABLE", "REVIEW_REQUIRED"}
OVERALL_STATUSES = {"CONSISTENT", "DISCREPANCY_DETECTED", "REVIEW_REQUIRED"}


def build_verification_prompt(raw_output: str | None, extracted_json: Any | None) -> str:
    raw = raw_output if raw_output else "[Raw Qwen response is unavailable for this older document.]"
    parsed = json.dumps(extracted_json, ensure_ascii=False, indent=2) if extracted_json is not None else "[Qwen output did not parse as JSON.]"
    return f"""You are an independent document transcription verifier. Inspect the supplied original document image(s) directly; do not trust, repeat, or assume Qwen is correct. Compare every leaf field in the extracted JSON below against visible source evidence. Include every field, including nulls, nested object values, and array items. Use field_path as a JSONPath beginning with $; object keys should use bracket quoting and array indexes [n].

Raw Qwen extraction output (untrusted):
{raw}

Parsed Qwen JSON (untrusted; may be absent):
{parsed}

Preserve exact decimals, units, dates, medication names/strength/dose/frequency, and ranges. For example 12.5 mg differs from 125 mg. Do not infer or fill values that are not visibly supported. Do not assume disagreement means you are right. If text is unclear, report UNREADABLE and leave observed_source_value null. If the image does not contain enough evidence, use NOT_VERIFIABLE. Use DISCREPANCY_DETECTED only when visible evidence conflicts; use REVIEW_REQUIRED when ambiguous. Include concise reason, observed_source_value only when actually legible, and optional page_number/image_region reference. Include proposed_value only if an alternative value is directly legible in the source; otherwise null. Never mark anything medically VERIFIED.

Return only a JSON object with exactly this general structure (values generated from this document):
{{"overall_status":"CONSISTENT|DISCREPANCY_DETECTED|REVIEW_REQUIRED","field_reviews":[{{"field_path":"$[\\\"key\\\"]","extracted_value":null,"status":"CONSISTENT|DISCREPANCY_DETECTED|UNREADABLE|NOT_VERIFIABLE|REVIEW_REQUIRED","observed_source_value":null,"proposed_value":null,"reason":"...","page_number":null,"image_region":null}}],"review_notes":[]}}"""


def _leaf_paths(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(_leaf_paths(child, f"{path}[{json.dumps(str(key), ensure_ascii=False)}]"))
    elif isinstance(value, list):
        for index, child in enumerate(value): result.extend(_leaf_paths(child, f"{path}[{index}]"))
    else:
        result.append((path, value))
    return result


class SecondModelVerifier:
    def __init__(self, model: str | None = None, extractor: Any | None = None, timeout: int | None = None):
        self.model = model or settings.SECOND_MODEL_VERIFICATION_MODEL
        self.extractor = extractor or OllamaExtractor(
            model=self.model,
            timeout=timeout if timeout is not None else settings.SECOND_MODEL_VERIFICATION_TIMEOUT,
            num_predict=settings.SECOND_MODEL_VERIFICATION_NUM_PREDICT,
            max_retries=0,
            keep_alive=0,
        )

    def verify(self, images: list[Any], raw_output: str | None, extracted_json: Any | None) -> dict[str, Any]:
        started = time.perf_counter()
        base = {"verification_model": self.model, "overall_status": "REVIEW_REQUIRED", "field_reviews": [],
                "review_notes": [], "availability": "AVAILABLE"}
        if not images:
            return {**base, "availability": "UNAVAILABLE", "review_notes": ["Original document image is unavailable."], "latency_ms": 0.0}
        if extracted_json is None:
            # Still inspect source and the raw output when JSON parsing failed, but no field-level comparison is possible.
            base["review_notes"].append("Qwen output was not valid JSON; field-level comparison requires human review.")
        prompt = build_verification_prompt(raw_output, extracted_json)
        try:
            generated = self.extractor.generate_raw_text(images, prompt)
        except Exception:
            return {**base, "availability": "UNAVAILABLE", "review_notes": ["Second-model inference failed; human review is required."],
                    "latency_ms": (time.perf_counter() - started) * 1000}
        if not generated or not generated.get("success"):
            return {**base, "availability": "UNAVAILABLE", "review_notes": ["Second model is unavailable or inference timed out; human review is required."],
                    "latency_ms": (time.perf_counter() - started) * 1000}
        raw_report = generated.get("text")
        parsed, _error = clean_and_parse_json(raw_report)
        if not isinstance(parsed, dict):
            return {**base, "availability": "UNAVAILABLE", "review_notes": ["Verifier returned invalid JSON; human review is required."],
                    "raw_verifier_output": raw_report, "latency_ms": (time.perf_counter() - started) * 1000}
        status = parsed.get("overall_status")
        fields = parsed.get("field_reviews")
        notes = parsed.get("review_notes", [])
        if status not in OVERALL_STATUSES or not isinstance(fields, list) or not isinstance(notes, list):
            return {**base, "availability": "UNAVAILABLE", "review_notes": ["Verifier response is incomplete or malformed; human review is required."],
                    "raw_verifier_output": raw_report, "latency_ms": (time.perf_counter() - started) * 1000}
        expected = dict(_leaf_paths(extracted_json)) if extracted_json is not None else {}
        normalized, seen, malformed = [], set(), False
        for item in fields:
            if not isinstance(item, dict) or not isinstance(item.get("field_path"), str) or item.get("status") not in FIELD_STATUSES:
                malformed = True; continue
            path = item["field_path"]
            if path not in expected or path in seen:
                malformed = True; continue
            seen.add(path)
            entry = {"field_path": path, "extracted_value": expected[path], "status": item["status"],
                     "observed_source_value": item.get("observed_source_value") if item["status"] == "DISCREPANCY_DETECTED" else None,
                     "reason": str(item.get("reason") or "No reason provided.")[:1000]}
            if item.get("proposed_value") is not None:
                entry["proposed_value"] = item["proposed_value"]
            for optional in ("page_number", "image_region"):
                if item.get(optional) is not None: entry[optional] = item[optional]
            normalized.append(entry)
        for path, value in expected.items():
            if path not in seen:
                normalized.append({"field_path": path, "extracted_value": value, "status": "REVIEW_REQUIRED",
                    "observed_source_value": None, "reason": "Verifier omitted this extracted field; human review is required."})
        if malformed or any(f["status"] == "DISCREPANCY_DETECTED" for f in normalized):
            status = "DISCREPANCY_DETECTED" if any(f["status"] == "DISCREPANCY_DETECTED" for f in normalized) else "REVIEW_REQUIRED"
        elif status == "CONSISTENT" and any(f["status"] in {"UNREADABLE", "NOT_VERIFIABLE", "REVIEW_REQUIRED"} for f in normalized):
            status = "REVIEW_REQUIRED"
        if extracted_json is None or malformed or len(seen) != len(expected): status = "REVIEW_REQUIRED"
        generated_notes = [str(n)[:1000] for n in notes if isinstance(n, str)]
        return {"verification_model": self.model, "overall_status": status, "field_reviews": normalized,
                "review_notes": base["review_notes"] + generated_notes,
                "availability": "AVAILABLE", "latency_ms": (time.perf_counter() - started) * 1000,
                "raw_verifier_output": raw_report}
