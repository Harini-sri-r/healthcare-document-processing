from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logging_utils import get_logger

logger = get_logger("json_validator")


def clean_and_parse_json(text: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    """Clean markdown fences and extract a JSON object or array from raw text.

    Returns:
        (parsed_json, error_message)
        If parsing succeeds, error_message is None.
        If parsing fails, parsed_json is None and error_message describes the issue.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "Empty or non-string response received."

    text = text.lstrip("\ufeff")

    # Remove markdown code fences if present (e.g. ```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()

    # Find the outermost JSON block: either object {...} or array [...]
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")

    start = -1
    end = -1

    # Check whether object or array appears first
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        if last_brace < first_brace:
            return None, "JSON response appears truncated before its final delimiter."
        if last_brace >= first_brace:
            start = first_brace
            end = last_brace
    elif first_bracket != -1:
        if last_bracket < first_bracket:
            return None, "JSON response appears truncated before its final delimiter."
        if last_bracket >= first_bracket:
            start = first_bracket
            end = last_bracket

    if start == -1 or end == -1:
        logger.warning("No JSON delimiters '{...}' or '[...]' found in model output.")
        return None, "No JSON delimiters '{...}' or '[...]' detected in output."

    if cleaned[end + 1:].strip():
        logger.warning("JSON response contains trailing non-JSON text.")
        return None, "JSON response contains trailing text."
    if cleaned[end] not in ("}", "]"):
        logger.warning("JSON response appears truncated before its final delimiter.")
        return None, "JSON response appears truncated."

    candidate_json = cleaned[start : end + 1]

    try:
        parsed = json.loads(candidate_json)
        return parsed, None
    except json.JSONDecodeError as err:
        logger.warning(f"JSON decoding error at line {err.lineno}, col {err.colno}: {err.msg}")
        repaired = repair_bare_object_strings(candidate_json)
        if repaired is not None:
            try:
                repaired_data = json.loads(repaired)
                return repaired_data, None
            except json.JSONDecodeError as repair_err:
                err = repair_err
        return None, f"JSON parse error: {err.msg} at line {err.lineno}, col {err.colno}"


def repair_bare_object_strings(candidate: str) -> Optional[str]:
    """Wrap unambiguous bare object string members as generic text/items fields."""
    out: List[str] = []
    i = 0
    stack: List[Dict[str, Any]] = []
    while i < len(candidate):
        ch = candidate[i]
        if ch == '"':
            end = _json_string_end(candidate, i)
            if end is None:
                return None
            if stack and stack[-1]["kind"] == "object" and stack[-1]["expect_member"]:
                value = _decode_json_string(candidate[i:end + 1])
                if value is None:
                    return None
                after = _skip_ws(candidate, end + 1)
                if after < len(candidate) and candidate[after] == ':':
                    out.append(candidate[i:end + 1])
                    stack[-1]["expect_member"] = False
                    i = end + 1
                    continue
                values = [value]
                cursor = after
                while cursor < len(candidate) and candidate[cursor] == ',':
                    next_start = _skip_ws(candidate, cursor + 1)
                    if next_start >= len(candidate) or candidate[next_start] != '"':
                        break
                    next_end = _json_string_end(candidate, next_start)
                    if next_end is None:
                        return None
                    next_after = _skip_ws(candidate, next_end + 1)
                    if next_after < len(candidate) and candidate[next_after] == ':':
                        break
                    next_value = _decode_json_string(candidate[next_start:next_end + 1])
                    if next_value is None:
                        return None
                    values.append(next_value)
                    cursor = next_after
                if len(values) == 1:
                    out.append('"text": ' + json.dumps(values[0], ensure_ascii=False))
                else:
                    out.append('"items": ' + json.dumps(values, ensure_ascii=False))
                stack[-1]["expect_member"] = False
                i = cursor
                continue
            out.append(candidate[i:end + 1])
            i = end + 1
            continue
        if ch == '{':
            out.append(ch)
            stack.append({"kind": "object", "expect_member": True})
        elif ch == '[':
            out.append(ch)
            stack.append({"kind": "array", "expect_member": False})
        elif ch in '}]':
            out.append(ch)
            if not stack:
                return None
            stack.pop()
            if stack and stack[-1]["kind"] == "object":
                stack[-1]["expect_member"] = False
        elif ch == ',':
            out.append(ch)
            if stack and stack[-1]["kind"] == "object":
                stack[-1]["expect_member"] = True
        else:
            out.append(ch)
            if ch == ':' and stack and stack[-1]["kind"] == "object":
                stack[-1]["expect_member"] = False
        i += 1
    if stack:
        return None
    return ''.join(out)


def _json_string_end(text: str, start: int) -> Optional[int]:
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '"':
            return index
    return None


def _decode_json_string(value: str) -> Optional[str]:
    try:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, str) else None
    except json.JSONDecodeError:
        return None


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def normalize_empty_strings_to_none(data: Any) -> Any:
    """Recursively converts empty strings to None without altering values."""
    if isinstance(data, dict):
        return {k: normalize_empty_strings_to_none(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalize_empty_strings_to_none(item) for item in data]
    if isinstance(data, str) and data.strip() == "":
        return None
    return data


class JSONValidator:
    """Validates dynamic JSON structure without requiring predefined fixed schemas.

    Rules:
    - Extraction must succeed and output a non-null JSON dictionary payload.
    - If document_type is provided or present, it must be valid.
    - Dotted field paths marked in source.uncertain_fields or explicit [UNCLEAR]
      values trigger requires_review=True.
    - Null fields, reference ranges, and empty values generate informational issues.
    """

    def validate(self, extraction_result: Dict[str, Any], document_type: Optional[str] = None) -> Dict[str, Any]:
        """Validate an extraction result dictionary dynamically.

        Args:
            extraction_result: dict containing "success", "data", etc.
            document_type: optional expected document type string.

        Returns:
            dict containing:
            - valid: bool
            - issue_count: int
            - issues: list of issue dicts
            - requires_review: bool
        """
        all_issues: List[Dict[str, Any]] = []
        blocking_issues: List[Dict[str, Any]] = []
        info_issues: List[Dict[str, Any]] = []

        if not extraction_result.get("success", False):
            msg = extraction_result.get("error") or "No valid JSON output was produced."
            issue = {"field": "$", "type": "extraction_failure", "message": msg}
            logger.error("Validation failed: %s", msg)
            all_issues.append(issue)
            blocking_issues.append(issue)
            return {
                "valid": False,
                "issue_count": len(all_issues),
                "issues": all_issues,
                "requires_review": True,
            }

        data = extraction_result.get("data")
        if data is None:
            issue = {"field": "$", "type": "null_payload", "message": "Extracted data payload is None."}
            all_issues.append(issue)
            blocking_issues.append(issue)
            return {
                "valid": False,
                "issue_count": len(all_issues),
                "issues": all_issues,
                "requires_review": True,
            }

        if not isinstance(data, dict):
            issue = {"field": "$", "type": "invalid_type", "message": "Extraction payload must be a JSON object."}
            all_issues.append(issue)
            blocking_issues.append(issue)
            return {
                "valid": False,
                "issue_count": len(all_issues),
                "issues": all_issues,
                "requires_review": True,
            }

        # Check document_type alignment if present
        doc_type_in_payload = data.get("document_type")
        effective_type = document_type or doc_type_in_payload
        if document_type and doc_type_in_payload and document_type != doc_type_in_payload:
            info_issues.append({
                "field": "document_type",
                "type": "type_mismatch",
                "message": f"Expected document_type '{document_type}', got '{doc_type_in_payload}'",
            })

        # Informational inspection: flag null fields, empty strings, unclear values
        self._inspect(data, info_issues, path="")
        all_issues.extend(info_issues)

        # Inspect uncertainty paths
        uncertainty_issues: List[Dict[str, Any]] = []
        self._inspect_uncertainty(data, uncertainty_issues)
        all_issues.extend(uncertainty_issues)

        has_uncertain = bool(
            isinstance(data, dict)
            and isinstance(data.get("source"), dict)
            and data["source"].get("uncertain_fields")
        )
        has_unclear = any(issue.get("type") in ("unclear_value", "uncertain_value") for issue in all_issues)

        requires_review = bool(blocking_issues) or bool(uncertainty_issues) or has_uncertain or has_unclear
        valid = not blocking_issues

        logger.info(
            "Dynamic validation completed: %d total issues (valid=%s, requires_review=%s).",
            len(all_issues),
            valid,
            requires_review,
        )

        return {
            "valid": valid,
            "issue_count": len(all_issues),
            "issues": all_issues,
            "requires_review": requires_review,
        }

    @staticmethod
    def _inspect_uncertainty(data: Dict[str, Any], issues: List[Dict[str, Any]]) -> None:
        source = data.get("source")
        if isinstance(source, dict) and isinstance(source.get("uncertain_fields"), list):
            for path in source["uncertain_fields"]:
                if not isinstance(path, str) or not path:
                    issues.append({
                        "field": "source.uncertain_fields",
                        "type": "invalid_type",
                        "message": "Uncertain field paths must be non-empty strings.",
                    })
                else:
                    issues.append({
                        "field": path,
                        "type": "uncertain_value",
                        "message": "Field was marked uncertain by extraction.",
                    })

    def _inspect(self, node: Any, issues: List[Dict[str, Any]], path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                current_path = f"{path}.{key}" if path else str(key)
                if value is None:
                    issues.append({
                        "field": current_path,
                        "type": "missing_value",
                        "message": "Value is null or unavailable in document.",
                    })
                elif isinstance(value, (dict, list)):
                    self._inspect(value, issues, current_path)
                elif isinstance(value, str):
                    self._inspect_string(value, current_path, issues)

        elif isinstance(node, list):
            for index, item in enumerate(node):
                current_path = f"{path}[{index}]"
                if isinstance(item, (dict, list)):
                    self._inspect(item, issues, current_path)
                elif item is None:
                    issues.append({
                        "field": current_path,
                        "type": "missing_value",
                        "message": "List item is null.",
                    })
                elif isinstance(item, str):
                    self._inspect_string(item, current_path, issues)

    def _inspect_string(self, text: str, field_path: str, issues: List[Dict[str, Any]]) -> None:
        trimmed = text.strip()
        if trimmed == "":
            issues.append({
                "field": field_path,
                "type": "empty_value",
                "message": "Field contains an empty string.",
            })
        elif trimmed.upper() in ("[UNCLEAR]", "UNCLEAR"):
            issues.append({
                "field": field_path,
                "type": "unclear_value",
                "message": "Value was explicitly marked as unreadable/unclear.",
            })
        elif self._looks_like_range(trimmed):
            issues.append({
                "field": field_path,
                "type": "reference_range",
                "message": "Value appears to contain a reference range rather than a single measurement.",
                "value": trimmed,
            })

    @staticmethod
    def _looks_like_range(text: str) -> bool:
        """Detect numeric ranges such as '8,000 - 12,000' or '1.7 - 4.3'."""
        if "-" not in text:
            return False
        parts = [p.strip().replace(",", "") for p in text.split("-")]
        if len(parts) != 2:
            return False
        try:
            float(parts[0])
            float(parts[1])
            return True
        except ValueError:
            return False


__all__ = [
    "clean_and_parse_json",
    "repair_bare_object_strings",
    "normalize_empty_strings_to_none",
    "JSONValidator",
]

