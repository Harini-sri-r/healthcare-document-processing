"""Ollama-based vision-language extractor for medical documents.

Uses the Ollama local HTTP API to run Qwen2.5-VL-3B inference
without loading the full Transformers model into CPU RAM.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

from src.config import settings
from src.extraction.json_validator import clean_and_parse_json, normalize_empty_strings_to_none
from src.extraction.prompts import build_medical_extraction_prompt
from src.preprocessing.image_preprocessing import ImageInput, load_document_pages
from src.utils.logging_utils import get_logger

logger = get_logger("ollama_extractor")


class OllamaExtractor:
    """Vision-Language extractor using Ollama's local API with qwen2.5vl:3b.

    Communicates with a locally running Ollama server over HTTP.
    Produces the same result structure as QwenExtractor for pipeline compatibility.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[Any] = None,
        max_retries: Optional[int] = None,
        num_predict: Optional[int] = None,
        preserve_raw_output: bool = False,
        keep_alive: Any = None,
        overall_timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout if timeout is not None else settings.OLLAMA_TIMEOUT
        self.max_retries = max(0, max_retries if max_retries is not None else settings.QWEN_MAX_RETRIES)
        self.num_predict = num_predict if num_predict is not None else settings.OLLAMA_NUM_PREDICT
        self.preserve_raw_output = preserve_raw_output
        self.keep_alive = keep_alive
        self.overall_timeout = overall_timeout if overall_timeout is not None else settings.OLLAMA_EXTRACTION_MAX_DURATION

        logger.info(
            f"OllamaExtractor initialized: model={self.model}, "
            f"base_url={self.base_url}, timeout={self.timeout}s"
        )

    @property
    def model_name(self) -> str:
        """Pipeline compatibility property returning the model identifier."""
        return self.model

    # ------------------------------------------------------------------
    # Connection health check
    # ------------------------------------------------------------------

    def check_connection(self) -> Dict[str, Any]:
        """Verify that the Ollama server is reachable and the model is available.

        Returns:
            dict with keys: reachable (bool), model_available (bool), error (str|None)
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.ConnectionError:
            return {
                "reachable": False,
                "model_available": False,
                "error": f"Cannot connect to Ollama at {self.base_url}. Is 'ollama serve' running?",
            }
        except requests.RequestException as err:
            return {
                "reachable": False,
                "model_available": False,
                "error": f"Ollama health check failed: {err}",
            }

        try:
            data = resp.json()
            available_models = [
                m.get("name", "") for m in data.get("models", [])
            ]
            # Ollama model names may include `:latest` suffix
            model_found = any(
                self.model == name or self.model == name.split(":")[0]
                for name in available_models
            )
            if not model_found:
                return {
                    "reachable": True,
                    "model_available": False,
                    "error": (
                        f"Model '{self.model}' not found in Ollama. "
                        f"Available: {available_models}. "
                        f"Run: ollama pull {self.model}"
                    ),
                }
            return {"reachable": True, "model_available": True, "error": None}
        except Exception as err:
            return {
                "reachable": True,
                "model_available": False,
                "error": f"Failed to parse Ollama model list: {err}",
            }

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode a PIL image to a base64 string for the Ollama API."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------
    # Raw generation via Ollama /api/chat
    # ------------------------------------------------------------------

    def generate_raw_text(
        self,
        image: Any,
        prompt: str,
        timeout: Optional[Any] = None,
    ) -> Dict[str, Optional[str]]:
        """Send an image and prompt to Ollama and return raw text output.

        Returns:
            dict with keys: success (bool), text (str|None), error (str|None)
        """
        images = image if isinstance(image, list) else [image]
        image_b64 = [self._encode_image(page) for page in images]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_b64,
                }
            ],
            "stream": False,
            # Ollama constrained JSON mode preserves dynamic document-specific keys.
            "format": "json",
            # Ollama otherwise starts this model with a 4096-token context.
            # Full-page documents can exceed that before Qwen emits any JSON.
            "options": {
                "num_ctx": settings.OLLAMA_NUM_CTX,
                "num_predict": self.num_predict,
            },
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        return self._send_chat_payload(payload, timeout=timeout)

    def unload_model(self) -> bool:
        """Ask Ollama to release this model before loading another large VLM."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=30,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            logger.warning("Ollama model release request failed for configured model.")
            return False

    def _send_chat_payload(self, payload: Dict[str, Any], timeout: Optional[Any] = None) -> Dict[str, Optional[str]]:
        """Submit one Ollama chat payload and return its complete text response."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout if timeout is not None else self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            msg = f"Cannot connect to Ollama at {self.base_url}. Is 'ollama serve' running?"
            logger.error(msg)
            return {"success": False, "text": None, "error": msg}
        except requests.Timeout:
            msg = f"Ollama request timed out after {timeout if timeout is not None else self.timeout}s."
            logger.error(msg)
            return {"success": False, "text": None, "error": msg}
        except requests.RequestException as err:
            # Ollama reports the useful reason for a 4xx response in its body.
            # Keep that reason in both the log and retry history; ``raise_for_status``
            # alone only exposes the generic HTTP status text.
            response_body = ""
            if err.response is not None:
                try:
                    response_body = err.response.text.strip()
                except Exception:  # pragma: no cover - defensive for custom responses
                    response_body = ""
            msg = f"Ollama HTTP error: {err}"
            if response_body:
                msg += f". Response body: {response_body}"
            logger.error(msg)
            return {"success": False, "text": None, "error": msg}

        try:
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            if not content or not content.strip():
                return {
                    "success": False,
                    "text": None,
                    "error": "Ollama returned an empty response.",
                }
            return {"success": True, "text": content, "error": None}
        except Exception as err:
            msg = f"Failed to parse Ollama response: {err}"
            logger.error(msg)
            return {"success": False, "text": None, "error": msg}

    # ------------------------------------------------------------------
    # Structured extraction with retry
    # ------------------------------------------------------------------

    def extract(
        self,
        image: ImageInput,
        prompt: Optional[str] = None,
        document_type: str = "Lab_Report",
    ) -> Dict[str, Any]:
        """Extract structured medical information from a document image.

        Args:
            image: Image file path, Path object, or PIL Image.
            prompt: Optional extraction prompt override.

        Returns:
            Dictionary with:
            - success: bool
            - attempts: int
            - data: dict (parsed JSON) or None
            - history: list of attempt records
            - error: str or None
        """
        try:
            pil_images = load_document_pages(image)
        except Exception as err:
            logger.warning("Document rendering failed: %s", type(err).__name__)
            return {
                "success": False,
                "attempts": 0,
                "data": None,
                "history": [],
                "error": "Failed to load image or render document input.",
            }

        if len(pil_images) > 20:
            return {"success": False, "attempts": 0, "data": None, "history": [],
                    "error": "PDF exceeds the 20-page inference limit; split it for review."}

        extraction_prompt = prompt or build_medical_extraction_prompt(document_type)
        history: List[Dict[str, Any]] = []
        total_allowed_attempts = self.max_retries + 1
        deadline = time.monotonic() + max(1, self.overall_timeout)
        retry_prompt = extraction_prompt
        raw_text = None

        for attempt in range(1, total_allowed_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                history.append({"attempt": attempt, "valid_json": False, "error": "Overall extraction timeout exceeded."})
                break
            request_budget = min(self.timeout, max(0.2, remaining))
            # requests' scalar timeout applies separately to connect and read.
            # Split one attempt's budget so the two phases together fit the
            # remaining extraction deadline as closely as requests allows.
            connect_timeout = min(10.0, request_budget / 2)
            request_timeout = (connect_timeout, max(0.1, request_budget - connect_timeout))
            attempt_started = time.monotonic()
            logger.info("Extraction attempt %s/%s", attempt, total_allowed_attempts)
            gen_result = self.generate_raw_text(pil_images, retry_prompt, timeout=request_timeout)
            duration_ms = (time.monotonic() - attempt_started) * 1000

            if not gen_result["success"]:
                history.append({
                    "attempt": attempt,
                    "valid_json": False,
                    "error": gen_result["error"],
                    "duration_ms": round(duration_ms, 1),
                })
                logger.info("Extraction attempt %s completed duration_ms=%.1f valid_json=false", attempt, duration_ms)
                # Do not retry connectivity or local model installation failures.
                safe_error = (gen_result["error"] or "").lower()
                if "cannot connect" in safe_error or "model not found" in safe_error:
                    break
                if "timed out" in safe_error and time.monotonic() >= deadline:
                    break
                continue

            raw_text = gen_result["text"]
            parsed_json, parse_err = clean_and_parse_json(raw_text)

            attempt_record = {
                "attempt": attempt,
                "valid_json": parsed_json is not None,
                "error": parse_err,
                "duration_ms": round(duration_ms, 1),
            }

            if parsed_json is not None:
                normalized_data = normalize_empty_strings_to_none(parsed_json)
                history.append(attempt_record)
                result = {
                    "success": True,
                    "attempts": attempt,
                    "data": normalized_data,
                    "history": history,
                    "error": None,
                }
                if self.preserve_raw_output: result["raw_output"] = raw_text
                logger.info("Extraction attempt %s completed duration_ms=%.1f valid_json=true", attempt, duration_ms)
                return result

            # Raw response can contain PHI; log only the attempt number and safe status.
            logger.warning("Qwen response failed JSON parsing on attempt %s/%s.",
                           attempt, total_allowed_attempts)
            history.append(attempt_record)
            logger.info("Extraction attempt %s completed duration_ms=%.1f valid_json=false", attempt, duration_ms)
            retry_prompt = (
                extraction_prompt
                + "\n\nYour previous response was not valid JSON. Parser feedback: "
                + str(parse_err)
                + " Return the complete extraction again as one valid JSON object. Preserve every value from the document; do not guess or add missing values."
            )

        result = {
            "success": False,
            "attempts": len(history),
            "data": None,
            "history": history,
            "error": history[-1].get("error") if history else "Failed to extract valid JSON within retry limit.",
        }
        logger.error("Extraction failed after %s attempts: %s", len(history), result["error"])
        if self.preserve_raw_output: result["raw_output"] = raw_text if "raw_text" in locals() else None
        return result


__all__ = ["OllamaExtractor"]
