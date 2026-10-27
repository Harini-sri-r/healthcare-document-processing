from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from src.classification.classification import DocumentClassifier
from src.config import settings
from src.extraction.json_validator import JSONValidator
from src.extraction.ollama_extractor import OllamaExtractor
from src.preprocessing.image_preprocessing import ImageInput, load_document, load_document_pages, prepare_for_classifier
from src.utils.logging_utils import get_logger
from src.validation.medical_document_gate import MedicalDocumentGate, SUPPORTED_DOCUMENT_TYPES

logger = get_logger("pipeline")


class MedicalDocumentPipeline:
    """End-to-end production AI pipeline for medical documents:

    Medical Document Image
            ↓
    Image Preprocessing
            ↓
    ResNet-18 Document Classifier
            ↓
    Document Type + Confidence
            ↓
    Qwen2.5-VL-3B via Ollama (default) or Transformers
            ↓
    Dynamic Medical Information Extraction
            ↓
    JSON Validation
            ↓
    Final Structured Result
    """

    def __init__(
        self,
        resnet_model_path: Optional[Union[str, Path]] = None,
        qwen_model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        classification_threshold: Optional[float] = None,
        extractor: Optional[Any] = None,
        source_verifier: Optional[Any] = None,
        classifier: Optional[DocumentClassifier] = None,
        extractor_backend: str = "ollama",
        medical_document_gate: Optional[MedicalDocumentGate] = None,
        second_model_verifier: Optional[Any] = None,
    ):
        self.device = device or settings.DEVICE
        self.classification_threshold = (
            classification_threshold
            if classification_threshold is not None
            else settings.CLASSIFICATION_CONFIDENCE_THRESHOLD
        )
        self.medical_document_gate = medical_document_gate or MedicalDocumentGate()

        # ResNet Classifier
        self.classifier = classifier or DocumentClassifier(
            model_path=resnet_model_path or settings.RESNET_V3_MODEL_PATH,
            device=self.device,
        )

        # Production extraction uses the local Ollama vision endpoint.
        if extractor is not None:
            self.extractor = extractor
        elif extractor_backend == "ollama":
            self.extractor = OllamaExtractor(preserve_raw_output=True)
        else:
            raise ValueError("Production extraction backend must be 'ollama'.")

        # JSON Validator
        self.validator = JSONValidator()
        # Kept as ignored constructor arguments for compatibility with callers
        # that inject legacy verifier doubles. The processing path ends at Qwen.

    def process(
        self,
        image: ImageInput,
        run_extraction: bool = True,
    ) -> Dict[str, Any]:
        """Process a medical document image through the complete architecture.

        Args:
            image: Path to image, Path object, or PIL Image.
            run_extraction: Deprecated compatibility argument. Supported medical
                documents always continue to Qwen; Non_Medical never does.

        Returns:
            Structured dictionary containing:
            - document_type: str
            - confidence: float
            - classification: dict
            - extracted_json: dict or None
            - extraction: dict
            - validation: dict
            - review: dict
            - errors: list
        """
        errors: List[Dict[str, str]] = []
        review_reasons: List[str] = []
        timings = {"classification_ms": 0.0, "qwen_extraction_ms": 0.0}

        # 1. Load document
        document_ref = _document_event_id(image)
        logger.info("document_id=%s event=processing_started", document_ref)
        try:
            pil_image = load_document(image)
            extraction_images = load_document_pages(image)
        except Exception as err:
            logger.error("Document load/preprocessing failed: %s", type(err).__name__)
            return {
                "document_type": None,
                "confidence": 0.0,
                "classification": None,
                "extracted_json": None,
                "extraction": {"success": False, "data": None, "attempts": 0, "history": []},
                "validation": {"valid": False, "issue_count": 1, "issues": [{"type": "load_error", "message": str(err)}]},
                "review": {"status": "FAILED", "reasons": ["Document could not be loaded or preprocessed"]},
                "errors": [{"code": "image_load_failed", "message": "Document could not be loaded or preprocessed."}],
            }

        input_type = "pdf" if isinstance(image, (str, Path)) and Path(image).suffix.lower() == ".pdf" else "image"
        # Classify each actual page. Mean page probabilities provide a stable
        # document-level decision for multi-page PDFs.
        classification_started = time.perf_counter()
        page_classifications = [
            self.classifier.predict(prepare_for_classifier(page))
            for page in extraction_images
        ]
        class_names = set().union(*(prediction.get("probabilities", {}).keys() for prediction in page_classifications))
        if class_names:
            probabilities = {
                name: sum(float(p.get("probabilities", {}).get(name, 0.0)) for p in page_classifications) / len(page_classifications)
                for name in class_names
            }
            document_type = max(probabilities, key=probabilities.get)
            confidence = probabilities[document_type]
            classification = dict(page_classifications[0])
            classification.update({"predicted_class": document_type, "document_type": document_type,
                                   "confidence": confidence, "probabilities": probabilities,
                                   "page_classifications": page_classifications})
        else:
            # Compatibility for injected legacy test classifiers without probabilities.
            classification = page_classifications[0]
        confidence = classification["confidence"]
        if not timings["classification_ms"]:
            timings["classification_ms"] = round((time.perf_counter() - classification_started) * 1000, 1)
        timings["classification_ms"] = round((time.perf_counter() - classification_started) * 1000, 1)
        document_type = classification.get("predicted_class", classification["document_type"])

        # 3. Gate strictly on the class predicted by ResNet v2. Only the three
        # supported medical classes continue to extraction.
        gate_result = self.medical_document_gate.evaluate(classification)
        logger.info(
            "document_id=%s input_type=%s gate_decision=%s gate_confidence=%.4f rejection_reason=%s resnet_invoked=true qwen_invoked=%s",
            document_ref, input_type,
            "ACCEPTED" if gate_result.is_supported_medical_document else "REJECTED",
            gate_result.confidence, gate_result.rejection_reason,
            gate_result.is_supported_medical_document,
        )
        if not gate_result.is_supported_medical_document:
            message = "The uploaded document is not a supported medical document."
            return {
                "accepted": False,
                "qwen_invoked": False,
                "status": "REJECTED",
                "document_type": gate_result.predicted_class,
                "confidence": gate_result.confidence,
                "classification_confidence": gate_result.confidence,
                "predicted_class": gate_result.predicted_class,
                "classification": classification,
                "classification_result": {
                    "is_medical": False,
                    "document_type": gate_result.predicted_class,
                    "confidence": gate_result.confidence,
                },
                "evaluation": {
                    "qwen_invoked": False,
                    "status": "REJECTED",
                    "validation_status": "SKIPPED",
                    "extracted_json": None,
                },
                "extracted_json": None,
                "extraction": {"status": "SKIPPED", "success": False, "data": None, "attempts": 0, "history": [], "error": message},
                "validation": {"valid": False, "issue_count": 1, "issues": [{"type": "unsupported_document", "message": message}], "requires_review": False},
                "review": {"status": "REJECTED", "reasons": [gate_result.rejection_reason or message]},
                "errors": [{"code": "unsupported_document", "message": message}],
                "supported_document_types": list(SUPPORTED_DOCUMENT_TYPES),
                "rejection_reason": gate_result.rejection_reason,
                "medical_document_gate": gate_result.to_dict(),
            }

        logger.info(
            f"Classification complete: type='{document_type}', "
            f"confidence={confidence:.4f} (threshold={self.classification_threshold:.2f})"
        )

        if confidence < self.classification_threshold:
            review_reasons.append("Low document-classification confidence")
        logger.info("document_id=%s classifier_type=%s confidence=%.4f", document_ref, document_type, confidence)

        # 3. Vision-Language Extraction
        qwen_invoked = False
        if isinstance(self.extractor, OllamaExtractor) and hasattr(self.extractor, "check_connection"):
            health = self.extractor.check_connection()
            if not health.get("reachable") or not health.get("model_available"):
                err_msg = health.get("error") or "Ollama health check failed."
                logger.error(err_msg)
                extraction = {
                    "success": False,
                    "attempts": 0,
                    "data": None,
                    "history": [],
                    "error": err_msg,
                }
                review_reasons.append("Ollama health check failed")
                if "Cannot connect" in err_msg:
                    errors.append({"code": "ollama_connection_error", "message": err_msg})
                else:
                    errors.append({"code": "ollama_model_missing", "message": err_msg})
            else:
                qwen_invoked = True
                extraction_started = time.perf_counter()
                extraction = self._extract(extraction_images if len(extraction_images) > 1 else pil_image, document_type)
                timings["qwen_extraction_ms"] = round((time.perf_counter() - extraction_started) * 1000, 1)
        else:
            qwen_invoked = True
            extraction_started = time.perf_counter()
            extraction = self._extract(extraction_images if len(extraction_images) > 1 else pil_image, document_type)
            timings["qwen_extraction_ms"] = round((time.perf_counter() - extraction_started) * 1000, 1)

        if not extraction["success"] and not errors:
            review_reasons.append("JSON extraction failed")
            if extraction.get("error"):
                errors.append({"code": "extraction_error", "message": extraction["error"]})
        # 4. JSON Validation
        # Injected legacy extractors are retained for compatibility; production
        # Ollama output is always validated against the classifier-selected schema.
        validation = self.validator.validate(extraction, document_type=document_type)
        if validation["requires_review"]:
            review_reasons.append("Extraction validation issues detected")

        if extraction.get("attempts", 0) > 1:
            review_reasons.append("Extraction required retry")
            validation["requires_review"] = True
        logger.info("document_id=%s extraction_complete type=%s attempts=%s validation=%s review_required=%s",
                    document_ref,
                    document_type, extraction.get("attempts", 0),
                    "VALID" if validation.get("valid") else "INVALID",
                    validation.get("requires_review"))
        if extraction.get("success") and not validation.get("valid", False):
            review_reasons.append("Structured extraction is invalid")
        elif extraction.get("success") and validation.get("requires_review"):
            review_reasons.append("Extraction contains uncertain fields")

        # 5. Determine Review Status
        if (not extraction["success"] or validation["requires_review"]
              or confidence < self.classification_threshold or review_reasons):
            review_status = "HUMAN_REVIEW"
        else:
            review_status = "AUTO_ACCEPT"
        review_fields = []
        if validation.get("requires_review"):
            for issue in validation.get("issues", []):
                if isinstance(issue, dict):
                    review_fields.append({
                        "field_path": issue.get("field") or issue.get("path") or "$",
                        "status": "REVIEW_REQUIRED",
                        "reason": issue.get("message") or issue.get("reason") or "Extraction validation requires review.",
                    })
        if review_status == "HUMAN_REVIEW" and not review_fields:
            review_fields.append({"field_path": "$", "status": "REVIEW_REQUIRED",
                                  "reason": "Pipeline checks require human review."})
        logger.info("document_id=%s review_required=%s", document_ref, review_status == "HUMAN_REVIEW")

        return {
            "accepted": True,
            "qwen_invoked": qwen_invoked,
            "status": "ACCEPTED",
            "document_type": document_type,
            "confidence": confidence,
            "classification_confidence": confidence,
            "classification": classification,
            "classification_result": {
                "is_medical": True,
                "document_type": document_type,
                "confidence": confidence,
            },
            "evaluation": {
                "qwen_invoked": qwen_invoked,
                "status": "ACCEPTED",
                "validation_status": "VALID" if validation.get("valid", False) else "INVALID",
                "extracted_json": extraction.get("data"),
            },
            "extracted_json": extraction.get("data"),
            "extracted_data": extraction.get("data"),
            "extraction": extraction,
            "validation": validation,
            "review": {
                "status": review_status,
                "reasons": review_reasons,
            },
            "errors": errors,
            "medical_document_gate": gate_result.to_dict(),
            "timings": timings,
        }

    def _extract(self, image: Any, document_type: str) -> Dict[str, Any]:
        """Pass the classifier result to Ollama; retain compatibility with injected test doubles."""
        if isinstance(self.extractor, OllamaExtractor):
            return self.extractor.extract(image, document_type=document_type)
        if isinstance(image, list):
            # Test doubles/legacy extractors accept one image; production Ollama
            # receives the entire page sequence above.
            return self.extractor.extract(image[0])
        return self.extractor.extract(image)


def _document_event_id(image: ImageInput) -> str:
    """Return a non-sensitive short reference suitable for correlation logs."""
    import hashlib
    source = str(image) if not hasattr(image, "size") else repr(image.size)
    return hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()[:12]


__all__ = ["MedicalDocumentPipeline"]
