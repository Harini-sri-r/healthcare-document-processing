"""Content classifier gate for supported medical document classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional



SUPPORTED_DOCUMENT_TYPES = ("Prescription", "Lab_Report", "Discharge_Summary")


@dataclass(frozen=True)
class MedicalDocumentGateResult:
    predicted_class: Optional[str]
    confidence: float
    is_supported_medical_document: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MedicalDocumentGate:
    """Accept only the three supported medical classes from ResNet v2."""

    def __init__(self, threshold: Optional[float] = None):
        # Retain the argument for callers from older versions; class membership
        # is the gate and confidence never substitutes for Non_Medical.
        self.threshold = threshold

    def evaluate(self, classification: dict[str, Any]) -> MedicalDocumentGateResult:
        predicted_class = classification.get("predicted_class", classification.get("document_type"))
        confidence = float(classification.get("confidence", 0.0))
        if predicted_class == "Non_Medical":
            return MedicalDocumentGateResult(
                predicted_class, confidence, False,
                "ResNet classified the document as Non_Medical.",
            )
        accepted = predicted_class in SUPPORTED_DOCUMENT_TYPES
        reason = None if accepted else "ResNet prediction is not a supported medical document class."
        return MedicalDocumentGateResult(predicted_class, confidence, accepted, reason)


__all__ = ["MedicalDocumentGate", "MedicalDocumentGateResult", "SUPPORTED_DOCUMENT_TYPES"]
