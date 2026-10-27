from pathlib import Path
import shutil
import uuid
from typing import Dict, Any, Union, Optional
import time

from sqlalchemy.orm import Session
from src.pipeline.pipeline import MedicalDocumentPipeline
from src.db.models import (Document, DocumentExtraction, DocumentExtractionRaw,
                           DocumentVerificationRun, ProcessingLog, Patient)
from src.preprocessing.image_preprocessing import ImageInput
from src.utils.logging_utils import get_logger

logger = get_logger("document_service")


def _clean_string_identifier(val: Any) -> Optional[str]:
    """Extract a clean scalar string representation from arbitrary dynamic JSON values."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        s = str(val).strip()
        return s if s else None
    if isinstance(val, dict):
        for k in ("id", "mrn", "value", "text", "number", "code", "patient_id"):
            sub = _clean_string_identifier(val.get(k))
            if sub:
                return sub
    if isinstance(val, (list, tuple)):
        for item in val:
            sub = _clean_string_identifier(item)
            if sub:
                return sub
    return None


def _sanitize_json_serializable(data: Any) -> Any:
    """Ensure data is cleanly JSON serializable for database JSON columns."""
    if data is None or isinstance(data, (int, float, bool, str)):
        return data
    if isinstance(data, dict):
        return {str(k): _sanitize_json_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [_sanitize_json_serializable(v) for v in data]
    return str(data)


def _extract_patient_info_generically(data: Any) -> Optional[Dict[str, Any]]:
    """Recursively search for a patient dictionary in dynamic JSON."""
    if not isinstance(data, dict):
        return None
    if "patient" in data and isinstance(data["patient"], dict):
        return data["patient"]
    if "patient_information" in data and isinstance(data["patient_information"], dict):
        return data["patient_information"]
    if "patient_details" in data and isinstance(data["patient_details"], dict):
        return data["patient_details"]
    if any(k in data for k in ("mrn", "patient_id", "medical_record_number")):
        return data
    for v in data.values():
        if isinstance(v, dict):
            res = _extract_patient_info_generically(v)
            if res:
                return res
    return None


def _extract_date_generically(data: Any) -> Optional[str]:
    """Recursively search for a date string in dynamic extracted JSON."""
    if not isinstance(data, (dict, list)):
        return None
    date_keys = (
        "date", "document_date", "prescription_date", "reported_at",
        "sample_collected_at", "collected_on", "discharge_date",
        "admission_date", "report_date", "visit_date"
    )
    if isinstance(data, dict):
        for key in date_keys:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for v in data.values():
            val = _extract_date_generically(v)
            if val:
                return val
    elif isinstance(data, list):
        for item in data:
            val = _extract_date_generically(item)
            if val:
                return val
    return None


class DocumentService:

    def __init__(self, db_session: Session, pipeline: MedicalDocumentPipeline):
        self.db = db_session
        self.pipeline = pipeline

    def process_and_store(
        self,
        image: ImageInput,
        original_filename: str,
        run_extraction: bool = True
    ) -> Document:
        """Runs the pipeline and stores the result in the database cleanly."""
        start_time = time.time()
        logger.info("document_processing_started extension=%s", Path(original_filename).suffix.lower())

        # 1. Run the Pipeline
        result = self.pipeline.process(image, run_extraction=run_extraction)
        # Legacy injected pipelines may omit the explicit flag while returning
        # a successful extraction; preserve their historical persistence.
        qwen_invoked = bool(result.get("qwen_invoked", result.get("extraction", {}).get("success", False)))
        duration_ms = (time.time() - start_time) * 1000

        # A rejected upload must not be retained as a medical record or create
        # an extraction row. The API can return the structured gate response.
        if result.get("status") == "REJECTED":
            logger.info("document_gate_rejected input_type=%s confidence=%.4f reason=%s pipeline_invoked=false",
                        Path(str(image)).suffix.lower() or "image",
                        float(result.get("confidence", 0.0)),
                        (result.get("medical_document_gate") or {}).get("rejection_reason"))
            return result

        # 2. Extract key fields
        classification = result.get("classification") or {}
        doc_type = result.get("document_type") or "Healthcare_Document"
        confidence = float(result.get("confidence", 0.0))
        extracted_json = result.get("extracted_json")
        validation = result.get("validation") or {"valid": False, "requires_review": True}
        review = result.get("review") or {"status": "HUMAN_REVIEW", "reasons": []}
        extraction_meta = result.get("extraction") or {}
        review_reasons = list(review.get("reasons", []))

        # 3. Patient Linking (Generic)
        patient_id = None
        patient_data = _extract_patient_info_generically(extracted_json)
        if patient_data:
            raw_id = patient_data.get("id") or patient_data.get("mrn")
            for extra_key in ("patient_id", "medical_record_number", "record_number"):
                if not raw_id:
                    raw_id = patient_data.get(extra_key)
            identifier = _clean_string_identifier(raw_id)
            if identifier:
                try:
                    patient = self.db.query(Patient).filter(Patient.id == identifier).first()
                    if not patient:
                        patient_name = _clean_string_identifier(patient_data.get("name") or patient_data.get("patient_name"))
                        dob = _clean_string_identifier(patient_data.get("dob") or patient_data.get("date_of_birth"))
                        gender = _clean_string_identifier(patient_data.get("gender") or patient_data.get("sex"))
                        patient = Patient(
                            id=identifier,
                            patient_name=patient_name,
                            date_of_birth=dob,
                            gender=gender
                        )
                        self.db.add(patient)
                        self.db.flush()
                    patient_id = patient.id
                except Exception as patient_err:
                    logger.warning("Patient linking failed error_type=%s", type(patient_err).__name__)
                    self.db.rollback()
                    patient_id = None

        # 4. Document Date (Generic)
        document_date = _clean_string_identifier(_extract_date_generically(extracted_json))

        # 5. Create Document record
        document_id = uuid.uuid4().hex
        original_root = Path(__file__).resolve().parents[2] / "data" / "originals"
        original_root.mkdir(parents=True, exist_ok=True)
        source_dir = original_root / document_id[:2]
        source_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(original_filename).suffix.lower()
        source_path = source_dir / f"{document_id}{suffix}"
        try:
            if hasattr(image, "save"):
                image.save(source_path)
            else:
                shutil.copy2(Path(image), source_path)
        except Exception as err:

            logger.error("Original document retention failed id=%s error_type=%s: %s", document_id, type(err).__name__, err, exc_info=True)
            raise

        is_success = bool(extraction_meta.get("success", False))
        ext_status = (
            "SKIPPED" if not qwen_invoked else
            "REVIEW_REQUIRED" if review.get("status") == "HUMAN_REVIEW" or not validation.get("valid", False) else
            "SUCCESS" if is_success else "FAILED"
        )

        document = Document(
            id=document_id,
            patient_id=patient_id,
            document_type=doc_type,
            original_filename=original_filename,
            original_document_path=str(source_path),
            document_date=document_date,
            classifier_confidence=confidence,
            extraction_status=ext_status,
            extraction_processing_status=("SKIPPED" if not qwen_invoked else
                                          "SUCCESS" if is_success else "FAILED"),
            processing_status="COMPLETED",
            validation_status="VALID" if validation.get("valid", False) else "INVALID",
            review_required=bool(review.get("status") == "HUMAN_REVIEW" or validation.get("requires_review", False)),
            review_reasons=_sanitize_json_serializable(review_reasons)
        )
        # Keep request-scoped pipeline details available to the API without
        # adding columns or changing the existing database schema.
        document._pipeline_result = result
        self.db.add(document)
        self.db.flush()

        # 6. Create DocumentExtraction record
        if qwen_invoked:
            extraction_record = DocumentExtraction(
                document_id=document.id,
                extraction_json=_sanitize_json_serializable(extracted_json or {}),
                normalized_data=None
            )
            self.db.add(extraction_record)
            extraction_result = result.get("extraction") or {}
            raw_extraction = extraction_result.get("raw_output")
            self.db.add(DocumentExtractionRaw(document_id=document.id,
                raw_text=raw_extraction if isinstance(raw_extraction, str) else None,
                json_valid=bool(extraction_result.get("success") and extraction_result.get("data") is not None)))

        # 7. Create ProcessingLog record
        model_name = getattr(self.pipeline.extractor, "model_name", "qwen2.5-vl") if qwen_invoked else "resnet18-only"
        if not isinstance(model_name, str):
            model_name = str(model_name)
        log_record = ProcessingLog(
            document_id=document.id,
            model_used=model_name,
            processing_duration_ms=duration_ms,
            validation_status="VALID" if validation.get("valid", False) else "INVALID",
            retry_count=int(extraction_meta.get("attempts", 0)),
            review_required=document.review_required
        )
        self.db.add(log_record)


        try:
            self.db.commit()
            self.db.refresh(document)
        except Exception as db_err:
            logger.error("Database commit failed for document_id=%s: %s", document_id, db_err, exc_info=True)
            self.db.rollback()
            raise

        logger.info("document_storage_complete id=%s type=%s duration_ms=%.1f validation=%s review=%s",
                    document.id, document.document_type, duration_ms, document.validation_status,
                    document.review_required)

        return document

    def get_document(self, document_id: str) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == document_id).first()

