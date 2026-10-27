from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form, Body
from sqlalchemy.orm import Session
from typing import Optional
import logging
import os
import shutil
from pathlib import Path

from src.db.database import get_db, engine, Base
from src.db.models import (Document, GroundTruthAnnotation, GroundTruthRevision,
                           DocumentExtractionRaw, DocumentVerificationRun)
from src.pipeline.pipeline import MedicalDocumentPipeline
from src.services.document_service import DocumentService
from src.services.patient_service import PatientService
from src.services.search_service import SearchService
from src.services.analytics_service import AnalyticsService
from src.config import settings
from src.preprocessing.image_preprocessing import load_document_pages
from src.extraction.second_model_verifier import SecondModelVerifier

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from datetime import datetime, timezone
import json
import csv
import io

# Models are imported above, so initialize every declared table before uploads.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Healthcare Document Intelligence API")
api_logger = logging.getLogger("healthcare.api")

# Keep the public API focused on the main document-processing workflow. Other
# routes remain available to the frontend and internal application features.
PUBLIC_API_ENDPOINTS = {
    ("POST", "/documents/process"),
    ("GET", "/documents/{document_id}"),
    ("GET", "/patients/{patient_id}/timeline"),
    ("GET", "/documents/search/"),
    ("GET", "/analytics/documents"),
}


def hide_internal_routes_from_schema():
    """Hide every route except the five documented public workflow endpoints."""
    for route in app.routes:
        if not hasattr(route, "methods") or not hasattr(route, "path"):
            continue
        route.include_in_schema = any(
            (method, route.path) in PUBLIC_API_ENDPOINTS
            for method in route.methods
        )
    app.openapi_schema = None

# Mount frontend interactive pipeline component
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

    @app.get("/ground-truth", include_in_schema=False)
    async def serve_ground_truth():
        return FileResponse(FRONTEND_DIR / "ground_truth.html")

    @app.get("/verification", include_in_schema=False)
    async def serve_verification():
        return FileResponse(FRONTEND_DIR / "verification.html")


GROUND_TRUTH_WORKFLOW = {"Draft", "In Review", "Verified", "Rejected"}
FIELD_STATUSES = {"Verified", "Unreadable", "Not Present", "Needs Review"}


def _json_value(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Annotation must be valid finite JSON.") from None


def _annotation_payload(a):
    return {"id": a.id, "document_id": a.document_id, "data": a.data_json,
            "field_statuses": a.field_statuses, "status": a.workflow_status,
            "created_at": a.created_at.isoformat(), "updated_at": a.updated_at.isoformat(),
            "revisions": [{"revision": r.revision_number, "action": r.action,
                           "status": r.workflow_status, "created_at": r.created_at.isoformat()}
                          for r in a.revisions]}


def _add_revision(db, annotation, action):
    db.flush()
    number = db.query(GroundTruthRevision).filter_by(annotation_id=annotation.id).count() + 1
    db.add(GroundTruthRevision(annotation_id=annotation.id, revision_number=number,
        data_json=_json_value(annotation.data_json), field_statuses=_json_value(annotation.field_statuses),
        workflow_status=annotation.workflow_status, action=action))


@app.get("/ground-truth/documents")
def ground_truth_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [{"id": d.id, "filename": d.original_filename, "document_type": d.document_type,
             "created_at": d.created_at.isoformat() if d.created_at else None,
             "has_original": bool(d.original_document_path and Path(d.original_document_path).is_file()),
             "annotation_status": d.ground_truth.workflow_status if getattr(d, "ground_truth", None) else None}
            for d in docs]


@app.get("/ground-truth/documents/{document_id}/source")
def ground_truth_source(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")
    path = Path(doc.original_document_path) if doc.original_document_path else None
    if not path or not path.is_file(): raise HTTPException(404, "Original document is unavailable")
    media = "application/pdf" if path.suffix.lower() == ".pdf" else ("image/png" if path.suffix.lower() == ".png" else "image/jpeg")
    return FileResponse(path, media_type=media, content_disposition_type="inline")


@app.get("/ground-truth/documents/{document_id}")
def get_ground_truth(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")
    annotation = db.query(GroundTruthAnnotation).filter_by(document_id=document_id).first()
    return {"document": {"id": doc.id, "filename": doc.original_filename,
        "document_type": doc.document_type, "source_url": f"/ground-truth/documents/{doc.id}/source"},
        "suggestion": doc.extraction.extraction_json if doc.extraction else None,
        "annotation": _annotation_payload(annotation) if annotation else None}


@app.put("/ground-truth/documents/{document_id}")
def save_ground_truth(document_id: str, body: dict = Body(...), db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")
    data = _json_value(body.get("data"))
    statuses = _json_value(body.get("field_statuses", {}))
    status = body.get("status", "Draft")
    if not isinstance(data, (dict, list)): raise HTTPException(422, "Ground truth root must be an object or array.")
    if not isinstance(statuses, dict) or any(v not in FIELD_STATUSES for v in statuses.values()):
        raise HTTPException(422, "Field statuses must map JSON paths to an allowed field status.")
    if status not in GROUND_TRUTH_WORKFLOW or status == "Verified": raise HTTPException(422, "Use the explicit confirmation action to verify an annotation.")
    annotation = db.query(GroundTruthAnnotation).filter_by(document_id=document_id).first()
    creating = annotation is None
    if creating:
        annotation = GroundTruthAnnotation(document_id=document_id, data_json=data, field_statuses=statuses, workflow_status="Draft")
        db.add(annotation)
    else:
        annotation.data_json, annotation.field_statuses = data, statuses
        # Editing a verified record invalidates its verified status pending reconfirmation.
        annotation.workflow_status = "Draft" if annotation.workflow_status == "Verified" else status
        annotation.updated_at = datetime.now(timezone.utc)
    db.flush(); _add_revision(db, annotation, "created" if creating else "saved")
    db.commit(); db.refresh(annotation)
    return _annotation_payload(annotation)


@app.post("/ground-truth/documents/{document_id}/workflow")
def ground_truth_workflow(document_id: str, body: dict = Body(...), db: Session = Depends(get_db)):
    annotation = db.query(GroundTruthAnnotation).filter_by(document_id=document_id).first()
    if not annotation: raise HTTPException(404, "Save an annotation before changing workflow status")
    status = body.get("status")
    if status not in GROUND_TRUTH_WORKFLOW: raise HTTPException(422, "Invalid workflow status.")
    if status == "Verified" and body.get("confirmed") is not True:
        raise HTTPException(400, "Explicit confirmation is required to mark ground truth verified.")
    annotation.workflow_status = status; annotation.updated_at = datetime.now(timezone.utc)
    _add_revision(db, annotation, "explicitly verified" if status == "Verified" else f"status changed to {status}")
    db.commit(); db.refresh(annotation)
    return _annotation_payload(annotation)


@app.delete("/ground-truth/documents/{document_id}")
def delete_ground_truth(document_id: str, db: Session = Depends(get_db)):
    annotation = db.query(GroundTruthAnnotation).filter_by(document_id=document_id).first()
    if not annotation: raise HTTPException(404, "Annotation not found")
    db.delete(annotation); db.commit(); return {"deleted": True}


def _verified_annotations(db):
    return db.query(GroundTruthAnnotation).filter_by(workflow_status="Verified").all()


@app.get("/ground-truth/export.json")
def export_ground_truth_json(db: Session = Depends(get_db)):
    rows = _verified_annotations(db)
    docs = {d.id: d for d in db.query(Document).filter(Document.id.in_([r.document_id for r in rows])).all()} if rows else {}
    content = [{"document_id": r.document_id, "filename": docs[r.document_id].original_filename,
                "data": r.data_json, "field_statuses": r.field_statuses, "status": r.workflow_status,
                "updated_at": r.updated_at.isoformat()} for r in rows if r.document_id in docs]
    return StreamingResponse(io.BytesIO(json.dumps(content, ensure_ascii=False, indent=2).encode()), media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=verified_ground_truth.json"})


@app.get("/ground-truth/export.csv")
def export_ground_truth_csv(db: Session = Depends(get_db)):
    rows = _verified_annotations(db)
    docs = {d.id: d for d in db.query(Document).filter(Document.id.in_([r.document_id for r in rows])).all()} if rows else {}
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=["document_id", "filename", "json_path", "value", "field_status"]); writer.writeheader()
    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for k, v in value.items(): yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(value, list):
            for i, v in enumerate(value): yield from flatten(v, f"{prefix}[{i}]")
        else: yield prefix, value
    for r in rows:
        if r.document_id not in docs: continue
        for path, value in flatten(r.data_json):
            writer.writerow({"document_id": r.document_id, "filename": docs[r.document_id].original_filename,
                "json_path": path, "value": json.dumps(value, ensure_ascii=False), "field_status": r.field_statuses.get(path, "Needs Review")})
    return StreamingResponse(io.BytesIO(stream.getvalue().encode("utf-8-sig")), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=verified_ground_truth.csv"})


def _verification_payload(run: DocumentVerificationRun):
    return {"id": run.id, "document_id": run.document_id, "verification_model": run.verification_model,
            "overall_status": run.overall_status, "availability": run.availability,
            "report": run.report_json, "latency_ms": run.latency_ms,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "reviewer_resolution": run.reviewer_resolution,
            "reviewer_resolved_at": run.reviewer_resolved_at.isoformat() if run.reviewer_resolved_at else None}


@app.get("/documents/{document_id}/verification-runs")
def list_verification_runs(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")
    raw = db.query(DocumentExtractionRaw).filter_by(document_id=document_id).first()
    qwen_json = doc.extraction.extraction_json if doc.extraction else None
    if raw and raw.json_valid is False: qwen_json = None
    return {"document_id": doc.id, "filename": doc.original_filename,
            "document_type": doc.document_type,
            "qwen_json": qwen_json,
            "qwen_raw_available": bool(raw and raw.raw_text),
            "runs": [_verification_payload(r) for r in db.query(DocumentVerificationRun).filter_by(document_id=document_id).order_by(DocumentVerificationRun.created_at.desc()).all()]}


@app.get("/verification/documents")
def verification_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.extraction_processing_status != "SKIPPED").order_by(Document.created_at.desc()).all()
    return [{"id": d.id, "filename": d.original_filename, "document_type": d.document_type,
             "has_original": bool(d.original_document_path and Path(d.original_document_path).is_file()),
             "has_extraction": bool(d.extraction)} for d in docs]


@app.post("/documents/{document_id}/verify")
def verify_document(document_id: str, db: Session = Depends(get_db)):
    if not settings.SECOND_MODEL_VERIFICATION_ENABLED:
        raise HTTPException(409, "Second-model verification is disabled. Set MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_ENABLED=true and restart the API.")
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc: raise HTTPException(404, "Document not found")
    if not doc.original_document_path or not Path(doc.original_document_path).is_file():
        raise HTTPException(409, "Original document image is unavailable; verification was not run.")
    raw_record = db.query(DocumentExtractionRaw).filter_by(document_id=document_id).first()
    raw_output = raw_record.raw_text if raw_record else None
    extraction = doc.extraction.extraction_json if doc.extraction else None
    if raw_record and raw_record.json_valid is False: extraction = None
    verifier = SecondModelVerifier()
    try:
        pages = load_document_pages(Path(doc.original_document_path))
        result = verifier.verify(pages, raw_output, extraction)
    except Exception as exc:
        # Keep response non-sensitive; do not log image paths or document text.
        result = {"verification_model": settings.SECOND_MODEL_VERIFICATION_MODEL,
                  "overall_status": "REVIEW_REQUIRED", "field_reviews": [],
                  "review_notes": ["Source image could not be loaded for independent verification."],
                  "availability": "UNAVAILABLE", "latency_ms": 0.0}
    raw_verifier = result.pop("raw_verifier_output", None)
    report = {k: v for k, v in result.items() if k not in {"availability", "latency_ms"}}
    run = DocumentVerificationRun(document_id=document_id, verification_model=result.get("verification_model", settings.SECOND_MODEL_VERIFICATION_MODEL),
        overall_status=result.get("overall_status", "REVIEW_REQUIRED"), availability=result.get("availability", "UNAVAILABLE"),
        report_json=report, raw_verifier_output=raw_verifier, latency_ms=result.get("latency_ms"))
    db.add(run)
    if run.overall_status != "CONSISTENT" or run.availability != "AVAILABLE":
        doc.review_required = True
        reasons = list(doc.review_reasons or [])
        if "Second-model verification requires human review" not in reasons:
            reasons.append("Second-model verification requires human review")
        doc.review_reasons = reasons
    db.commit(); db.refresh(run)
    return _verification_payload(run)


@app.put("/verification-runs/{run_id}/resolution")
def resolve_verification_run(run_id: str, body: dict = Body(...), db: Session = Depends(get_db)):
    run = db.query(DocumentVerificationRun).filter_by(id=run_id).first()
    if not run: raise HTTPException(404, "Verification run not found")
    resolutions = body.get("field_resolutions", {})
    note = body.get("reviewer_note", "")
    if not isinstance(resolutions, dict) or not isinstance(note, str):
        raise HTTPException(422, "field_resolutions must be an object and reviewer_note must be text.")
    valid_actions = {"confirmed_discrepancy", "accepted_extraction", "corrected", "unresolved"}
    expected_paths = {item.get("field_path") for item in run.report_json.get("field_reviews", []) if isinstance(item, dict)}
    if any(path not in expected_paths for path in resolutions): raise HTTPException(422, "Resolution contains an unknown field path.")
    if any(not isinstance(v, dict) or v.get("action") not in valid_actions for v in resolutions.values()):
        raise HTTPException(422, "Each field resolution needs an allowed action.")
    for resolution in resolutions.values():
        if resolution["action"] == "corrected":
            if "value" not in resolution: raise HTTPException(422, "Corrected resolutions must include a value.")
            resolution["value"] = _json_value(resolution["value"])
    run.reviewer_resolution = {"field_resolutions": resolutions, "reviewer_note": note[:4000]}
    run.reviewer_resolved_at = datetime.now(timezone.utc)
    db.commit(); db.refresh(run)
    return _verification_payload(run)

    @app.get("/pipeline", include_in_schema=False)
    @app.get("/flowchart", include_in_schema=False)
    async def serve_pipeline_flowchart():
        return FileResponse(FRONTEND_DIR / "index.html")

# Load v3 explicitly for the production upload route. Keep the API and Swagger
# available when model loading fails; the upload route returns a safe 503.
pipeline: Optional[MedicalDocumentPipeline]
pipeline_load_error: Optional[Exception] = None
try:
    pipeline = MedicalDocumentPipeline(
        resnet_model_path=settings.RESNET_V3_MODEL_PATH,
        classification_threshold=settings.CLASSIFICATION_CONFIDENCE_THRESHOLD,
        extractor_backend="ollama",
    )
    api_logger.info("Loaded ResNet %s checkpoint: %s", pipeline.classifier.model_version, pipeline.classifier.checkpoint_path)
    api_logger.info(
        "API ResNet model_version=%s class_to_idx=%s",
        pipeline.classifier.model_version,
        pipeline.classifier.class_to_idx,
    )
except Exception as exc:
    pipeline = None
    pipeline_load_error = exc
    api_logger.exception("Production pipeline model initialization failed")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}

@app.post("/documents/process")
async def process_document(
    file: UploadFile = File(...),
    run_extraction: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Process a medical document image.
    Saves it temporarily, runs the pipeline, stores results in DB, and returns the result.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: PNG, JPG, JPEG, WEBP, PDF."
        )

    expected_content_type = "application/pdf" if file_ext == ".pdf" else "image/"
    if file.content_type and not (file.content_type == "application/pdf" if file_ext == ".pdf" else file.content_type.startswith(expected_content_type)):
        raise HTTPException(status_code=400, detail="Upload content type does not match a supported document format.")

    import tempfile
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(prefix="healthcare_upload_", suffix=file_ext, delete=False) as buffer:
            temp_path = Path(buffer.name)
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > 50 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Document exceeds the 50 MB upload limit.")
                buffer.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="The document classification model is unavailable. Check the v3 checkpoint and server logs.",
            )

        # Validate actual image/PDF contents before database storage. This uses
        # the shared PDF renderer, so malformed documents receive a clear 400.
        try:
            load_document_pages(temp_path)
        except Exception as exc:
            api_logger.info("upload_content_invalid extension=%s error_type=%s", file_ext, type(exc).__name__)
            raise HTTPException(
                status_code=400,
                detail="Uploaded image or PDF is corrupted or cannot be rendered.",
            ) from None

        api_logger.info("document_upload_received extension=%s", file_ext)
        doc_service = DocumentService(db, pipeline)
        document = doc_service.process_and_store(
            image=temp_path,
            original_filename=file.filename,
            run_extraction=run_extraction
        )
        if isinstance(document, dict) and document.get("status") == "REJECTED":
            return {
                "accepted": False,
                "qwen_invoked": False,
                "status": "REJECTED",
                "classification": document.get("classification_result") or {
                    "is_medical": False,
                    "document_type": document.get("predicted_class"),
                    "confidence": document.get("confidence", 0.0),
                },
                "evaluation": document.get("evaluation") or {
                    "qwen_invoked": False,
                    "status": "REJECTED",
                    "validation_status": "SKIPPED",
                    "extracted_json": None,
                },
                "document_type": document.get("predicted_class"),
                "extracted_json": None,
                "message": "The uploaded document is not a supported medical document.",
                "supported_document_types": document.get("supported_document_types", ["Prescription", "Lab_Report", "Discharge_Summary"]),
                "confidence": document.get("confidence", 0.0),
                "classification_confidence": document.get("confidence", 0.0),
                "predicted_class": document.get("predicted_class"),
                "rejection_reason": (document.get("medical_document_gate") or {}).get("rejection_reason"),
            }
        pipeline_result = getattr(document, "_pipeline_result", {})
        pipeline_errors = pipeline_result.get("errors", []) if isinstance(pipeline_result, dict) else []
        error_codes = {item.get("code") for item in pipeline_errors if isinstance(item, dict)}
        if error_codes & {"ollama_connection_error", "ollama_model_missing"}:
            raise HTTPException(
                status_code=503,
                detail="Qwen/Ollama is unavailable. Start Ollama and ensure qwen2.5vl:3b is installed.",
            )
        if "extraction_error" in error_codes:
            extraction = pipeline_result.get("extraction") or {}
            raise HTTPException(status_code=502, detail={
                "code": "qwen_extraction_failed",
                "message": "Qwen could not return valid structured JSON. The document was retained for human review.",
                "reason": extraction.get("error") or "Extraction failed validation.",
                "attempts": extraction.get("attempts", 0),
                "review_required": True,
            })
        duration_ms = document.logs[0].processing_duration_ms if document.logs else None
        api_logger.info("document_processed id=%s type=%s confidence=%.4f validation=%s review=%s duration_ms=%.1f",
                        document.id, document.document_type, document.classifier_confidence,
                        document.validation_status, document.review_required, duration_ms or 0.0)
        extracted_data = (pipeline_result.get("extracted_json") if isinstance(pipeline_result, dict)
                          and "extracted_json" in pipeline_result
                          else document.extraction.extraction_json if document.extraction else None)
        qwen_invoked = pipeline_result.get(
            "qwen_invoked",
            document.extraction_processing_status != "SKIPPED",
        ) if isinstance(pipeline_result, dict) else document.extraction_processing_status != "SKIPPED"
        classification_result = (pipeline_result.get("classification_result")
                                  if isinstance(pipeline_result, dict) else None) or {
            "is_medical": document.document_type in {"Discharge_Summary", "Lab_Report", "Prescription"},
            "document_type": document.document_type,
            "confidence": document.classifier_confidence,
        }
        evaluation = (pipeline_result.get("evaluation")
                      if isinstance(pipeline_result, dict) else None) or {
            "qwen_invoked": qwen_invoked,
            "status": "ACCEPTED",
            "validation_status": document.validation_status,
            "extracted_json": extracted_data,
        }
        return {
            "accepted": True,
            "qwen_invoked": qwen_invoked,
            "classification": classification_result,
            "evaluation": evaluation,
            "document_id": document.id,
            "patient_id": document.patient_id,
            "document_type": document.document_type,
            "document_date": document.document_date,
            "classifier_confidence": document.classifier_confidence,
            "classification_confidence": document.classifier_confidence,
            "status": document.extraction_status,
            "extraction_status": document.extraction_status,
            "processing_status": document.processing_status,
            "extraction_processing_status": document.extraction_processing_status,
            "validation": {"status": document.validation_status},
            "validation_status": document.validation_status,
            "review_required": document.review_required,
            "review_reasons": document.review_reasons,
            "processing_duration_ms": duration_ms,
            "extracted_json": extracted_data,
            "extracted_data": extracted_data,
            "timings": pipeline_result.get("timings", {}),
            "errors": pipeline_errors,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        api_logger.error("document_processing_failed error_type=%s", type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail="Document processing failed due to an internal server error. Check server logs.",
        )

    finally:
        if temp_path is not None and temp_path.exists():
            try:
                os.remove(temp_path)
            except PermissionError:
                # PyMuPDF may retain a failed-open handle briefly on Windows.
                import gc
                gc.collect()
                try:
                    os.remove(temp_path)
                except OSError:
                    api_logger.warning("Could not remove temporary upload after processing")

@app.get("/documents/{document_id}")
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc_service = DocumentService(db, pipeline)
    doc = doc_service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    duration_ms = doc.logs[0].processing_duration_ms if doc.logs else None
    extracted_data = doc.extraction.extraction_json if doc.extraction else None
    return {
        "id": doc.id,
        "document_id": doc.id,
        "patient_id": doc.patient_id,
        "document_type": doc.document_type,
        "classifier_confidence": doc.classifier_confidence,
        "status": doc.extraction_status,
        "extraction_status": doc.extraction_status,
        "processing_status": doc.processing_status,
        "extraction_processing_status": doc.extraction_processing_status,
        "original_filename": doc.original_filename,
        "original_document_available": bool(doc.original_document_path and Path(doc.original_document_path).is_file()),
        "document_date": doc.document_date,
        "extraction": extracted_data,
        "extracted_json": extracted_data,
        "review_required": doc.review_required,
        "review_reasons": doc.review_reasons,
        "processing_duration_ms": duration_ms
    }

@app.get("/patients/{patient_id}/timeline")
def get_patient_timeline(patient_id: str, db: Session = Depends(get_db)):
    patient_service = PatientService(db)
    timeline = patient_service.get_patient_timeline(patient_id)
    return {"patient_id": patient_id, "timeline": timeline}

@app.get("/documents/search/")
def search_documents(
    patient_id: Optional[str] = None,
    document_type: Optional[str] = None,
    document_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    processing_status: Optional[str] = None,
    review_required: Optional[bool] = None,
    keyword: Optional[str] = None,
    db: Session = Depends(get_db)
):
    search_service = SearchService(db)
    docs = search_service.search_documents(
        patient_id=patient_id, 
        document_type=document_type, 
        date_from=date_from,
        date_to=date_to,
        keyword=keyword
    )
    if document_id:
        docs = [d for d in docs if d.id == document_id]
    if processing_status:
        docs = [d for d in docs if d.processing_status == processing_status]
    if review_required is not None:
        docs = [d for d in docs if d.review_required == review_required]
    return [{"document_id": d.id, "patient_id": d.patient_id, "document_type": d.document_type,
             "document_date": d.document_date, "processing_status": d.processing_status,
             "extraction_status": d.extraction_status, "review_required": d.review_required}
            for d in docs]

@app.get("/analytics/documents")
def get_analytics(db: Session = Depends(get_db)):
    analytics_service = AnalyticsService(db)
    return analytics_service.get_document_analytics()

@app.get("/documents/review/pending")
def get_documents_for_review(db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.review_required == True).all()
    return [{"document_id": d.id, "document_type": d.document_type, "patient_id": d.patient_id,
             "document_date": d.document_date, "validation_status": d.validation_status,
             "reasons": d.review_reasons} for d in docs]


# Run after all routes are registered so OpenAPI is filtered from the first
# request, including when /openapi.json is fetched before application startup.
hide_internal_routes_from_schema()
