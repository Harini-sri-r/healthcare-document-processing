import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Float, Boolean, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship

from .database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    patient_name = Column(String, nullable=True) # Usually shouldn't store names unnecessarily, but good for local dev
    date_of_birth = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    documents = relationship("Document", back_populates="patient", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.id"), nullable=True)
    document_type = Column(String, index=True)
    original_filename = Column(String)
    original_document_path = Column(String, nullable=True)
    document_date = Column(String, nullable=True)
    processing_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    classifier_confidence = Column(Float)
    extraction_status = Column(String)
    extraction_processing_status = Column(String, default="PENDING")
    processing_status = Column(String, default="COMPLETED")
    validation_status = Column(String)
    review_required = Column(Boolean, default=False)
    review_reasons = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    patient = relationship("Patient", back_populates="documents")
    extraction = relationship("DocumentExtraction", back_populates="document", uselist=False, cascade="all, delete-orphan")
    extraction_raw = relationship("DocumentExtractionRaw", back_populates="document", uselist=False, cascade="all, delete-orphan")
    verification_runs = relationship("DocumentVerificationRun", back_populates="document", cascade="all, delete-orphan", order_by="DocumentVerificationRun.created_at")
    logs = relationship("ProcessingLog", back_populates="document", cascade="all, delete-orphan")
    ground_truth = relationship("GroundTruthAnnotation", back_populates="document", uselist=False, cascade="all, delete-orphan")


class DocumentExtraction(Base):
    __tablename__ = "document_extractions"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), unique=True)
    extraction_json = Column(JSON, nullable=False)
    normalized_data = Column(JSON, nullable=True)
    
    document = relationship("Document", back_populates="extraction")


class DocumentExtractionRaw(Base):
    """Original VLM text retained separately from parsed extraction JSON."""
    __tablename__ = "document_extraction_raw"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), unique=True, nullable=False, index=True)
    raw_text = Column(Text, nullable=True)
    json_valid = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    document = relationship("Document", back_populates="extraction_raw")


class DocumentVerificationRun(Base):
    __tablename__ = "document_verification_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    verification_model = Column(String, nullable=False)
    overall_status = Column(String, nullable=False, index=True)
    availability = Column(String, nullable=False, default="AVAILABLE")
    report_json = Column(JSON, nullable=False)
    raw_verifier_output = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    reviewer_resolution = Column(JSON, nullable=True)
    reviewer_resolved_at = Column(DateTime, nullable=True)
    document = relationship("Document", back_populates="verification_runs")


class GroundTruthAnnotation(Base):
    __tablename__ = "ground_truth_annotations"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), unique=True, nullable=False, index=True)
    data_json = Column(JSON, nullable=False, default=dict)
    field_statuses = Column(JSON, nullable=False, default=dict)
    workflow_status = Column(String, nullable=False, default="Draft", index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    revisions = relationship("GroundTruthRevision", back_populates="annotation", cascade="all, delete-orphan", order_by="GroundTruthRevision.revision_number")
    document = relationship("Document", back_populates="ground_truth")


class GroundTruthRevision(Base):
    __tablename__ = "ground_truth_revisions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    annotation_id = Column(String, ForeignKey("ground_truth_annotations.id"), nullable=False, index=True)
    revision_number = Column(Integer, nullable=False)
    data_json = Column(JSON, nullable=False)
    field_statuses = Column(JSON, nullable=False)
    workflow_status = Column(String, nullable=False)
    action = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    annotation = relationship("GroundTruthAnnotation", back_populates="revisions")


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_used = Column(String)
    processing_duration_ms = Column(Float)
    validation_status = Column(String)
    retry_count = Column(Integer, default=0)
    review_required = Column(Boolean, default=False)
    
    document = relationship("Document", back_populates="logs")
