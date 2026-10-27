from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.db.models import Document, Patient, DocumentExtraction

class PatientService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        return self.db.query(Patient).filter(Patient.id == patient_id).first()

    def get_patient_timeline(self, patient_id: str) -> List[dict]:
        """
        Retrieves a patient's document timeline ordered by document date (descending).
        """
        documents = (
            self.db.query(Document)
            .filter(Document.patient_id == patient_id)
            .order_by(desc(Document.document_date), desc(Document.created_at))
            .all()
        )
        
        timeline = []
        for doc in documents:
            timeline.append({
                "document_id": doc.id,
                "patient_id": doc.patient_id,
                "document_type": doc.document_type,
                "document_date": doc.document_date,
                "classifier_confidence": doc.classifier_confidence,
                "status": doc.extraction_status,
                "review_required": doc.review_required,
                "review_reasons": doc.review_reasons,
                "extraction": doc.extraction.extraction_json if doc.extraction else None
            })
            
        return timeline
