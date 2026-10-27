from typing import List, Optional, Any, Dict
from sqlalchemy.orm import Session
from sqlalchemy import or_, cast, String

from src.db.models import Document, DocumentExtraction

class SearchService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def search_documents(
        self, 
        patient_id: Optional[str] = None,
        document_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> List[Document]:
        """
        Search documents by various metadata filters and a basic keyword search
        on the JSON structure (if using SQLite, string cast is used. Postgres has JSONB querying).
        """
        query = self.db.query(Document)
        
        if patient_id:
            query = query.filter(Document.patient_id == patient_id)
            
        if document_type:
            query = query.filter(Document.document_type == document_type)
            
        if date_from:
            query = query.filter(Document.document_date >= date_from)
            
        if date_to:
            query = query.filter(Document.document_date <= date_to)
            
        if keyword:
            # Search the filename and structured extraction without returning either in results.
            # For SQLite compatibility, we cast JSON to String. For Postgres, we could use ->> or @>.
            query = query.join(DocumentExtraction).filter(
                or_(
                    Document.original_filename.ilike(f"%{keyword}%"),
                    cast(DocumentExtraction.extraction_json, String).ilike(f"%{keyword}%")
                )
            )
        query = query.order_by(Document.created_at.desc())
            
        return query.order_by(Document.created_at.desc()).all()
