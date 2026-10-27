from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any

from src.db.models import Document, DocumentExtraction

class AnalyticsService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def get_document_analytics(self) -> Dict[str, Any]:
        """
        Calculates total documents, documents by type, success rates, and review required counts.
        """
        total_documents = self.db.query(Document).count()
        
        # Documents by type
        doc_type_counts = self.db.query(
            Document.document_type, func.count(Document.id)
        ).group_by(Document.document_type).all()
        documents_by_type = {doc_type: count for doc_type, count in doc_type_counts}
        
        # Extraction success
        successful_extractions = self.db.query(Document).filter(
            Document.extraction_status == "SUCCESS"
        ).count()
        failed_extractions = total_documents - successful_extractions
        
        # Review required
        needs_review = self.db.query(Document).filter(
            Document.review_required == True
        ).count()
        
        success_rate = (successful_extractions / total_documents * 100) if total_documents > 0 else 0
        needs_review_rate = (needs_review / total_documents * 100) if total_documents > 0 else 0
        
        return {
            "total_documents": total_documents,
            "documents_pending_review": needs_review,
            "documents_by_type": documents_by_type,
            "extraction_stats": {
                "successful": successful_extractions,
                "failed": failed_extractions,
                "success_rate_percentage": round(success_rate, 2),
                "needs_review": needs_review,
                "needs_review_percentage": round(needs_review_rate, 2)
            }
        }
