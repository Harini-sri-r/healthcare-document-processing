import unittest
from unittest.mock import MagicMock, patch
import sqlite3
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.database import Base
from src.db.models import Document, DocumentExtraction, ProcessingLog
from src.extraction.json_validator import JSONValidator
from src.pipeline.pipeline import MedicalDocumentPipeline
from src.validation.medical_document_gate import MedicalDocumentGate
from src.services.document_service import DocumentService, _clean_string_identifier, _sanitize_json_serializable


class TestRobustnessAndFailures(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db = Session()
        self.validator = JSONValidator()
        self.test_img = Image.new("RGB", (32, 32))

    def tearDown(self):
        self.db.close()

    # 1. Valid dynamic JSON test
    def test_valid_dynamic_json(self):
        dynamic_json = {
            "document_type": "Medical_Form",
            "sections": [{"title": "General Info", "status": "Active"}]
        }
        res = self.validator.validate({"success": True, "data": dynamic_json}, "Medical_Form")
        self.assertTrue(res["valid"])
        self.assertFalse(res["requires_review"])

    # 2. Malformed JSON after retries
    def test_malformed_json_after_retries(self):
        res = self.validator.validate({"success": False, "error": "JSON parse error: expecting ':'"})
        self.assertFalse(res["valid"])
        self.assertTrue(res["requires_review"])

    # 3. Validator returns valid=False for malformed JSON
    def test_validator_returns_false_for_malformed(self):
        res_null = self.validator.validate({"success": True, "data": None})
        self.assertFalse(res_null["valid"])
        res_list = self.validator.validate({"success": True, "data": ["not", "object"]})
        self.assertFalse(res_list["valid"])

    # 4. Pipeline receives the same validation state as validator
    def test_pipeline_receives_matching_validation_state(self):
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = {"document_type": "Lab_Report", "confidence": 0.99}

        mock_extractor = MagicMock()
        mock_extractor.model_name = "qwen2.5vl:3b"
        mock_extractor.extract.return_value = {
            "success": True,
            "attempts": 2,
            "data": {"document_type": "Lab_Report", "results": [{"test": "CBC"}]},
            "error": None
        }

        gate = MedicalDocumentGate(threshold=0.0)
        pipeline = MedicalDocumentPipeline(classifier=mock_classifier, extractor=mock_extractor, medical_document_gate=gate)

        with patch("src.pipeline.pipeline.load_document", return_value=self.test_img), \
             patch("src.pipeline.pipeline.load_document_pages", return_value=[self.test_img]):
            result = pipeline.process(self.test_img)

        # Validator says valid=True
        validator_res = self.validator.validate(result["extraction"], "Lab_Report")
        self.assertEqual(result["validation"]["valid"], validator_res["valid"])
        self.assertTrue(result["validation"]["valid"])
        self.assertTrue(result["validation"]["requires_review"])

    # 5. Malformed Qwen output produces REVIEW_REQUIRED rather than an unhandled exception
    def test_malformed_qwen_output_produces_review_required(self):
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = {"document_type": "Prescription", "confidence": 0.95}

        mock_extractor = MagicMock()
        mock_extractor.model_name = "qwen2.5vl:3b"
        mock_extractor.extract.return_value = {
            "success": False,
            "attempts": 3,
            "data": None,
            "error": "Failed to parse JSON"
        }

        gate = MedicalDocumentGate(threshold=0.0)
        pipeline = MedicalDocumentPipeline(classifier=mock_classifier, extractor=mock_extractor, medical_document_gate=gate)

        with patch("src.pipeline.pipeline.load_document", return_value=self.test_img), \
             patch("src.pipeline.pipeline.load_document_pages", return_value=[self.test_img]):
            result = pipeline.process(self.test_img)

        self.assertFalse(result["validation"]["valid"])
        self.assertTrue(result["validation"]["requires_review"])
        self.assertEqual(result["review"]["status"], "HUMAN_REVIEW")

    # 6. Database persistence of dynamic JSON
    def test_database_persistence_of_dynamic_json(self):
        mock_pipeline = MagicMock()
        mock_pipeline.extractor.model_name = "qwen2.5vl:3b"
        mock_pipeline.process.return_value = {
            "document_type": "Discharge_Summary",
            "confidence": 0.98,
            "classification": {"document_type": "Discharge_Summary", "confidence": 0.98},
            "extracted_json": {
                "document_type": "Discharge_Summary",
                "patient": {"id": "P-99", "name": "Alice"},
                "arbitrary_section": {"notes": "Patient discharged"}
            },
            "extraction": {"success": True, "attempts": 1, "data": {}},
            "validation": {"valid": True, "requires_review": False},
            "review": {"status": "AUTO_ACCEPT", "reasons": []}
        }

        doc_service = DocumentService(self.db, mock_pipeline)
        with patch("shutil.copy2"):
            doc = doc_service.process_and_store(self.test_img, "fake_file.png")

        stored_doc = self.db.query(Document).filter(Document.id == doc.id).first()
        self.assertIsNotNone(stored_doc)
        self.assertEqual(stored_doc.validation_status, "VALID")
        self.assertFalse(stored_doc.review_required)
        self.assertEqual(stored_doc.extraction.extraction_json["patient"]["id"], "P-99")

    # 7. Database failure is logged with the actual exception
    def test_database_failure_logging(self):
        mock_pipeline = MagicMock()
        mock_pipeline.extractor.model_name = "qwen2.5vl:3b"
        mock_pipeline.process.return_value = {
            "document_type": "Lab_Report",
            "confidence": 0.99,
            "classification": {"document_type": "Lab_Report", "confidence": 0.99},
            "extracted_json": {"document_type": "Lab_Report"},
            "extraction": {"success": True, "attempts": 1, "data": {}},
            "validation": {"valid": True, "requires_review": False},
            "review": {"status": "AUTO_ACCEPT", "reasons": []}
        }

        mock_db = MagicMock()
        mock_db.commit.side_effect = sqlite3.OperationalError("database is locked")

        doc_service = DocumentService(mock_db, mock_pipeline)
        with patch("shutil.copy2"), patch("src.services.document_service.logger") as mock_logger:
            with self.assertRaises(sqlite3.OperationalError):
                doc_service.process_and_store(self.test_img, "fake_file.png")
            mock_logger.error.assert_called()

    # 8. API response consistency helper test
    def test_string_identifier_and_json_sanitizer(self):
        self.assertEqual(_clean_string_identifier({"id": ["P123"]}), "P123")
        self.assertEqual(_clean_string_identifier(456), "456")
        self.assertIsNone(_clean_string_identifier(None))
        sanitized = _sanitize_json_serializable({1: (1, 2, 3), "key": {1, 2}})
        self.assertEqual(sanitized, {"1": [1, 2, 3], "key": [1, 2]})

    # 9 & 10. Dynamic JSON structures remain accepted without fixed medical schemas
    def test_unseen_healthcare_document_structures_accepted(self):
        unseen_doc = {
            "document_type": "Insurance_Claim_Form",
            "claim_id": "CLM-10029",
            "line_items": [{"code": "A1", "cost": 150.0}]
        }
        res = self.validator.validate({"success": True, "data": unseen_doc}, "Insurance_Claim_Form")
        self.assertTrue(res["valid"])


if __name__ == "__main__":
    unittest.main()
