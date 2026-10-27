from io import BytesIO
from types import SimpleNamespace

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.api as api
from src.db.database import Base, engine
from src.extraction.ollama_extractor import OllamaExtractor
from src.pipeline.pipeline import MedicalDocumentPipeline


@pytest.fixture(scope="module")
def api_client():
    Base.metadata.create_all(bind=engine)
    yield TestClient(api.app)


def _real_receipt_bytes():
    path = api.settings.RAW_DATA_PATH / "non_medical" / "cord_receipt_0001.jpg"
    return path.read_bytes()


def _image_bytes(format_name="JPEG"):
    output = BytesIO()
    with Image.open(BytesIO(_real_receipt_bytes())) as image:
        image.convert("RGB").save(output, format=format_name)
    return output.getvalue()


def _pdf_bytes(page_count=1):
    image_bytes = _real_receipt_bytes()
    pdf = fitz.open()
    for _ in range(page_count):
        page = pdf.new_page(width=432, height=648)
        page.insert_image(page.rect, stream=image_bytes)
    payload = pdf.tobytes()
    pdf.close()
    return payload


class FixedClassifier:
    def __init__(self, category, events=None):
        self.category = category
        self.calls = 0
        self.events = events if events is not None else []

    def predict(self, _tensor):
        self.calls += 1
        self.events.append("resnet")
        probabilities = {
            name: 0.99 if name == self.category else 0.01 / 3
            for name in ("Discharge_Summary", "Lab_Report", "Prescription", "Non_Medical")
        }
        return {
            "predicted_class": self.category,
            "document_type": self.category,
            "confidence": probabilities[self.category],
            "probabilities": probabilities,
            "model_version": "v2",
        }


class MockExtractor:
    model_name = "qwen2.5vl:3b"

    def __init__(self, events=None):
        self.calls = 0
        self.events = events if events is not None else []

    def extract(self, _image):
        self.calls += 1
        self.events.append("qwen")
        return {
            "success": True,
            "attempts": 1,
            "data": {"document_type": self.document_type, "observed_fields": {"example": "from document"}},
            "history": [],
            "error": None,
        }


class FakeDocumentService:
    """Exercise the route/pipeline while avoiding persistent test DB records."""

    def __init__(self, _db, pipeline):
        self.pipeline = pipeline

    def process_and_store(self, image, original_filename, run_extraction=True):
        result = self.pipeline.process(image, run_extraction=run_extraction)
        if result.get("status") == "REJECTED":
            return result
        data = result.get("extracted_data")
        extraction = SimpleNamespace(extraction_json=data) if data is not None else None
        document = SimpleNamespace(
            _pipeline_result=result,
            id="api-test-document",
            patient_id=None,
            document_type=result["document_type"],
            document_date=None,
            classifier_confidence=result["classification_confidence"],
            extraction_status="SUCCESS" if result.get("qwen_invoked") else "SKIPPED",
            extraction_processing_status="SUCCESS" if result.get("qwen_invoked") else "SKIPPED",
            processing_status="COMPLETED",
            validation_status="VALID",
            review_required=False,
            review_reasons=[],
            extraction=extraction,
            logs=[],
        )
        return document


def configure_pipeline(monkeypatch, category):
    events = []
    classifier = FixedClassifier(category, events)
    extractor = MockExtractor(events)
    extractor.document_type = category
    pipeline = MedicalDocumentPipeline(
        classifier=classifier,
        extractor=extractor,
        classification_threshold=0.0,
    )
    monkeypatch.setattr(api, "pipeline", pipeline)
    monkeypatch.setattr(api, "DocumentService", FakeDocumentService)
    return classifier, extractor


def upload(client, filename, content, content_type):
    return client.post(
        "/documents/process",
        files={"file": (filename, content, content_type)},
    )


def test_upload_endpoint_uses_v3_checkpoint(api_client):
    assert api.pipeline is not None
    assert api.pipeline.classifier.model_version == "v3"
    assert api.pipeline.classifier.checkpoint_path == api.settings.RESNET_V3_MODEL_PATH.resolve()
    assert api.pipeline.classifier.class_to_idx == {
        "Discharge_Summary": 0,
        "Lab_Report": 1,
        "Prescription": 2,
        "Non_Medical": 3,
    }
    assert api.settings.RESNET_V3_MODEL_PATH.is_file()
    assert "/documents/process" in api.app.openapi()["paths"]


def test_nonmedical_image_is_rejected_without_qwen(api_client, monkeypatch):
    classifier, extractor = configure_pipeline(monkeypatch, "Non_Medical")
    response = upload(api_client, "resume.jpg", _image_bytes(), "image/jpeg")
    assert response.status_code == 200
    result = response.json()
    assert result["accepted"] is False
    assert result["document_type"] == "Non_Medical"
    assert result["qwen_invoked"] is False
    assert result["classification"] == {
        "is_medical": False, "document_type": "Non_Medical", "confidence": pytest.approx(0.99)
    }
    assert result["evaluation"]["status"] == "REJECTED"
    assert result["evaluation"]["extracted_json"] is None
    assert result["classification_confidence"] == pytest.approx(0.99)
    assert result["message"] == "The uploaded document is not a supported medical document."
    assert classifier.calls == 1
    assert extractor.calls == 0
    assert classifier.events == ["resnet"]
    assert extractor.events == ["resnet"]


@pytest.mark.parametrize("category", ["Lab_Report", "Prescription", "Discharge_Summary"])
def test_medical_classes_continue_to_mock_qwen(api_client, monkeypatch, category):
    _classifier, extractor = configure_pipeline(monkeypatch, category)
    response = upload(api_client, "upload.png", _image_bytes("PNG"), "image/png")
    assert response.status_code == 200
    result = response.json()
    assert result["accepted"] is True
    assert result["document_type"] == category
    assert result["qwen_invoked"] is True
    assert result["classification"] == {
        "is_medical": True, "document_type": category, "confidence": pytest.approx(0.99)
    }
    assert result["evaluation"]["qwen_invoked"] is True
    assert result["evaluation"]["extracted_json"] == result["extracted_data"]
    assert result["extracted_data"]["document_type"] == category
    assert "observed_fields" in result["extracted_data"]
    assert extractor.calls == 1
    assert extractor.events == ["resnet", "qwen"]


def test_pdf_renders_and_classifies_each_page_before_extraction(api_client, monkeypatch):
    classifier, extractor = configure_pipeline(monkeypatch, "Lab_Report")
    response = upload(api_client, "unrelated-name.pdf", _pdf_bytes(2), "application/pdf")
    assert response.status_code == 200
    result = response.json()
    assert result["document_type"] == "Lab_Report"
    assert result["qwen_invoked"] is True
    assert classifier.calls == 2
    assert extractor.calls == 1


def test_unsupported_empty_and_corrupt_upload_errors(api_client):
    unsupported = upload(api_client, "notes.txt", b"text", "text/plain")
    empty = upload(api_client, "empty.jpg", b"", "image/jpeg")
    corrupt_image = upload(api_client, "broken.png", b"not an image", "image/png")
    corrupt_pdf = upload(api_client, "broken.pdf", b"not a pdf", "application/pdf")
    assert unsupported.status_code == 400
    assert "Unsupported file format" in unsupported.json()["detail"]
    assert empty.status_code == 400
    assert "empty" in empty.json()["detail"].lower()
    assert corrupt_image.status_code == 400
    assert corrupt_pdf.status_code == 400
    assert "corrupted or cannot be rendered" in corrupt_pdf.json()["detail"]


def test_model_and_ollama_errors_are_safe_http_errors(api_client, monkeypatch):
    monkeypatch.setattr(api, "pipeline", None)
    response = upload(api_client, "valid.jpg", _image_bytes(), "image/jpeg")
    assert response.status_code == 503
    assert "classification model is unavailable" in response.json()["detail"]

    class UnavailableOllama(OllamaExtractor):
        def __init__(self):
            pass

        def check_connection(self):
            return {"reachable": False, "model_available": False,
                    "error": "Cannot connect to Ollama at an internal test URL"}

    classifier = FixedClassifier("Lab_Report")
    unavailable = UnavailableOllama()
    pipeline = MedicalDocumentPipeline(
        classifier=classifier,
        extractor=unavailable,
        classification_threshold=0.0,
    )
    monkeypatch.setattr(api, "pipeline", pipeline)
    monkeypatch.setattr(api, "DocumentService", FakeDocumentService)
    response = upload(api_client, "valid.jpg", _image_bytes(), "image/jpeg")
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Qwen/Ollama is unavailable. Start Ollama and ensure qwen2.5vl:3b is installed."
    )

