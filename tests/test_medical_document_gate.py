from pathlib import Path

from PIL import Image

from src.classification.classification import DocumentClassifier
from src.config import settings
from src.pipeline.pipeline import MedicalDocumentPipeline
from src.validation.medical_document_gate import MedicalDocumentGate


def test_gate_accepts_supported_classes_at_configured_threshold():
    gate = MedicalDocumentGate(threshold=0.8)
    for category in ("Lab_Report", "Prescription", "Discharge_Summary"):
        result = gate.evaluate({"document_type": category, "confidence": 0.81})
        assert result.is_supported_medical_document
        assert result.predicted_class == category
        assert result.confidence == 0.81


def test_gate_uses_class_membership_and_ignores_confidence():
    gate = MedicalDocumentGate(threshold=0.8)
    low = gate.evaluate({"document_type": "Lab_Report", "confidence": 0.79})
    unknown = gate.evaluate({"document_type": "Invoice", "confidence": 0.99})
    assert low.is_supported_medical_document
    assert not unknown.is_supported_medical_document


def test_gate_rejects_non_medical_prediction_regardless_of_confidence():
    result = MedicalDocumentGate(threshold=0.8).evaluate(
        {"document_type": "Non_Medical", "confidence": 0.99}
    )
    assert not result.is_supported_medical_document
    assert result.rejection_reason == "ResNet classified the document as Non_Medical."


def test_pipeline_rejects_before_qwen_but_preserves_resnet_prediction():
    image = Image.new("RGB", (300, 400), "white")

    class FixedClassifier:
        def predict(self, _image):
            return {"document_type": "Non_Medical", "predicted_class": "Non_Medical", "confidence": 0.42, "probabilities": {}}

    class NeverCalledExtractor:
        def extract(self, _image):
            raise AssertionError("Qwen must not run for rejected input")

    pipeline = MedicalDocumentPipeline(
        classifier=FixedClassifier(), extractor=NeverCalledExtractor(),
        medical_document_gate=MedicalDocumentGate(threshold=0.8),
    )
    result = pipeline.process(image)
    assert result["status"] == "REJECTED"
    assert result["predicted_class"] == "Non_Medical"
    assert result["confidence"] == 0.42
    assert result["classification"]["document_type"] == "Non_Medical"
    assert result["extraction"]["status"] == "SKIPPED"
    assert result["accepted"] is False
    assert result["qwen_invoked"] is False


def test_pipeline_accepts_medical_prediction_and_invokes_qwen():
    image = Image.new("RGB", (120, 180), "white")

    class FixedClassifier:
        def predict(self, _image):
            return {"predicted_class": "Lab_Report", "document_type": "Lab_Report", "confidence": 0.3,
                    "probabilities": {"Lab_Report": 1.0}}

    class RecordingExtractor:
        called = False
        def extract(self, _image):
            self.called = True
            return {"success": True, "attempts": 1, "data": {"document_type": "Lab_Report"}, "history": []}

    extractor = RecordingExtractor()
    result = MedicalDocumentPipeline(classifier=FixedClassifier(), extractor=extractor).process(image)
    assert extractor.called
    assert result["accepted"] is True
    assert result["qwen_invoked"] is True
    assert result["classification_result"] == {
        "is_medical": True, "document_type": "Lab_Report", "confidence": 1.0
    }
    assert result["evaluation"]["qwen_invoked"] is True


def test_real_resnet_predicts_known_document_images_unchanged():
    classifier = DocumentClassifier()
    samples = [
        (settings.RAW_DATA_PATH / "lab_reports" / "GUR-0425-PA-0057256_E-LABREPORTS_250427_1425@E.pdf_page_1.png", "Lab_Report"),
        (settings.RAW_DATA_PATH / "handwritten_prescription" / "1.jpg", "Prescription"),
        (settings.RAW_DATA_PATH / "discharge_summaries" / "med_doc_discharge_summary_200001_noisy.jpg", "Discharge_Summary"),
    ]
    gate = MedicalDocumentGate()
    for path, expected in samples:
        assert path.is_file(), f"Missing known test image: {path}"
        prediction = classifier.predict(path)
        print(f"{path.name}: predicted={prediction['document_type']} confidence={prediction['confidence']:.4f}")
        assert prediction["document_type"] == expected
        assert gate.evaluate(prediction).is_supported_medical_document


def test_filename_does_not_affect_resnet_prediction(tmp_path):
    classifier = DocumentClassifier()
    image_path = settings.RAW_DATA_PATH / "handwritten_prescription" / "1.jpg"
    disguised_path = tmp_path / "holiday.jpg"
    disguised_path.write_bytes(image_path.read_bytes())
    original = classifier.predict(image_path)
    disguised = classifier.predict(disguised_path)
    assert original == disguised


def test_pdf_gate_uses_resnet_prediction_on_rendered_first_page(tmp_path):
    import fitz

    page_image_path = settings.RAW_DATA_PATH / "handwritten_prescription" / "1.jpg"
    pdf_path = tmp_path / "unrelated_name.pdf"
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(page_image_path))
    document.save(pdf_path)
    document.close()

    classifier = DocumentClassifier()
    pipeline = MedicalDocumentPipeline(
        classifier=classifier,
        extractor=type("RecordingExtractor", (), {
            "extract": lambda self, _image: {
                "success": True, "attempts": 1, "data": {"document_type": "Prescription"}, "history": []
            }
        })(),
        medical_document_gate=MedicalDocumentGate(threshold=0.0),
    )
    result = pipeline.process(pdf_path, run_extraction=False)
    direct = classifier.predict(page_image_path)
    assert result["classification"]["document_type"] == direct["document_type"]
    assert abs(result["confidence"] - direct["confidence"]) < 0.02


def test_pipeline_classifies_all_pdf_pages(tmp_path):
    import fitz

    pdf_path = tmp_path / "multi.pdf"
    document = fitz.open()
    for color in ("white", "black"):
        page = document.new_page(width=200, height=250)
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1) if color == "white" else (0, 0, 0))
    document.save(pdf_path)
    document.close()

    class PageCountingClassifier:
        calls = 0
        def predict(self, _image):
            self.calls += 1
            label = "Lab_Report" if self.calls == 1 else "Prescription"
            return {"predicted_class": label, "document_type": label, "confidence": 0.8,
                    "probabilities": {"Lab_Report": 0.6 if self.calls == 1 else 0.4,
                                      "Prescription": 0.4 if self.calls == 1 else 0.6}}

    classifier = PageCountingClassifier()
    extractor = type("RecordingExtractor", (), {
        "extract": lambda self, _image: {
            "success": True, "attempts": 1, "data": {"document_type": "Lab_Report"}, "history": []
        }
    })()
    result = MedicalDocumentPipeline(classifier=classifier, extractor=extractor).process(pdf_path, run_extraction=False)
    assert classifier.calls == 2
    assert result["document_type"] in {"Lab_Report", "Prescription"}


def test_pipeline_returns_load_error_for_missing_and_invalid_files(tmp_path):
    class NeverCalledClassifier:
        def predict(self, _image):
            raise AssertionError("classification must not run after preprocessing fails")

    pipeline = MedicalDocumentPipeline(classifier=NeverCalledClassifier(), extractor=object())
    missing = pipeline.process(tmp_path / "missing.png")
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_bytes(b"not an image")
    invalid = pipeline.process(invalid_path)
    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")
    empty = pipeline.process(empty_path)
    for result in (missing, invalid, empty):
        assert result["errors"][0]["code"] == "image_load_failed"

