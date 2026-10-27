"""The active pipeline returns validated Qwen output without verification."""

from PIL import Image

from src.pipeline.pipeline import MedicalDocumentPipeline
from src.validation.medical_document_gate import MedicalDocumentGate


class FixedClassifier:
    def predict(self, _tensor):
        return {"predicted_class": "Prescription", "document_type": "Prescription",
                "confidence": .99, "probabilities": {"Prescription": 1.0}}


class FixedExtractor:
    model_name = "qwen2.5vl:3b"

    def __init__(self, data):
        self.data, self.calls = data, 0

    def extract(self, _image):
        self.calls += 1
        return {"success": True, "attempts": 1, "data": self.data,
                "raw_output": '{"dose":"12.5 mg"}', "history": [], "error": None}


class MustNotRun:
    def __init__(self):
        self.calls = 0

    def verify(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("verification must not run during extraction")


class FixedValidator:
    def validate(self, _extraction, document_type=None):
        return {"valid": True, "requires_review": False, "issues": [], "issue_count": 0}


def test_success_returns_qwen_json_without_source_or_second_model_verification(monkeypatch):
    import src.pipeline.pipeline as pipeline_module

    qwen = {"dose": "12.5 mg", "nested": [{"note": None}]}
    source = MustNotRun()
    second = MustNotRun()
    monkeypatch.setattr(pipeline_module.settings, "SECOND_MODEL_VERIFICATION_ENABLED", True)
    pipeline = MedicalDocumentPipeline(
        classifier=FixedClassifier(), extractor=FixedExtractor(qwen),
        source_verifier=source, second_model_verifier=second,
        medical_document_gate=MedicalDocumentGate(threshold=0),
    )
    pipeline.validator = FixedValidator()

    result = pipeline.process(Image.new("RGB", (16, 16), "white"))

    assert result["extracted_json"] == qwen
    assert result["extraction"]["data"] == qwen
    assert result["validation"]["valid"] is True
    assert source.calls == second.calls == 0
    assert "source_verification" not in result
    assert "second_model_verification" not in result
    assert "final_output" not in result


def test_failed_qwen_extraction_returns_clear_error_without_verification():
    class FailedExtractor(FixedExtractor):
        def extract(self, _image):
            return {"success": False, "attempts": 2, "data": None,
                    "raw_output": "not json", "history": [], "error": "Invalid JSON"}

    source = MustNotRun()
    second = MustNotRun()
    pipeline = MedicalDocumentPipeline(
        classifier=FixedClassifier(), extractor=FailedExtractor(None),
        source_verifier=source, second_model_verifier=second,
        medical_document_gate=MedicalDocumentGate(threshold=0),
    )
    result = pipeline.process(Image.new("RGB", (16, 16), "white"))

    assert result["extracted_json"] is None
    assert result["validation"]["valid"] is False
    assert any(error["code"] == "extraction_error" for error in result["errors"])
    assert source.calls == second.calls == 0
