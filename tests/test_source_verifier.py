"""Source verification stays a separate feature, outside normal extraction."""

from PIL import Image

from src.extraction.source_verifier import SourceVerifier
from src.pipeline.pipeline import MedicalDocumentPipeline


class FixedClassifier:
    def __init__(self, category="Lab_Report"):
        self.category = category
        self.calls = 0

    def predict(self, _tensor):
        self.calls += 1
        return {"predicted_class": self.category, "document_type": self.category,
                "confidence": 0.99, "probabilities": {self.category: 1.0}}


class FixedExtractor:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def extract(self, _image):
        self.calls += 1
        return self.result


class MustNotRunVerifier:
    def __init__(self):
        self.calls = 0

    def verify(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("verification must not run during normal processing")


def extraction_result(data=None, success=True):
    return {"success": success, "attempts": 1, "data": data,
            "raw_output": '{"value":"12"}' if success else "invalid JSON",
            "history": [], "error": None if success else "invalid JSON"}


def test_successful_qwen_json_returns_without_calling_either_verifier():
    payload = {"document_type": "Lab_Report", "results": [{"label": "X", "value": "12"}]}
    classifier = FixedClassifier()
    extractor = FixedExtractor(extraction_result(payload))
    source_verifier, second_verifier = MustNotRunVerifier(), MustNotRunVerifier()
    pipeline = MedicalDocumentPipeline(classifier=classifier, extractor=extractor,
        source_verifier=source_verifier, second_model_verifier=second_verifier,
        classification_threshold=0)

    result = pipeline.process(Image.new("RGB", (32, 32), "white"))

    assert result["extracted_json"] == payload
    assert result["validation"]["valid"] is True
    assert extractor.calls == classifier.calls == 1
    assert source_verifier.calls == second_verifier.calls == 0
    assert "source_verification" not in result
    assert "second_model_verification" not in result


def test_failed_extraction_reports_clear_error_without_verification():
    extractor = FixedExtractor(extraction_result(None, success=False))
    source_verifier, second_verifier = MustNotRunVerifier(), MustNotRunVerifier()
    pipeline = MedicalDocumentPipeline(classifier=FixedClassifier(), extractor=extractor,
        source_verifier=source_verifier, second_model_verifier=second_verifier,
        classification_threshold=0)

    result = pipeline.process(Image.new("RGB", (32, 32), "white"))

    assert result["validation"]["valid"] is False
    assert result["review"]["status"] == "HUMAN_REVIEW"
    assert result["extracted_json"] is None
    assert any(error["code"] == "extraction_error" for error in result["errors"])
    assert source_verifier.calls == second_verifier.calls == 0


def test_nonmedical_is_rejected_without_calling_qwen_or_verifiers():
    extractor = FixedExtractor(extraction_result({}))
    source_verifier, second_verifier = MustNotRunVerifier(), MustNotRunVerifier()
    pipeline = MedicalDocumentPipeline(classifier=FixedClassifier("Non_Medical"), extractor=extractor,
        source_verifier=source_verifier, second_model_verifier=second_verifier,
        classification_threshold=0)

    result = pipeline.process(Image.new("RGB", (32, 32), "white"))

    assert result["qwen_invoked"] is False
    assert result["status"] == "REJECTED"
    assert extractor.calls == 0
    assert source_verifier.calls == second_verifier.calls == 0


# The verifier itself remains available for separate, explicitly requested use.
def test_source_verifier_parses_explicit_response():
    class FakeOllama:
        def generate_raw_text(self, images, prompt):
            assert len(images) == 1
            assert "arbitrary" in prompt
            return {"success": True,
                    "text": '{"verification_status":"VERIFIED","confidence":0.9,"issues":[]}'}

    result = SourceVerifier(extractor=FakeOllama()).verify(
        Image.new("RGB", (10, 10)), {"arbitrary": {"deep": "value"}})
    assert result == {"status": "VERIFIED", "confidence": 0.9, "issues": []}


def test_source_verifier_rejects_malformed_or_multiple_json():
    class FakeOllama:
        def __init__(self, response):
            self.response = response

        def generate_raw_text(self, _images, _prompt):
            return {"success": True, "text": self.response}

    for text in ('{"verification_status":"VERIFIED",',
                 '{"verification_status":"VERIFIED","confidence":0.9,"issues":[]} {"extra":true}'):
        result = SourceVerifier(extractor=FakeOllama(text)).verify(
            Image.new("RGB", (10, 10)), {"x": 1})
        assert result["status"] == "UNAVAILABLE"
        assert result["issues"][0]["reason"]


def test_incomplete_or_unrecognized_explicit_verifier_result_is_unavailable():
    class FakeOllama:
        def __init__(self, response):
            self.response = response

        def generate_raw_text(self, _images, _prompt):
            return {"success": True, "text": self.response}

    for response in (
        '{"verification_status":"VERIFIED","confidence":0.9}',
        '{"verification_status":"SOMETHING_ELSE","confidence":0.9,"issues":[]}',
    ):
        result = SourceVerifier(extractor=FakeOllama(response)).verify(
            Image.new("RGB", (10, 10)), {"arbitrary": "value"})
        assert result["status"] == "UNAVAILABLE"


def test_explicit_issue_result_with_malformed_entries_is_serializable():
    result = SourceVerifier._normalize_result({
        "verification_status": "REVIEW_REQUIRED", "confidence": 0.5,
        "issues": [None, {"path": "custom.field", "reason": "unclear"}],
    })
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["issues"][0]["reason"] == "Verifier returned a malformed issue entry."
    assert result["issues"][1]["path"] == "custom.field"
