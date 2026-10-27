import json
import base64
import os
import subprocess
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from src.extraction.second_model_verifier import SecondModelVerifier, build_verification_prompt


class FakeModel:
    def __init__(self, response=None, error=None): self.response, self.error, self.images, self.prompt = response, error, None, None
    def generate_raw_text(self, images, prompt):
        self.images, self.prompt = images, prompt
        if self.error: raise self.error
        return self.response


def response(report): return {"success": True, "text": json.dumps(report)}


def test_granite_default_and_environment_override():
    env = os.environ.copy()
    env.pop("MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_MODEL", None)
    env.pop("MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_ENABLED", None)
    code = "from src.config import settings; print(settings.SECOND_MODEL_VERIFICATION_MODEL + '|' + str(settings.SECOND_MODEL_VERIFICATION_ENABLED))"
    default = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
    assert default == "granite3.2-vision:2b|False"
    env["MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_MODEL"] = "local-vision-override:latest"
    override = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
    assert override == "local-vision-override:latest|False"


def test_ollama_chat_sends_original_image_and_qwen_json_to_granite():
    original = Image.new("RGB", (13, 9), (17, 83, 201))
    report = {"overall_status": "CONSISTENT", "review_notes": [], "field_reviews": [
        {"field_path": '$["value"]', "status": "CONSISTENT"}]}
    fake_response = MagicMock()
    fake_response.json.return_value = {"message": {"content": json.dumps(report)}}
    fake_response.raise_for_status.return_value = None
    verifier = SecondModelVerifier(model="granite3.2-vision:2b", timeout=4)
    with patch("src.extraction.ollama_extractor.requests.post", return_value=fake_response) as post:
        result = verifier.verify([original], '{"value":12.5}', {"value": 12.5})
    assert result["overall_status"] == "CONSISTENT"
    request = post.call_args.kwargs
    assert request["timeout"] == 4
    payload = request["json"]
    assert payload["model"] == "granite3.2-vision:2b"
    assert payload["keep_alive"] == 0
    message = payload["messages"][0]
    assert message["role"] == "user"
    assert '{"value":12.5}' in message["content"] and '"value": 12.5' in message["content"]
    assert len(message["images"]) == 1
    decoded = Image.open(BytesIO(base64.b64decode(message["images"][0]))).convert("RGB")
    assert decoded.size == original.size and decoded.tobytes() == original.tobytes()


def test_consistent_dynamic_nested_and_array_fields():
    data = {"reading": {"value": 12.5, "unit": "mg"}, "repeat": [{"date": "2025-01-02"}]}
    paths = ['$["reading"]["value"]', '$["reading"]["unit"]', '$["repeat"][0]["date"]']
    fake = FakeModel(response({"overall_status": "CONSISTENT", "review_notes": [], "field_reviews": [
        {"field_path": p, "status": "CONSISTENT"} for p in paths]}))
    result = SecondModelVerifier(model="test-vlm", extractor=fake).verify([Image.new("RGB", (5, 5))], "raw qwen", data)
    assert result["overall_status"] == "CONSISTENT"
    assert len(result["field_reviews"]) == 3
    assert "raw qwen" in fake.prompt and "12.5" in fake.prompt
    assert result["field_reviews"][0]["extracted_value"] == 12.5


def test_numeric_or_unit_disagreement_is_preserved_and_flagged():
    data = {"result": 12.5, "unit": "mg"}
    fake = FakeModel(response({"overall_status": "DISCREPANCY_DETECTED", "review_notes": [], "field_reviews": [
        {"field_path": '$["result"]', "status": "DISCREPANCY_DETECTED", "observed_source_value": "125", "reason": "Decimal differs"},
        {"field_path": '$["unit"]', "status": "CONSISTENT"}]}))
    result = SecondModelVerifier(extractor=fake).verify([Image.new("RGB", (4, 4))], None, data)
    assert result["overall_status"] == "DISCREPANCY_DETECTED"
    assert result["field_reviews"][0]["extracted_value"] == 12.5
    assert result["field_reviews"][0]["observed_source_value"] == "125"
    assert result["field_reviews"][1]["extracted_value"] == "mg"


def test_unreadable_and_not_verifiable_do_not_invent_observation():
    data = {"handwriting": "unclear", "missing_context": None}
    fake = FakeModel(response({"overall_status": "REVIEW_REQUIRED", "review_notes": ["Human review"], "field_reviews": [
        {"field_path": '$["handwriting"]', "status": "UNREADABLE", "observed_source_value": "invented", "reason": "Illegible"},
        {"field_path": '$["missing_context"]', "status": "NOT_VERIFIABLE", "observed_source_value": "other", "reason": "Not shown"}]}))
    result = SecondModelVerifier(extractor=fake).verify([Image.new("RGB", (4, 4))], None, data)
    assert result["overall_status"] == "REVIEW_REQUIRED"
    assert [x["observed_source_value"] for x in result["field_reviews"]] == [None, None]


def test_invalid_extraction_json_runs_model_but_routes_to_review():
    fake = FakeModel(response({"overall_status": "CONSISTENT", "review_notes": [], "field_reviews": []}))
    result = SecondModelVerifier(extractor=fake).verify([Image.new("RGB", (3, 3))], "not valid json", None)
    assert result["overall_status"] == "REVIEW_REQUIRED"
    assert any("not valid JSON" in note for note in result["review_notes"])
    assert "not valid json" in fake.prompt


def test_invalid_verifier_json_and_incomplete_or_bad_paths_are_not_success():
    for fake in (FakeModel({"success": True, "text": "not json"}),
                 FakeModel(response({"overall_status": "CONSISTENT", "review_notes": [], "field_reviews": [
                     {"field_path": '$["wrong"]', "status": "CONSISTENT"}]}))):
        result = SecondModelVerifier(extractor=fake).verify([Image.new("RGB", (3, 3))], None, {"actual": 1})
        assert result["overall_status"] == "REVIEW_REQUIRED"
        if result["availability"] == "AVAILABLE":
            assert result["field_reviews"][0]["status"] == "REVIEW_REQUIRED"
        else:
            assert result["availability"] == "UNAVAILABLE"


def test_model_failure_timeout_and_missing_image_are_unavailable():
    for error in (TimeoutError("timeout"), RuntimeError("offline")):
        result = SecondModelVerifier(extractor=FakeModel(error=error)).verify([Image.new("RGB", (2, 2))], None, {"x": 1})
        assert result["availability"] == "UNAVAILABLE"
        assert result["overall_status"] == "REVIEW_REQUIRED"
    result = SecondModelVerifier(extractor=FakeModel()).verify([], None, {"x": 1})
    assert result["availability"] == "UNAVAILABLE"


def test_prompt_requires_independent_image_check_and_precision():
    prompt = build_verification_prompt("raw", {"dose": "12.5 mg"})
    assert "directly" in prompt and "do not trust" in prompt
    assert "12.5 mg differs from 125 mg" in prompt
    assert "12.5 mg" in prompt
