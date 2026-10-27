import unittest
from pathlib import Path
import torch
from PIL import Image

from src.config import settings
from src.extraction.json_validator import JSONValidator, clean_and_parse_json
from src.extraction.prompts import build_medical_extraction_prompt
from src.extraction.qwen_extractor import QwenExtractor


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return _FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, *args, **kwargs):
        return ['```json\n{"patient_name": "John Doe", "test": "Hemoglobin", "value": "14.2", "unit": "g/dL"}\n```']


class _FakeModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])


class QwenExtractorTests(unittest.TestCase):
    def test_local_qwen_model_path_exists(self):
        self.assertTrue(
            settings.QWEN_MODEL_PATH.is_dir(),
            f"Local Qwen directory missing: {settings.QWEN_MODEL_PATH}",
        )
        self.assertTrue(
            (settings.QWEN_MODEL_PATH / "config.json").is_file(),
            "config.json missing from local Qwen directory",
        )
        self.assertTrue(
            (settings.QWEN_MODEL_PATH / "model.safetensors.index.json").is_file(),
            "model.safetensors.index.json missing from local Qwen directory",
        )

    def test_initialization_is_lazy(self):
        extractor = QwenExtractor()
        self.assertFalse(extractor.is_loaded)
        self.assertEqual(extractor.model_path, settings.QWEN_MODEL_PATH)

    def test_prompt_satisfies_requirements(self):
        prompt = build_medical_extraction_prompt().lower()
        self.assertIn("json", prompt)
        self.assertIn("document_type", prompt)
        self.assertIn("do not provide medical diagnoses", prompt)
        self.assertIn("null", prompt)
        self.assertIn("unclear", prompt)


    def test_json_cleaner_handles_code_fences(self):
        raw = '```json\n{"test": "WBC", "value": "8.5"}\n```'
        parsed, err = clean_and_parse_json(raw)
        self.assertIsNone(err)
        self.assertEqual(parsed, {"test": "WBC", "value": "8.5"})

    def test_extraction_with_injected_model(self):
        extractor = QwenExtractor(
            model=_FakeModel(),
            processor=_FakeProcessor(),
            device="cpu",
        )
        test_img = Image.new("RGB", (32, 32))
        result = extractor.extract(test_img)

        self.assertTrue(result["success"])
        self.assertIsNotNone(result["data"])
        self.assertEqual(result["data"]["patient_name"], "John Doe")
        self.assertEqual(result["data"]["unit"], "g/dL")

    def test_validator_on_extracted_data(self):
        validator = JSONValidator()
        valid_res = {
            "success": True,
            "data": {"measurement": "Glucose", "result": "95", "unit": "mg/dL"},
        }
        val = validator.validate(valid_res)
        self.assertTrue(val["valid"])
        self.assertFalse(val["requires_review"])

    def test_invalid_image_path_reports_structured_error(self):
        extractor = QwenExtractor()
        result = extractor.extract("non_existent_document_image_xyz.jpg")
        self.assertFalse(result["success"])
        self.assertIn("Failed to load image", result["error"])


if __name__ == "__main__":
    unittest.main()
