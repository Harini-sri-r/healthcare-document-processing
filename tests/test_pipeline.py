import unittest
from pathlib import Path
import torch
from PIL import Image

from src.config import settings
from src.extraction.qwen_extractor import QwenExtractor
from src.pipeline.pipeline import MedicalDocumentPipeline
from src.validation.medical_document_gate import MedicalDocumentGate


class _MockProcessor:
    def apply_chat_template(self, *args, **kwargs):
        class _Inputs(dict):
            def to(self, device):
                return self
        return _Inputs(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, *args, **kwargs):
        return ['{"document_type":"Prescription","patient":{"name":null,"age":null,"sex":null},'
                '"doctor":{"name":null,"address":null,"telephone":null},'
                '"prescription":{"date":null,"medications":[]},"additional_information":[],'
                '"source":{"uncertain_fields":[]}}']


class _MockModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_image = next((settings.RAW_DATA_PATH / "handwritten_prescription").glob("*.jpg"))

    def test_pipeline_always_extracts_after_medical_classification(self):
        class RecordingExtractor:
            called = False

            def extract(self, _image):
                self.called = True
                return {"success": True, "attempts": 1,
                        "data": {"document_type": "Prescription"}, "history": []}

        extractor = RecordingExtractor()
        gate = MedicalDocumentGate()
        pipeline = MedicalDocumentPipeline(medical_document_gate=gate, extractor=extractor)
        result = pipeline.process(self.sample_image, run_extraction=False)

        self.assertIn("document_type", result)
        self.assertIn("confidence", result)
        self.assertIn("classification", result)
        self.assertEqual(result["document_type"], result["classification"]["document_type"])
        self.assertTrue(extractor.called)
        self.assertTrue(result["qwen_invoked"])
        self.assertTrue(result["classification_result"]["is_medical"])

    def test_pipeline_with_mock_extractor(self):
        mock_extractor = QwenExtractor(
            model=_MockModel(),
            processor=_MockProcessor(),
            device="cpu",
        )
        gate = MedicalDocumentGate(threshold=0.0)
        pipeline = MedicalDocumentPipeline(extractor=mock_extractor, medical_document_gate=gate)
        result = pipeline.process(self.sample_image, run_extraction=True)

        self.assertTrue(result["extraction"]["success"])
        self.assertIsNotNone(result["extracted_json"])
        self.assertEqual(result["extracted_json"]["document_type"], "Prescription")
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(len(result["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
