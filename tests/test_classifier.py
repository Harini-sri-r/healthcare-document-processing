from pathlib import Path
import unittest

from src.classification.classification import DocumentClassifier
from src.config import settings

EXPECTED_MAPPING = {
    "Discharge_Summary": 0,
    "Lab_Report": 1,
    "Prescription": 2,
    "Non_Medical": 3,
}
EXPECTED_CLASSES = set(EXPECTED_MAPPING)


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_image = next((settings.RAW_DATA_PATH / "handwritten_prescription").glob("*.jpg"))

    def test_checkpoint_loads_with_expected_classes(self):
        self.assertTrue(
            settings.RESNET_V3_MODEL_PATH.is_file(),
            f"ResNet v3 checkpoint missing at {settings.RESNET_V3_MODEL_PATH}",
        )
        classifier = DocumentClassifier()
        self.assertEqual(set(classifier.classes), EXPECTED_CLASSES)
        self.assertEqual(classifier.class_to_idx, EXPECTED_MAPPING)
        self.assertEqual(classifier.model_version, "v3")
        self.assertEqual(classifier.checkpoint_path, settings.RESNET_V3_MODEL_PATH.resolve())

    def test_classifier_returns_valid_result_for_real_document(self):
        classifier = DocumentClassifier()
        result = classifier.predict(self.sample_image)

        self.assertEqual(set(result.keys()), {"predicted_class", "document_type", "confidence", "probabilities", "model_version"})
        self.assertEqual(result["predicted_class"], result["document_type"])
        self.assertIn(result["document_type"], EXPECTED_CLASSES)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(set(result["probabilities"].keys()), EXPECTED_CLASSES)
        self.assertTrue(all(0.0 <= val <= 1.0 for val in result["probabilities"].values()))
        self.assertLess(abs(sum(result["probabilities"].values()) - 1.0), 0.001)

    def test_explicit_baseline_checkpoint_remains_loadable(self):
        classifier = DocumentClassifier(model_path=settings.MODELS_DIR / "medical_document_resnet18.pth")
        self.assertEqual(set(classifier.classes), {"Discharge_Summary", "Lab_Report", "Prescription"})
        self.assertEqual(classifier.model_version, "baseline")
        self.assertEqual(classifier.checkpoint_path, (settings.MODELS_DIR / "medical_document_resnet18.pth").resolve())


if __name__ == "__main__":
    unittest.main()

