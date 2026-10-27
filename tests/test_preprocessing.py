from pathlib import Path
import unittest

import torch
from PIL import Image

from src.config import settings
from src.preprocessing.image_preprocessing import load_document, prepare_for_classifier


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_image = next((settings.RAW_DATA_PATH / "handwritten_prescription").glob("*.jpg"))

    def test_load_document_converts_real_image_to_rgb(self):
        image = load_document(self.sample_image)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")

    def test_prepare_for_classifier_returns_resnet_batch(self):
        tensor = prepare_for_classifier(self.sample_image)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
