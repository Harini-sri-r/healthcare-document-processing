from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torchvision import models

from src.config import settings
from src.preprocessing.image_preprocessing import ImageInput, prepare_for_classifier
from src.utils.logging_utils import get_logger

logger = get_logger("classification")


class DocumentClassifier:
    """ResNet-18 Document Classifier fine-tuned on medical documents.

    ResNet v2 classifies documents into:
    - Discharge_Summary
    - Lab_Report
    - Prescription
    - Non_Medical
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        checkpoint_path = Path(model_path or self._default_model_path()).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ResNet checkpoint not found at: {checkpoint_path}")

        requested_device = device or settings.DEVICE
        self.device = torch.device("cuda" if requested_device.startswith("cuda") and torch.cuda.is_available()
                                   else "cpu" if requested_device.startswith("cuda") else requested_device)
        logger.info("Loading ResNet-18 model from %s onto %s", checkpoint_path, self.device)

        # Load checkpoint
        try:
            # The trusted project checkpoint contains a TorchVersion metadata
            # value alongside tensors. Allowlist that value while keeping the
            # restricted weights-only unpickler enabled.
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:  # Compatibility with older PyTorch versions.
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        class_to_idx = checkpoint.get("class_to_idx", checkpoint.get("class_mapping"))
        if state_dict is None or class_to_idx is None:
            raise ValueError("Invalid checkpoint; missing model weights or class mapping.")

        # v3 stores its class order as a mapping rather than a separate list.
        self.class_to_idx = class_to_idx
        self.classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        if set(self.classes) != set(self.class_to_idx) or len(self.classes) != len(self.class_to_idx):
            raise ValueError("Invalid checkpoint; classes and class_to_idx do not match.")

        self.checkpoint_path = checkpoint_path
        self.model_version = (
            "v3" if checkpoint_path == settings.RESNET_V3_MODEL_PATH.resolve()
            else "v2" if checkpoint_path == settings.RESNET_V2_MODEL_PATH.resolve()
            else "baseline"
        )
        expected_mapping = {
            "Discharge_Summary": 0,
            "Lab_Report": 1,
            "Prescription": 2,
            "Non_Medical": 3,
        }
        if checkpoint_path == settings.RESNET_MODEL_PATH.resolve() and checkpoint_path.name == "medical_document_resnet18.pth":
            self.model_version = "baseline"
        if self.model_version in {"v2", "v3"} and self.class_to_idx != expected_mapping:
            raise ValueError(
                f"ResNet {self.model_version} checkpoint has an unexpected class mapping: "
                f"{self.class_to_idx!r}; expected {expected_mapping!r}."
            )
        if self.model_version in {"v2", "v3"}:
            logger.info("Loaded ResNet %s checkpoint: %s", self.model_version, checkpoint_path)
        logger.info("ResNet checkpoint model_version=%s class_to_idx=%s", self.model_version, self.class_to_idx)

        # Instantiate ResNet-18 architecture with matching classification head
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

        # Load trained weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _default_model_path() -> Path:
        """Prefer trained v3, then retain older checkpoint fallback behavior."""
        if settings.RESNET_V3_MODEL_PATH.is_file():
            return settings.RESNET_V3_MODEL_PATH
        if settings.RESNET_V2_MODEL_PATH.is_file():
            return settings.RESNET_V2_MODEL_PATH
        return settings.RESNET_MODEL_PATH

    def predict(self, image: Union[ImageInput, torch.Tensor]) -> Dict[str, Any]:
        """Predict document type, confidence, and class probabilities.

        Args:
            image: Image file path, Path object, PIL Image, or batched torch.Tensor [1, 3, 224, 224].

        Returns:
            Dictionary containing:
            - document_type: str (e.g., "Prescription")
            - confidence: float (between 0.0 and 1.0)
            - probabilities: dict of class_name -> float
        """
        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = prepare_for_classifier(image)

        if image_tensor.ndim != 4:
            raise ValueError(
                f"Expected 4D image tensor [batch, channels, height, width], got shape {image_tensor.shape}"
            )

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)

        pred_idx = predicted_index.item()
        confidence_val = confidence.item()
        document_type = self.classes[pred_idx]

        all_probabilities = {
            self.classes[i]: round(probabilities[0][i].item(), 4)
            for i in range(len(self.classes))
        }

        return {
            "predicted_class": document_type,
            "document_type": document_type,
            "confidence": round(confidence_val, 4),
            "probabilities": all_probabilities,
            "model_version": self.model_version,
        }


__all__ = ["DocumentClassifier"]
