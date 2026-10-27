from __future__ import annotations

from pathlib import Path
from typing import Union
import io

import torch
from PIL import Image
from torchvision import transforms

# ResNet-18 normalization using ImageNet statistics matching training
RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

ImageInput = Union[str, Path, Image.Image]


def load_document_pages(image: ImageInput) -> list[Image.Image]:
    """Render every page of a PDF or return one RGB image page."""
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    path = Path(image)
    if path.suffix.lower() != ".pdf":
        return [load_document(path)]
    try:
        import fitz
    except ImportError as exc:
        raise ValueError("PDF support requires PyMuPDF (pip install pymupdf).") from exc
    pages = []
    with fitz.open(path) as pdf:
        for page in pdf:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pages.append(Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB"))
    if not pages:
        raise ValueError("PDF contains no pages.")
    return pages


def load_document(image: ImageInput) -> Image.Image:
    """Load and return an RGB PIL document image from a file path or PIL image.

    Preserves image structure for both ResNet-18 and Vision-Language processing.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"Document image file not found: {path}")

    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise ValueError("PDF support requires PyMuPDF (pip install pymupdf).") from exc
        with fitz.open(path) as pdf:
            if not pdf.page_count:
                raise ValueError("PDF contains no pages.")
            # First-page classification; extraction uses all rendered pages.
            pixmap = pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    with Image.open(path) as src_img:
        return src_img.convert("RGB")


def prepare_for_classifier(image: ImageInput) -> torch.Tensor:
    """Convert an image path or PIL image into a 4D batch tensor for ResNet-18 [1, 3, 224, 224]."""
    pil_image = load_document(image)
    tensor = RESNET_TRANSFORM(pil_image)
    return tensor.unsqueeze(0)


__all__ = [
    "RESNET_TRANSFORM",
    "ImageInput",
    "load_document",
    "load_document_pages",
    "prepare_for_classifier",
]
