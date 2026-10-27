import json
from pathlib import Path

import fitz
import pytest
import torch
from PIL import Image

import predict_document
from predict_document import DocumentPredictor


REAL_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "non_medical" / "cord_receipt_0001.jpg"
)


class SequenceModel:
    def __init__(self, logits):
        self.logits = [torch.tensor([row], dtype=torch.float32) for row in logits]
        self.index = 0

    def __call__(self, _tensor):
        value = self.logits[min(self.index, len(self.logits) - 1)]
        self.index += 1
        return value


def predictor(logits=None, threshold=0.70):
    instance = DocumentPredictor.__new__(DocumentPredictor)
    instance.threshold = threshold
    instance.device = torch.device("cpu")
    instance.model = SequenceModel(logits or [[0.0, 10.0, 0.0, 0.0]])
    return instance


def make_pdf(path: Path, image_path: Path, pages=1):
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page(width=432, height=648)
        page.insert_image(page.rect, filename=str(image_path))
    document.save(path)
    document.close()
    return path


@pytest.fixture
def real_image_path():
    assert REAL_IMAGE.is_file(), "The real CORD receipt image is required by these tests."
    return REAL_IMAGE


def test_single_jpg_and_png_use_content_and_return_prediction(tmp_path, real_image_path):
    png = tmp_path / "document123.png"
    with Image.open(real_image_path) as source:
        source.save(png)
    jpg_result = predictor().predict(real_image_path)
    png_result = predictor().predict(png)
    assert jpg_result["file_type"] == "image"
    assert png_result["file_type"] == "image"
    assert jpg_result["predicted_class"] == "Lab_Report"
    assert png_result["model_version"] == "v2"
    assert jpg_result["accepted"] is True


def test_single_pdf_prediction(tmp_path, real_image_path):
    pdf = make_pdf(tmp_path / "document.pdf", real_image_path)
    result = predictor().predict(pdf)
    assert result["file_type"] == "pdf"
    assert result["page_count"] == 1
    assert result["per_page_predictions"][0]["predicted_class"] == "Lab_Report"


def test_multipage_pdf_averages_probabilities_and_reports_each_page(tmp_path, real_image_path):
    pdf = make_pdf(tmp_path / "multi.pdf", real_image_path, pages=2)
    mock_predictor = predictor([
        [5.0, 5.1, 0.0, 0.0],  # Lab_Report page, close to the boundary
        [10.0, 0.0, 0.0, 0.0],  # Discharge_Summary page
    ])
    result = mock_predictor.predict(pdf)
    assert result["page_count"] == 2
    assert [p["predicted_class"] for p in result["per_page_predictions"]] == [
        "Lab_Report", "Discharge_Summary"
    ]
    assert result["predicted_class"] == "Discharge_Summary"


def test_low_confidence_is_unknown_and_rejected(tmp_path, real_image_path):
    result = predictor([[0.0, 0.0, 0.0, 0.0]], threshold=0.70).predict(real_image_path)
    assert result["predicted_class"] == "Unknown/Low_Confidence"
    assert result["model_prediction"] == "Discharge_Summary"
    assert result["accepted"] is False


def test_nonmedical_is_rejected_without_extraction(real_image_path):
    result = predictor([[0.0, 0.0, 0.0, 10.0]]).predict(real_image_path)
    assert result["predicted_class"] == "Non_Medical"
    assert result["accepted"] is False


def test_folder_recursively_includes_images_and_pdfs_and_writes_results(
    tmp_path, real_image_path, monkeypatch, capsys
):
    folder = tmp_path / "test_new_documents"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    image_copy = nested / "real_receipt.jpg"
    image_copy.write_bytes(real_image_path.read_bytes())
    make_pdf(folder / "real_receipt.pdf", real_image_path)
    (folder / "notes.txt").write_text("unsupported", encoding="utf-8")

    class FakePredictor:
        def __init__(self, *_args, **_kwargs):
            pass

        def predict(self, path, filename=None):
            file_type = "pdf" if path.suffix.lower() == ".pdf" else "image"
            result = {
                "filename": filename,
                "file_type": file_type,
                "predicted_class": "Non_Medical",
                "confidence": 0.96,
                "model_version": "v2",
                "accepted": False,
            }
            if file_type == "pdf":
                result.update({"page_count": 1, "per_page_predictions": []})
            return result

    monkeypatch.setattr(predict_document, "DocumentPredictor", FakePredictor)
    results_dir = tmp_path / "results"
    code = predict_document.main([
        str(folder), "--results-dir", str(results_dir),
    ])
    output = capsys.readouterr().out
    assert code == 0
    assert "Supported files found: 2" in output
    assert "Ignored unsupported files: 1" in output
    assert (results_dir / "predictions.csv").is_file()
    saved = json.loads((results_dir / "predictions.json").read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert {row["file_type"] for row in saved} == {"image", "pdf"}


def test_empty_folder_and_list_mode_are_clear(tmp_path, capsys):
    folder = tmp_path / "empty"
    folder.mkdir()
    assert predict_document.main([str(folder), "--list"]) == 0
    output = capsys.readouterr().out
    assert f"No supported documents found in {folder}." in output
    assert ".jpg, .jpeg, .png, .webp, .pdf" in output


def test_unsupported_file_is_reported(tmp_path, capsys):
    unsupported = tmp_path / "readme.txt"
    unsupported.write_text("not a document image", encoding="utf-8")
    assert predict_document.main([str(unsupported)]) == 2
    assert "Unsupported file type" in capsys.readouterr().err


def test_missing_path_is_reported(tmp_path, capsys):
    assert predict_document.main([str(tmp_path / "missing.pdf")]) == 2
    assert "Input path does not exist" in capsys.readouterr().err


def test_corrupt_pdf_returns_document_error(tmp_path):
    corrupt = tmp_path / "broken.pdf"
    corrupt.write_bytes(b"not a PDF")
    result = predictor().predict(corrupt)
    assert result["file_type"] == "pdf"
    assert result["accepted"] is False
    assert result["page_count"] == 0
    assert "error" in result
