<<<<<<< HEAD
# Production Healthcare Document AI System

A production-oriented AI system for medical document classification, extraction, and validation.

## Architecture

```text
Medical Document Image
        ↓
Image Preprocessing
        ↓
ResNet-18 Document Classifier
        ↓
Document Type + Confidence
        ↓
Qwen2.5-VL-3B-Instruct (Local Model)
        ↓
Dynamic Medical Information Extraction
        ↓
JSON Validation & Audit
        ↓
Final Structured Result
```

## Supported Document Classes

1. `Prescription` (Handwritten & typed)
2. `Lab_Report` (Pathology, hematology, biochemistry, etc.)
3. `Discharge_Summary` (Inpatient clinical discharge reports)

## Medical Document Confidence Gate

The upload pipeline runs the existing ResNet-18 classifier first, then applies
a configurable confidence gate before Qwen extraction. Set
`RESNET_MEDICAL_CONFIDENCE_THRESHOLD` (default `0.80`) to adjust the minimum
confidence. This is a provisional rejection mechanism, not a validated
non-medical detector: the existing ResNet has only the three supported medical
classes and always chooses one. Evaluate representative medical and
non-medical images before selecting a production threshold. Rejected inputs
return the ResNet prediction and confidence and are not sent to Qwen.

## Project Structure

```text
healthcare_document_processing/
│
├── .venv/                              # Python virtual environment
│
├── models/
│   ├── medical_document_resnet18.pth   # Fine-tuned ResNet-18 checkpoint (44.7 MB)
│   └── qwen2.5-vl-3b-instruct/         # Model metadata (Ollama supplies runtime weights)
│       ├── config.json
│       ├── generation_config.json
│       ├── preprocessor_config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── vocab.json
│       └── model.safetensors.index.json (metadata only; Ollama runs Qwen in production)
│
??? data/
?   ??? raw/                           # Complete combined source dataset; folders are classes
?   ?   ??? discharge_summaries/       # 500 readable source images
?   ?   ??? lab_reports/               # 516 readable source images
?   ?   ??? handwritten_prescription/  # 129 readable source images
?   ?   ??? non_medical/               # 700 readable source images
?   ??? splits/final_reproducible_v1/  # Frozen stratified train/validation/test subsets
?   ??? processed/                     # Processed / cached dataset artifacts
├── src/
│   ├── __init__.py
│   ├── config.py                       # Centralized, portable paths and settings
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── image_preprocessing.py      # ResNet normalization & image loading
│   ├── classification/
│   │   ├── __init__.py
│   │   └── classification.py           # ResNet-18 DocumentClassifier
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── qwen_extractor.py           # Local Qwen2.5-VL extractor with lazy load
│   │   ├── prompts.py                  # Dynamic schema extraction instructions
│   │   └── json_validator.py           # JSON cleaner, fence stripper & validator
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── pipeline.py                 # End-to-end MedicalDocumentPipeline
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py            # Standardized logger
│       └── evaluation.py               # Field-level ground truth evaluation
│   ├── db/
│   │   ├── database.py                 # SQLAlchemy connection setup
│   │   └── models.py                   # Patient, Document, Extraction schema
│   └── services/
│       ├── document_service.py         # DB orchestration for documents
│       ├── patient_service.py          # Timeline retrieval
│       ├── search_service.py           # Document search
│       └── analytics_service.py        # Hospital analytics
│
├── tests/
│   ├── test_classifier.py              # ResNet classifier unit tests
│   ├── test_preprocessing.py           # Preprocessing and tensor shape tests
│   ├── test_qwen_extractor.py          # Local Qwen extractor & validator tests
│   ├── test_pipeline.py                # Pipeline integration tests
│   ├── verify_resnet.py                # Standalone ResNet verification script
│   └── verify_qwen.py                  # Standalone local Qwen path verification
│
├── reports/                            # Evaluation and audit reports
│
├── app/
│   ├── api.py                          # FastAPI backend application
│   └── main.py                         # Minimal CLI pipeline entry point
│
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation
└── .gitignore                          # Git exclusions
```

## Setup and Installation

1. Activate your virtual environment:
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run Classification Only (Quick)
```bash
python app/main.py data/raw/handwritten_prescription/1.jpg --classifier-only
```

Output:
```text
============================================================
 HEALTHCARE DOCUMENT AI PIPELINE
============================================================
 Document Image : data/raw/handwritten_prescription/1.jpg
 ResNet Path    : .../models/medical_document_resnet18.pth
 Qwen Path      : .../models/qwen2.5-vl-3b-instruct
 Mode           : Classifier Only
============================================================

--- 1. DOCUMENT CLASSIFICATION ---
Predicted Type : Prescription
Confidence     : 99.85%
Probabilities  :
  - Discharge_Summary   : 0.05%
  - Lab_Report          : 0.10%
  - Prescription        : 99.85%

--- 4. REVIEW STATUS ---
Status  : CLASSIFICATION_COMPLETE
```

### 2. Run Complete Pipeline (Classification + Extraction + Validation)
```bash
python app/main.py path/to/document.jpg
```

### 3. Run the FastAPI Backend Server
Start the API to expose the document intelligence layer:
```bash
uvicorn app.api:app --reload
```
The API documentation (Swagger UI) will be available at `http://127.0.0.1:8000/docs`.
The frontend is available at `http://127.0.0.1:8000/pipeline`.

The production extraction path sends document images to the local Ollama
`qwen2.5vl:3b` model. Run `ollama serve` and `ollama pull qwen2.5vl:3b` first.
PDF uploads are rendered page by page for visual extraction; install PyMuPDF for
PDF support; extraction currently caps PDFs at 20 pages per request. Original uploads are retained under `data/originals/` and are excluded
from version control. Patient timeline linking requires a visible stable patient ID;
names alone are not used to merge records.

Install local dependencies with `python -m pip install -r requirements.txt`. Install
PyTorch/torchvision compatible with your machine before the requirements if needed. Set
`DATABASE_URL` to a SQLAlchemy PostgreSQL URL for PostgreSQL; SQLite is the local default.
Start Ollama with `ollama serve`, pull `qwen2.5vl:3b`, then run `uvicorn app.api:app --reload`.
Existing SQLite databases created before the schema additions require a migration before
startup; `create_all` does not add columns to existing tables.

## Testing

Run the mocked unit and component tests (does not call Ollama):
```bash
python -m pytest tests -v
```

Live Qwen inference is never run by pytest. Run `python tests/test_ollama_live.py`
manually after starting Ollama to validate a real document. No extraction ground truth
is configured, so the project does not report extraction accuracy. The Qwen metadata
directory does not include weight shards; install `qwen2.5vl:3b` in Ollama for inference.


Verify models individually:
```bash
python tests/verify_resnet.py
python tests/verify_qwen.py
```


## Dataset splitting (reproducible v1)

`data/raw/` is the complete combined source dataset. Its four subfolders are document classes, not preassigned training, validation, or test folders. Every readable image is validated and hashed with SHA-256; exact duplicate files stay in `data/raw/`, while one deterministic representative per hash enters the candidate pool. The unique candidates are split by class with seed 42: 80% training, 10% validation, and 10% testing. The final split contains 1,468 training, 183 validation, and 184 held-out test images. The held-out test set is part of the combined source dataset, not an external dataset.

The physical copies and six-column manifest are under `data/splits/final_reproducible_v1/`. The manifest records `split`, `class`, `filename`, `source_path`, `split_path`, and `sha256`. Exact split counts, class counts, duplicate audit, and leakage verification are documented in `reports/reproducible_v1_dataset_split_report.txt` and `reports/reproducible_v1_data_leakage_check.json`. The leakage check confirms disjoint SHA-256 sets and source paths across all three subsets.

Pipeline:

```text
RAW COMBINED DATASET
        ?
IMAGE VALIDATION
        ?
SHA-256 DUPLICATE CHECK
        ?
UNIQUE IMAGE POOL
        ?
STRATIFIED SPLIT (seed 42; 80% / 10% / 10%)
        ??? TRAIN
        ??? VALIDATION
        ??? HELD-OUT TEST
                 ?
          ResNet-18 Training
                 ?
           Best Checkpoint
                 ?
          Final Test Evaluation
```
# Optional second-model extraction verification

When enabled, the normal document upload pipeline runs a separate Granite Vision verification pass immediately after the existing Qwen extraction and validation. It sends the original rendered image/PDF pages, the raw Qwen response when retained, and the parsed Qwen JSON when valid through the existing local Ollama service. It does not change ResNet classification or overwrite `DocumentExtraction.extraction_json`. With verification disabled, uploads follow the existing pipeline unchanged.

Automatic verification stays disabled by default. To enable it locally, set `MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_ENABLED=true` in the API process environment and restart the API. The default model is `granite3.2-vision:2b`; override it with `MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_MODEL` if needed. `.env.example` lists the options and keeps the default false. The configured tag must exist in local Ollama; the application does not pull models or substitute another model. The upload result shows Qwen's original JSON, Granite findings, the unchanged pipeline JSON, and fields requiring review. The `/verification` page remains available for a separate manual rerun and saving reviewer resolutions. A report showing model agreement is not ground truth; only human confirmation can establish ground truth.

The verifier sends original rendered image pages in Ollama `/api/chat`'s `messages[].images` base64 image input and includes Qwen's original parsed JSON (and raw text when stored) in the prompt. When automatic verification is enabled, the pipeline asks Ollama to release Qwen before loading Granite; Granite is requested with `keep_alive=0` so it is unloaded afterward. Failed Qwen release prevents Granite inference and routes the document to human review. This reduces overlapping model residency on CPU-only computers. Ollama/model inference can take several minutes; `MEDICAL_DOCUMENT_SECOND_MODEL_VERIFICATION_TIMEOUT` defaults to 600 seconds. Actual latency is recorded per run. CPU-only use may be slow; available memory requirements depend on the Ollama model build and quantization.

New raw Qwen output is stored in `document_extraction_raw`; verification reports and separate reviewer resolutions are stored in `document_verification_runs`. Older records have no recoverable raw output. They can still be compared using stored parsed JSON. For privacy, verification uses the configured local Ollama endpoint; do not point it at a remote endpoint unless that data handling is approved for the documents being processed.
=======
# healthcare-document-processing
AI-powered healthcare document classification and structured information extraction using ResNet-18, Qwen2.5-VL, and FastAPI.
>>>>>>> origin/main
