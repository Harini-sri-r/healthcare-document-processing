# 🏥 AI-Based Healthcare Document Processing System

An AI-powered document processing system that classifies healthcare documents and extracts structured information from medical records using deep learning, Vision-Language Models (VLMs), and API-based processing.

The system combines a **ResNet-18 image classification model** with **Qwen2.5-VL-3B**, running through Ollama, to identify document types and extract relevant information into structured JSON.

## ✨ Key Features

- **Medical Document Classification** — Classifies input documents into Lab Reports, Prescriptions, Discharge Summaries, and Non-Medical Documents.
- **AI-Powered Information Extraction** — Uses Qwen2.5-VL-3B to interpret document content and extract relevant fields.
- **Structured JSON Output** — Organizes extracted information into a machine-readable format.
- **Input Validation** — Uses Pydantic-based validation for structured outputs where configured.
- **Review Support** — Supports reviewing extracted information and identifying documents that may require human attention.
- **Ground Truth Comparison** — Provides functionality to compare predictions with reference information where ground-truth data is available.
- **REST API** — Provides backend endpoints through FastAPI.
- **Web Interface** — Offers a frontend for interacting with the document-processing workflow.

## 🧠 System Architecture

```text
             Healthcare Document
                      |
                      v
            Image / Document Input
                      |
                      v
          Preprocessing and Validation
                      |
                      v
           ResNet-18 Classifier
                      |
                      v
              Document Category
                      |
                      v
          Qwen2.5-VL-3B via Ollama
                      |
                      v
          Structured Information Extraction
                      |
                      v
               JSON Validation
                      |
                      v
           Results and Review Interface
```

## 🔍 How It Works

1. **Document input:** A user submits a medical document through the application.
2. **Preprocessing:** The input is prepared for the classification and extraction stages.
3. **Document classification:** A trained ResNet-18 model predicts the document category.
4. **Information extraction:** Qwen2.5-VL-3B processes the document through Ollama and extracts relevant information.
5. **Output validation:** The application validates the generated structured data using the configured validation rules.
6. **Result presentation:** The extracted data is displayed for review, with ground-truth comparison available when reference data exists.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Image classification | PyTorch, Torchvision, ResNet-18 |
| Vision-language model | Qwen2.5-VL-3B |
| Local model runtime | Ollama |
| Backend API | FastAPI |
| Data validation | Pydantic |
| Image processing | Pillow |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest |
| Version control | Git, GitHub |

## 📁 Project Structure

```text
healthcare-document-processing/
├── app/             # Application and API components
├── frontend/        # User interface
├── intelligence/    # AI processing and intelligence modules
├── notebooks/       # Experiments and model development
├── scripts/         # Utility and execution scripts
├── src/             # Core source code and configuration
├── tests/           # Automated tests
├── .env.example     # Example environment configuration
├── .gitignore       # Excluded files and directories
├── predict_document.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the project dependencies
- Git
- Ollama
- Sufficient memory and disk space for the vision-language model

### 1. Clone the repository

```bash
git clone https://github.com/Harini-sri-r/healthcare-document-processing.git
cd healthcare-document-processing
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` if the application uses environment variables.

**PowerShell:**

```powershell
Copy-Item .env.example .env
```

Update the values according to your local environment. Do not commit `.env` or other files containing credentials or sensitive information.

### 5. Set up Ollama

Install Ollama from [ollama.com](https://ollama.com/), start the Ollama service, and obtain the model required by your application.

For example, if your local Ollama setup exposes the model under this name:

```bash
ollama pull qwen2.5vl:3b
```

Confirm the installed model name matches the one configured in your application. The ResNet-18 weights must also be available locally at the configured model path.

### 6. Run the application

Start the FastAPI application using the entry point defined in your project. For example, if the FastAPI instance is `app` in `app.main`:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API if Swagger UI is enabled.

> **Note:** Confirm the actual FastAPI entry-point module and frontend startup instructions in your source code before using these example commands.

## 📊 Model Evaluation

The ResNet-18 classifier achieved **98.37% test accuracy** in the project's reported evaluation experiment.

This figure describes the evaluated classification experiment; it does not represent the accuracy of the entire document-processing pipeline or the correctness of every extracted field.

For reproducible evaluation, document the dataset split, class distribution, evaluation procedure, and per-class precision, recall, and F1-score alongside the accuracy.

## 🔐 Privacy and Responsible Use

Healthcare documents may contain sensitive personal and medical information.

- Use synthetic, anonymized, or appropriately authorized documents for testing.
- Do not upload identifiable medical records, private datasets, credentials, or local database files to a public repository.
- Review model outputs before relying on extracted information.
- Treat generated fields as predictions that may contain errors.
- This project is intended for research and educational purposes, not as a substitute for professional medical judgment.


*Built to explore the application of deep learning and vision-language models to structured healthcare document processing.*
