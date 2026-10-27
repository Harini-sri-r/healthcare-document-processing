/**
 * Healthcare Document Processing Pipeline
 * =========================================
 * Connects the visual pipeline to the REAL FastAPI backend:
 *   POST /documents/process  (multipart/form-data)
 *   GET  /documents/{id}      (confirmation retrieval)
 *
 * Component Architecture:
 * - PIPELINE_DATA         : Static stage configuration (educational content)
 * - FlowArrow             : Connector arrows between stages
 * - PipelineStage         : Stage card with embedded status badge
 * - StageDetails          : Modal with educational content (unchanged)
 * - DocumentProcessor     : Orchestrates upload → API → result display
 * - HealthcarePipeline    : Root orchestrator
 */

/* =============================================================
   API Configuration — resolved from current page origin
   ============================================================= */
const API_BASE = window.location.origin;

/* =============================================================
   Pipeline Stages Configuration Data (Educational / Static)
   ============================================================= */
const PIPELINE_DATA = [
  {
    id: "real-image",
    stageNumber: 1,
    title: "Real Image",
    subtitle: "Input image data",
    annotation: "Input",
    color: "blue",
    description: "Upload a real medical document — lab report, prescription, or discharge summary.",
    iconSvg: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
      <circle cx="8.5" cy="8.5" r="1.5"></circle>
      <polyline points="21 15 16 10 5 21"></polyline>
    </svg>`,
    isUploadStage: true,
    details: {
      headerBadge: "Stage 01 • Document Acquisition",
      summary: "Provides the original visual document to the AI pipeline without lossy downscaling or loss of textual detail.",
      fields: [
        { label: "Input", value: "Medical document image" },
        { label: "Supported Examples", value: ["Lab Report", "Prescription", "Discharge Summary"], isPills: true },
        { label: "Supported Formats", value: ["PNG", "JPG", "JPEG", "WEBP", "PDF"], isPills: true },
        { label: "Purpose", value: "Provides the original visual document to the AI pipeline for upstream classification and multimodal extraction." }
      ],
      notice: "High-resolution captures ensure handwritten prescriptions and small-font laboratory units remain legible."
    }
  },
  {
    id: "resnet",
    stageNumber: 2,
    title: "ResNet",
    subtitle: "Feature extraction",
    annotation: "CNN backbone",
    color: "teal",
    description: "ResNet-18 extracts visual features and classifies the healthcare document type.",
    iconSvg: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="2" y="2" width="6" height="6" rx="1"></rect>
      <rect x="16" y="2" width="6" height="6" rx="1"></rect>
      <rect x="9" y="9" width="6" height="6" rx="1"></rect>
      <rect x="2" y="16" width="6" height="6" rx="1"></rect>
      <rect x="16" y="16" width="6" height="6" rx="1"></rect>
      <line x1="8" y1="5" x2="16" y2="5"></line>
      <line x1="5" y1="8" x2="12" y2="9"></line>
      <line x1="19" y1="8" x2="12" y2="9"></line>
      <line x1="12" y1="15" x2="5" y2="16"></line>
      <line x1="12" y1="15" x2="19" y2="16"></line>
    </svg>`,
    details: {
      headerBadge: "Stage 02 • Deep Visual Classifier",
      summary: "A fine-tuned ResNet-18 convolutional backbone evaluates deep visual morphology to route documents through specialized pipelines.",
      fields: [
        { label: "Model", value: "ResNet-18 (PyTorch fine-tuned checkpoint)" },
        { label: "Role", value: "Image feature extraction and document classification" },
        { label: "Classes", value: ["Lab Report", "Prescription", "Discharge Summary"], isPills: true },
        { label: "Output", value: "Document type + classification confidence" }
      ],
      notice: "Confidence scores are computed dynamically in real-time per document inference. No fabricated confidence values are displayed."
    }
  },
  {
    id: "qwen",
    stageNumber: 3,
    title: "Qwen2.5-VL",
    subtitle: "Vision-language understanding",
    annotation: "LLM reasoning",
    color: "coral",
    description: "Qwen2.5-VL analyzes the document image and extracts meaningful information dynamically.",
    iconSvg: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 2a8 8 0 0 0-8 8c0 3.37 2.08 6.25 5 7.42V20a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2v-2.58c2.92-1.17 5-4.05 5-7.42a8 8 0 0 0-8-8z"></path>
      <path d="M9.5 9h.01"></path>
      <path d="M14.5 9h.01"></path>
      <path d="M10 13a4 4 0 0 0 4 0"></path>
    </svg>`,
    details: {
      headerBadge: "Stage 03 • Multimodal LLM Reasoning",
      summary: "Local Vision-Language Foundation Model parsing unconstrained layouts, tables, and doctor handwriting into contextual entities.",
      fields: [
        { label: "Model", value: "Qwen2.5-VL 3B" },
        { label: "Runtime", value: "Ollama / Local Transformers engine" },
        { label: "Role", value: "Vision-language understanding" },
        { label: "Input", value: "Original medical document image + classification context" },
        { label: "Task", value: "Extract meaningful visible information" },
        { label: "Output", value: "Dynamic structured JSON" }
      ],
      importantBehaviors: [
        "No fixed medical schema — adapts dynamically to any report format",
        "Strict grounding: Never invent or hallucinate missing information",
        "Preserve original visible values, clinical units, and labels verbatim",
        "Use null or [UNCLEAR] when text cannot be reliably read"
      ],
      clinicalDisclaimer: "Clinical Disclaimer: Qwen2.5-VL is utilized strictly for document parsing and structural information extraction. It does not diagnose patients or provide medical recommendations."
    }
  },
  {
    id: "dynamic-json",
    stageNumber: 4,
    title: "Correct Dynamic JSON",
    subtitle: "Structured output",
    annotation: "Output",
    color: "green",
    description: "Validated structured information extracted from the original medical document.",
    iconSvg: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="16 18 22 12 16 6"></polyline>
      <polyline points="8 6 2 12 8 18"></polyline>
      <line x1="12" y1="2" x2="12" y2="22"></line>
    </svg>`,
    details: {
      headerBadge: "Stage 04 • Validated Schema Output",
      summary: "Post-processed, normalized, and schema-audited dynamic payload ready for EHR integration, patient timelines, and clinical search.",
      jsonSample: {
        "document_type": "Lab_Report",
        "patient_information": {
          "name": "...",
          "age": "..."
        },
        "laboratory_results": [
          {
            "test_name": "...",
            "result": "...",
            "unit": "...",
            "reference_range": "..."
          }
        ]
      },
      notice: "Illustrative Structure: This example illustrates key value structures. The actual JSON is generated dynamically according to the specific document layout. The extraction system does NOT hardcode this schema."
    }
  }
];

/* =============================================================
   FlowArrow Component
   ============================================================= */
class FlowArrow {
  static render(fromStage, toStage) {
    return `
      <div class="pipeline-arrow-wrapper" aria-hidden="true">
        <div class="pipeline-arrow-track">
          <div class="arrow-line"></div>
          <div class="arrow-pulse"></div>
          <div class="arrow-head">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <polyline points="19 12 12 19 5 12"></polyline>
            </svg>
          </div>
        </div>
        <span class="sr-only">Flows into ${toStage.title}</span>
      </div>
    `;
  }
}

/* =============================================================
   PipelineStage Component
   Renders the stage card with a runtime status badge slot.
   ============================================================= */
class PipelineStage {
  static render(stage) {
    const uploadHint = stage.isUploadStage
      ? `<div class="stage-upload-hint" id="upload-hint-${stage.id}">
           <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
           Click to upload document
         </div>`
      : "";

    return `
      <div class="pipeline-stage-row stage-color-${stage.color}" data-stage-id="${stage.id}">
        <!-- Desktop Dotted Connector Line & Side Annotation -->
        <div class="stage-annotation-container" aria-label="Pipeline classification: ${stage.annotation}">
          <div class="annotation-badge">
            <span class="annotation-indicator"></span>
            <span class="annotation-text">${stage.annotation}</span>
          </div>
          <div class="annotation-connector" aria-hidden="true"></div>
        </div>

        <!-- Main Stage Card -->
        <button
          type="button"
          class="stage-card${stage.isUploadStage ? ' stage-card-upload' : ''}"
          data-stage-id="${stage.id}"
          aria-haspopup="dialog"
          aria-expanded="false"
          aria-controls="stage-modal"
          id="stage-card-${stage.id}"
          title="${stage.isUploadStage ? 'Click to upload a medical document' : 'Click to view details for ' + stage.title}"
        >
          <div class="stage-card-accent-bar" aria-hidden="true"></div>

          <div class="stage-card-header">
            <div class="stage-icon-pill" aria-hidden="true">
              ${stage.iconSvg}
            </div>
            <div class="stage-header-right">
              <div class="stage-step-pill">Stage 0${stage.stageNumber}</div>
              <div class="stage-status-badge" id="status-badge-${stage.id}" data-status="idle" aria-live="polite"></div>
            </div>
          </div>

          <div class="stage-card-body">
            <h3 class="stage-title">${stage.title}</h3>
            <p class="stage-subtitle">${stage.subtitle}</p>
            <p class="stage-description">${stage.description}</p>
          </div>

          <!-- Mobile Annotation -->
          <div class="stage-annotation-mobile" aria-hidden="true">
            <span class="annotation-badge-mobile">
              <span class="annotation-dot"></span>
              Role: <strong>${stage.annotation}</strong>
            </span>
          </div>

          ${uploadHint}

          <div class="stage-card-footer">
            <span class="view-details-cta">
              <span>${stage.isUploadStage ? 'Upload & process' : 'View details'}</span>
              <span class="cta-arrow" aria-hidden="true">→</span>
            </span>
          </div>
        </button>
      </div>
    `;
  }
}

/* =============================================================
   StageDetails Modal — Educational content (unchanged behavior)
   ============================================================= */
class StageDetails {
  constructor(container) {
    this.container = container;
    this.modalEl = null;
    this.activeTrigger = null;
    this.handleKeyDown = this.handleKeyDown.bind(this);
  }

  init() {
    if (!document.getElementById("stage-modal")) {
      const modalMarkup = `
        <div id="stage-modal" class="stage-modal-backdrop" aria-hidden="true" role="dialog" aria-modal="true" aria-labelledby="modal-main-title-text">
          <div class="stage-modal-dialog" role="document">
            <div class="stage-modal-header">
              <div class="modal-badge-row">
                <span id="modal-header-badge" class="modal-badge">Stage Info</span>
                <span id="modal-stage-color-pill" class="modal-color-pill"></span>
              </div>
              <button type="button" class="modal-close-btn" aria-label="Close stage details" id="modal-close-btn">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>

            <div class="stage-modal-body" id="modal-content">
              <!-- Dynamically populated -->
            </div>

            <div class="stage-modal-footer">
              <button type="button" class="modal-dismiss-btn" id="modal-dismiss-btn">Close Details</button>
            </div>
          </div>
        </div>
      `;
      document.body.insertAdjacentHTML("beforeend", modalMarkup);
    }

    this.modalEl = document.getElementById("stage-modal");

    this.modalEl.addEventListener("click", (e) => {
      if (e.target === this.modalEl) this.close();
    });
    document.getElementById("modal-close-btn").addEventListener("click", () => this.close());
    document.getElementById("modal-dismiss-btn").addEventListener("click", () => this.close());
  }

  open(stageId, triggerEl) {
    const stage = PIPELINE_DATA.find((s) => s.id === stageId);
    if (!stage) return;

    this.activeTrigger = triggerEl;
    if (this.activeTrigger) this.activeTrigger.setAttribute("aria-expanded", "true");

    const badgeEl = document.getElementById("modal-header-badge");
    const colorPill = document.getElementById("modal-stage-color-pill");
    const contentEl = document.getElementById("modal-content");

    badgeEl.textContent = stage.details.headerBadge || `Stage 0${stage.stageNumber}`;
    colorPill.className = `modal-color-pill pill-${stage.color}`;
    colorPill.textContent = stage.annotation;

    let bodyHtml = `
      <div class="modal-title-section stage-theme-${stage.color}">
        <div class="modal-icon-header">
          <div class="modal-icon-pill">${stage.iconSvg}</div>
          <div>
            <h2 class="modal-main-title" id="modal-main-title-text">${stage.title}</h2>
            <p class="modal-sub-title">${stage.subtitle}</p>
          </div>
        </div>
        <p class="modal-summary-text">${stage.details.summary}</p>
      </div>
    `;

    /* ---- Stage 1: Upload Panel (Real upload form inside modal) ---- */
    if (stage.isUploadStage) {
      bodyHtml += `<div class="modal-upload-panel" id="modal-upload-panel">
        <div class="upload-section-label">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
          Process a Real Document
        </div>
        <div class="upload-drop-zone" id="modal-drop-zone" tabindex="0" role="button" aria-label="Click to select a medical document image">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
          <p class="drop-zone-text">Drop your medical document here<br><span class="drop-zone-sub">or click to browse</span></p>
          <div class="drop-zone-formats">PNG · JPG · JPEG · WEBP</div>
          <input type="file" id="modal-file-input" accept=".png,.jpg,.jpeg,.webp,.pdf" aria-label="Select medical document file" style="display:none">
        </div>
        <div class="upload-file-selected" id="upload-file-selected" style="display:none">
          <div class="file-chip">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
            <span id="upload-filename"></span>
            <button type="button" class="file-chip-remove" id="file-remove-btn" aria-label="Remove selected file">✕</button>
          </div>
        </div>
        <div class="upload-error-msg" id="upload-error-msg" role="alert" style="display:none"></div>
        <button type="button" class="process-document-btn" id="process-document-btn" disabled>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          Process Document
        </button>
      </div>`;
    }

    /* ---- Standard fields ---- */
    if (stage.details.fields) {
      bodyHtml += `<div class="modal-details-grid">`;
      stage.details.fields.forEach((field) => {
        bodyHtml += `
          <div class="detail-item">
            <span class="detail-label">${field.label}</span>
            <div class="detail-value">
              ${
                field.isPills && Array.isArray(field.value)
                  ? field.value.map((v) => `<span class="value-chip">${v}</span>`).join("")
                  : `<strong>${field.value}</strong>`
              }
            </div>
          </div>
        `;
      });
      bodyHtml += `</div>`;
    }

    /* ---- Stage 3: Key Behaviors ---- */
    if (stage.details.importantBehaviors) {
      bodyHtml += `
        <div class="modal-section-card behavior-card">
          <h4 class="section-heading">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            Key Extraction Guardrails &amp; Behavior
          </h4>
          <ul class="behavior-checklist">
            ${stage.details.importantBehaviors.map((item) => `
              <li>
                <span class="check-icon">✓</span>
                <span>${item}</span>
              </li>
            `).join("")}
          </ul>
        </div>
      `;
    }

    /* ---- Stage 3: Clinical Disclaimer ---- */
    if (stage.details.clinicalDisclaimer) {
      bodyHtml += `
        <div class="modal-alert-box alert-warning">
          <div class="alert-icon">⚠️</div>
          <div class="alert-content">
            <strong>Medical Scope Boundary:</strong>
            <p>${stage.details.clinicalDisclaimer}</p>
          </div>
        </div>
      `;
    }

    /* ---- Stage 4: JSON Schema Sample ---- */
    if (stage.details.jsonSample) {
      const formattedJson = JSON.stringify(stage.details.jsonSample, null, 2);
      bodyHtml += `
        <div class="modal-section-card json-preview-card">
          <div class="json-header-bar">
            <span class="code-badge">Dynamic Extracted JSON Output</span>
            <span class="code-status">Schema Validated</span>
          </div>
          <pre class="json-code-block"><code>${this.syntaxHighlight(formattedJson)}</code></pre>
        </div>
      `;
    }

    /* ---- Informational Notice ---- */
    if (stage.details.notice) {
      bodyHtml += `
        <div class="modal-alert-box alert-info">
          <div class="alert-icon">ℹ️</div>
          <div class="alert-content">
            <p>${stage.details.notice}</p>
          </div>
        </div>
      `;
    }

    contentEl.innerHTML = bodyHtml;

    /* ---- Show modal ---- */
    this.modalEl.classList.add("is-open");
    this.modalEl.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-scroll-lock");
    document.addEventListener("keydown", this.handleKeyDown);

    setTimeout(() => {
      document.getElementById("modal-close-btn").focus();
    }, 50);

    /* ---- Bind upload interactions if this is stage 1 ---- */
    if (stage.isUploadStage) {
      this._bindUploadEvents();
    }
  }

  _bindUploadEvents() {
    const dropZone = document.getElementById("modal-drop-zone");
    const fileInput = document.getElementById("modal-file-input");
    const fileSelectedRow = document.getElementById("upload-file-selected");
    const filenameSpan = document.getElementById("upload-filename");
    const removeBtn = document.getElementById("file-remove-btn");
    const processBtn = document.getElementById("process-document-btn");
    const errorMsg = document.getElementById("upload-error-msg");

    if (!dropZone || !fileInput) return;

    const ALLOWED_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".pdf"];
    let selectedFile = null;

    const setFile = (file) => {
      const ext = "." + file.name.split(".").pop().toLowerCase();
      if (!ALLOWED_EXTS.includes(ext)) {
        errorMsg.textContent = `Unsupported format '${ext}'. Please use PNG, JPG, JPEG, or WEBP.`;
        errorMsg.style.display = "block";
        return;
      }
      errorMsg.style.display = "none";
      selectedFile = file;
      filenameSpan.textContent = file.name;
      fileSelectedRow.style.display = "flex";
      processBtn.disabled = false;
    };

    const clearFile = () => {
      selectedFile = null;
      fileInput.value = "";
      fileSelectedRow.style.display = "none";
      processBtn.disabled = true;
      errorMsg.style.display = "none";
    };

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      const files = e.dataTransfer.files;
      if (files && files[0]) setFile(files[0]);
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
    });

    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearFile();
    });

    processBtn.addEventListener("click", () => {
      if (!selectedFile) return;
      this.close();
      // Trigger real processing via DocumentProcessor
      window._docProcessor && window._docProcessor.start(selectedFile);
    });
  }

  close() {
    if (!this.modalEl || !this.modalEl.classList.contains("is-open")) return;
    this.modalEl.classList.remove("is-open");
    this.modalEl.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-scroll-lock");
    document.removeEventListener("keydown", this.handleKeyDown);

    if (this.activeTrigger) {
      this.activeTrigger.setAttribute("aria-expanded", "false");
      this.activeTrigger.focus();
      this.activeTrigger = null;
    }
  }

  handleKeyDown(e) {
    if (e.key === "Escape") this.close();
  }

  syntaxHighlight(jsonStr) {
    const escaped = jsonStr
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    return escaped.replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
      (match) => {
        let cls = "token-number";
        if (/^"/.test(match)) {
          cls = /:$/.test(match) ? "token-key" : "token-string";
        } else if (/true|false/.test(match)) {
          cls = "token-boolean";
        } else if (/null/.test(match)) {
          cls = "token-null";
        }
        return `<span class="${cls}">${match}</span>`;
      }
    );
  }
}

/* =============================================================
   DocumentProcessor
   Orchestrates: upload → POST /documents/process → display
   ============================================================= */
class DocumentProcessor {
  constructor(pipelineInstance) {
    this.pipeline = pipelineInstance;
    this._resultData = null;
  }

  /** Entry point — called with the selected File object */
  async start(file) {
    this._clearResults();
    this._setStageStatus("real-image", "done", "Uploaded");
    this._setStageStatus("resnet", "processing", "Processing…");
    this._setStageStatus("qwen", "waiting", "Waiting");
    this._setStageStatus("dynamic-json", "waiting", "Waiting");

    // Scroll the page to show pipeline progress
    const pipelineEl = document.querySelector(".healthcare-pipeline-wrapper");
    if (pipelineEl) pipelineEl.scrollIntoView({ behavior: "smooth", block: "start" });

    let apiResult;
    try {
      apiResult = await this._callProcessAPI(file);
    } catch (err) {
      this._setStageStatus("resnet", "error", "Failed");
      this._setStageStatus("qwen", "error", "Failed");
      this._setStageStatus("dynamic-json", "error", "Failed");
      this._renderResultsPanel({
        type: "network_error",
        message: err.message || "Network error — could not reach the server."
      });
      return;
    }

    // ResNet always runs successfully if we got a response
    this._setStageStatus("resnet", "done", "Done");

    // Short visual pause so stages feel sequential
    await this._delay(300);

    this._setStageStatus("qwen", "processing", "Processing…");

    await this._delay(400);

    // Detect Ollama unavailability
    const ollamaError = (apiResult.errors || []).find(
      (e) => e.code === "ollama_connection_error" || e.code === "ollama_model_missing"
    );

    if (ollamaError) {
      this._setStageStatus("qwen", "error", "Ollama unavailable");
      await this._delay(200);
      this._setStageStatus("dynamic-json", "error", "Human Review Required");
    } else if (!apiResult.extracted_json) {
      this._setStageStatus("qwen", "error", "Extraction failed");
      await this._delay(200);
      this._setStageStatus("dynamic-json", "error", "Extraction failed");
    } else {
      this._setStageStatus("qwen", "done", "Done");
      await this._delay(200);
      this._setStageStatus("dynamic-json", "done", "Validated ✓");
    }

    // Confirm storage via GET /documents/{id}
    let confirmedDoc = null;
    if (apiResult.document_id) {
      try {
        confirmedDoc = await this._fetchDocument(apiResult.document_id);
      } catch (_) {
        // Non-fatal — we already have the process response
      }
    }

    this._renderResultsPanel({ type: "success", apiResult, confirmedDoc, ollamaError });
  }

  async _callProcessAPI(file) {
    const formData = new FormData();
    formData.append("file", file, file.name);
    formData.append("run_extraction", "true");

    const response = await fetch(`${API_BASE}/documents/process`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const errBody = await response.json();
        detail = errBody.detail || JSON.stringify(errBody);
      } catch (_) {
        detail = await response.text().catch(() => detail);
      }
      const httpErr = new Error(detail);
      httpErr.status = response.status;
      throw httpErr;
    }

    return response.json();
  }

  async _fetchDocument(docId) {
    const response = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`);
    if (!response.ok) throw new Error(`GET /documents/${docId} returned ${response.status}`);
    return response.json();
  }

  _delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /** Update a stage card's status badge */
  _setStageStatus(stageId, status, label) {
    const badge = document.getElementById(`status-badge-${stageId}`);
    if (!badge) return;
    badge.setAttribute("data-status", status);
    badge.innerHTML = this._badgeContent(status, label);
  }

  _badgeContent(status, label) {
    const icons = {
      idle: "",
      waiting: `<span class="badge-dot badge-dot-waiting"></span>`,
      processing: `<span class="badge-spinner" aria-hidden="true"></span>`,
      done: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>`,
      error: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`
    };
    const icon = icons[status] || "";
    return status === "idle" ? "" : `${icon}<span>${label}</span>`;
  }

  _clearResults() {
    const existing = document.getElementById("pipeline-results-panel");
    if (existing) existing.remove();
    // Reset all badges
    PIPELINE_DATA.forEach((s) => this._setStageStatus(s.id, "idle", ""));
  }

  /** Render the results panel below the pipeline */
  _renderResultsPanel(opts) {
    const wrapper = document.querySelector(".healthcare-pipeline-wrapper");
    if (!wrapper) return;

    let html = `<section class="pipeline-results-panel" id="pipeline-results-panel" aria-label="Processing Results">
      <div class="results-panel-header">
        <div class="results-header-left">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          <span>Pipeline Results</span>
        </div>
        <div class="results-header-badge ${this._resultsBadgeClass(opts)}">
          ${this._resultsBadgeLabel(opts)}
        </div>
      </div>`;

    if (opts.type === "network_error") {
      html += `<div class="result-error-banner">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
        <div>
          <strong>Connection Failed</strong>
          <p>${this._escapeHtml(opts.message)}</p>
        </div>
      </div>`;
    } else if (opts.type === "success") {
      const r = opts.apiResult;

      /* ---- Metadata row ---- */
      html += `<div class="result-meta-grid">`;
      if (r.document_id) {
        html += `<div class="meta-item"><span class="meta-label">Document ID</span><code class="meta-value meta-id">${this._escapeHtml(String(r.document_id))}</code></div>`;
      }
      if (r.document_type) {
        html += `<div class="meta-item"><span class="meta-label">Document Type</span><span class="meta-value meta-type">${this._escapeHtml(r.document_type)}</span></div>`;
      }
      if (r.classifier_confidence != null) {
        const pct = (r.classifier_confidence * 100).toFixed(2);
        const confClass = r.classifier_confidence >= 0.8 ? "conf-high" : r.classifier_confidence >= 0.5 ? "conf-mid" : "conf-low";
        html += `<div class="meta-item"><span class="meta-label">Classifier Confidence</span><span class="meta-value ${confClass}">${pct}%</span></div>`;
      }
      if (r.extraction_status) {
        html += `<div class="meta-item"><span class="meta-label">Extraction</span><span class="meta-value meta-status-${r.extraction_status.toLowerCase()}">${this._escapeHtml(r.extraction_status)}</span></div>`;
      }
      if (r.validation_status) {
        html += `<div class="meta-item"><span class="meta-label">Validation</span><span class="meta-value">${this._escapeHtml(r.validation_status)}</span></div>`;
      }
      if (r.processing_duration_ms != null) {
        html += `<div class="meta-item"><span class="meta-label">Processing Time</span><span class="meta-value">${Math.round(r.processing_duration_ms).toLocaleString()} ms</span></div>`;
      }
      html += `</div>`;

      /* ---- Distinct section label ---- */
      html += `<div class="result-section-divider">
        <span>Actual Processing Result</span>
        <span class="divider-note">Live data from pipeline — not fabricated</span>
      </div>`;

      /* ---- Ollama unavailable banner ---- */
      if (opts.ollamaError) {
        html += `<div class="result-ollama-banner">
          <div class="ollama-banner-header">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
            <strong>Qwen2.5-VL · Ollama Unavailable</strong>
          </div>
          <p>${this._escapeHtml(opts.ollamaError.message)}</p>
          <p class="ollama-hint">Start Ollama with <code>ollama serve</code> and ensure <code>qwen2.5vl:3b</code> is pulled.</p>
        </div>`;
      }

      /* ---- Human Review Required banner ---- */
      if (r.review_required) {
        const reasons = Array.isArray(r.review_reasons) ? r.review_reasons : [];
        html += `<div class="result-review-banner">
          <div class="review-banner-header">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            <strong>Human Review Required</strong>
          </div>
          ${reasons.length > 0 ? `<ul class="review-reasons">${reasons.map((r) => `<li>${this._escapeHtml(r)}</li>`).join("")}</ul>` : ""}
        </div>`;
      }

      if (r.extracted_json && r.extracted_json.document_type === "Lab_Report") {
        const investigations = Array.isArray(r.extracted_json.investigations) ? r.extracted_json.investigations : [];
        const rows = investigations.map((item) => `<tr><td>${this._escapeHtml(item.name ?? "")}</td><td>${this._escapeHtml(String(item.observed_value ?? ""))}</td><td>${this._escapeHtml(item.unit ?? "")}</td><td>${this._escapeHtml(item.reference_interval ?? "")}</td></tr>`).join("");
        html += `<section class="result-json-section"><h3>Extracted information · Investigations</h3><div class="table-wrap"><table><thead><tr><th>Investigation</th><th>Observed value</th><th>Unit</th><th>Reference interval</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
      } else if (r.extracted_json && r.extracted_json.document_type === "Prescription") {
        const medications = (r.extracted_json.prescription || {}).medications || [];
        const rows = medications.map((item) => `<tr><td>${this._escapeHtml(item.name ?? "")}</td><td>${this._escapeHtml(item.strength ?? "")}</td><td>${this._escapeHtml(item.dosage ?? "")}</td><td>${this._escapeHtml(item.frequency ?? "")}</td><td>${this._escapeHtml(item.instructions ?? "")}</td></tr>`).join("");
        html += `<section class="result-json-section"><h3>Extracted information · Medications</h3><div class="table-wrap"><table><thead><tr><th>Name</th><th>Strength</th><th>Dosage</th><th>Frequency</th><th>Instructions</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
      } else if (r.extracted_json && r.extracted_json.document_type === "Discharge_Summary") {
        const clinical = r.extracted_json.clinical_information || {};
        html += `<section class="result-json-section"><h3>Extracted information · Admission and clinical information</h3><p>Admission: ${this._escapeHtml((r.extracted_json.admission || {}).admission_date ?? "Not provided")} · Discharge: ${this._escapeHtml((r.extracted_json.admission || {}).discharge_date ?? "Not provided")}</p><pre>${this._escapeHtml(JSON.stringify(clinical, null, 2))}</pre></section>`;
      }

      /* ---- JSON output or failure ---- */
      if (r.extracted_json) {
        const jsonStr = JSON.stringify(r.extracted_json, null, 2);
        const highlighted = (new StageDetails()).syntaxHighlight(jsonStr);
        html += `<div class="result-json-section">
          <div class="result-json-toolbar">
            <div class="json-toolbar-left">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline><line x1="12" y1="2" x2="12" y2="22"></line></svg>
              <span>Qwen Extracted JSON</span>
              <span class="json-lines-count">${jsonStr.split("\n").length} lines</span>
            </div>
            <div class="json-toolbar-right">
              <button type="button" class="json-tool-btn" id="json-expand-btn" aria-label="Expand JSON viewer">⤢ Expand</button>
              <button type="button" class="json-tool-btn json-copy-btn" id="json-copy-btn" aria-label="Copy JSON to clipboard">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                Copy JSON
              </button>
            </div>
          </div>
          <div class="result-json-container" id="result-json-container">
            <pre class="json-code-block result-json-pre" id="result-json-pre"><code>${highlighted}</code></pre>
          </div>
        </div>`;
        this._resultData = jsonStr;
      } else if (!opts.ollamaError) {
        const reasons = Array.isArray(r.review_reasons) ? r.review_reasons.join("; ") : "No JSON returned from extraction.";
        html += `<div class="result-extraction-failed">
          <div class="extraction-failed-header">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
            Extraction Failed
          </div>
          <p>Reason: ${this._escapeHtml(reasons)}</p>
        </div>`;
      }

      /* ---- Storage confirmation ---- */
      if (opts.confirmedDoc) {
        html += `<div class="result-storage-confirm">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
          Document confirmed in database — ID: <code>${this._escapeHtml(String(opts.confirmedDoc.id || opts.confirmedDoc.document_id))}</code>
        </div>`;
      }
    }

    /* ---- Process Another Document button ---- */
    html += `<div class="result-actions">
      <button type="button" class="process-another-btn" id="process-another-btn">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 .49-3.5"></path></svg>
        Process Another Document
      </button>
    </div>`;

    html += `</section>`;

    wrapper.insertAdjacentHTML("beforeend", html);

    // Scroll results into view
    setTimeout(() => {
      const panel = document.getElementById("pipeline-results-panel");
      if (panel) panel.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);

    // Bind action buttons
    this._bindResultActions(opts);
  }

  _bindResultActions(opts) {
    // Copy JSON
    const copyBtn = document.getElementById("json-copy-btn");
    if (copyBtn && this._resultData) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(this._resultData).then(() => {
          copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!`;
          setTimeout(() => {
            copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy JSON`;
          }, 2000);
        }).catch(() => {
          // Fallback for environments without clipboard API
          const ta = document.createElement("textarea");
          ta.value = this._resultData;
          ta.style.cssText = "position:absolute;left:-9999px";
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
          copyBtn.textContent = "Copied!";
          setTimeout(() => { copyBtn.textContent = "Copy JSON"; }, 2000);
        });
      });
    }

    // Expand/collapse JSON
    const expandBtn = document.getElementById("json-expand-btn");
    const jsonContainer = document.getElementById("result-json-container");
    if (expandBtn && jsonContainer) {
      let expanded = false;
      expandBtn.addEventListener("click", () => {
        expanded = !expanded;
        jsonContainer.classList.toggle("json-expanded", expanded);
        expandBtn.textContent = expanded ? "⤡ Collapse" : "⤢ Expand";
      });
    }

    // Process Another
    const anotherBtn = document.getElementById("process-another-btn");
    if (anotherBtn) {
      anotherBtn.addEventListener("click", () => {
        this._clearResults();
        // Open stage 1 upload modal
        const card = document.getElementById("stage-card-real-image");
        if (card) {
          this.pipeline.stageDetails.open("real-image", card);
        }
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    }
  }

  _resultsBadgeClass(opts) {
    if (opts.type === "network_error") return "results-badge-error";
    if (!opts.apiResult) return "results-badge-error";
    const r = opts.apiResult;
    if (opts.ollamaError || r.review_required) return "results-badge-review";
    return "results-badge-success";
  }

  _resultsBadgeLabel(opts) {
    if (opts.type === "network_error") return "Connection Error";
    if (!opts.apiResult) return "Error";
    const r = opts.apiResult;
    if (opts.ollamaError) return "Ollama Unavailable";
    if (r.review_required) return "Human Review Required";
    return "Processing Complete";
  }

  _escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
}

/* =============================================================
   HealthcarePipeline Orchestrator
   ============================================================= */
class HealthcarePipeline {
  constructor(containerId = "healthcare-pipeline-container") {
    this.container = document.getElementById(containerId);
    this.stageDetails = new StageDetails(this.container);
    this.docProcessor = new DocumentProcessor(this);
    // Expose processor globally so upload modal can call it
    window._docProcessor = this.docProcessor;
  }

  init() {
    if (!this.container) {
      console.error("HealthcarePipeline: target container not found");
      return;
    }

    this.render();
    this.stageDetails.init();
    this.bindEvents();
    this.setupIntersectionObserver();
  }

  render() {
    let html = `
      <div class="healthcare-pipeline" role="region" aria-label="Healthcare Document Processing Pipeline Flowchart">
    `;

    PIPELINE_DATA.forEach((stage, index) => {
      html += PipelineStage.render(stage);
      if (index < PIPELINE_DATA.length - 1) {
        html += FlowArrow.render(stage, PIPELINE_DATA[index + 1]);
      }
    });

    html += `</div>`;
    this.container.innerHTML = html;
  }

  bindEvents() {
    this.container.addEventListener("click", (e) => {
      const cardBtn = e.target.closest(".stage-card");
      if (cardBtn) {
        const stageId = cardBtn.getAttribute("data-stage-id");
        this.stageDetails.open(stageId, cardBtn);
      }
    });
  }

  setupIntersectionObserver() {
    if (!("IntersectionObserver" in window)) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) entry.target.classList.add("in-view");
        });
      },
      { threshold: 0.2 }
    );

    const stages = this.container.querySelectorAll(".pipeline-stage-row");
    const arrows = this.container.querySelectorAll(".pipeline-arrow-wrapper");
    stages.forEach((el) => observer.observe(el));
    arrows.forEach((el) => observer.observe(el));
  }
}

/* =============================================================
   Module exports
   ============================================================= */
if (typeof window !== "undefined") {
  window.HealthcarePipeline = HealthcarePipeline;
  window.PIPELINE_DATA = PIPELINE_DATA;
}
