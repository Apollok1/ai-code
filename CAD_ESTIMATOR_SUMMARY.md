# CAD Estimator Pro - Analiza "Przenieść vs Przepisać"

## 1. Drzewo Projektu (kluczowe ścieżki)

```
ai-code/
├── src/
│   ├── cad/                              # <── CAD ESTIMATOR (główny moduł)
│   │   ├── application/
│   │   │   ├── estimation_pipeline.py    # Główna orkiestracja estymacji
│   │   │   └── batch_importer.py         # Import historycznych danych
│   │   ├── domain/
│   │   │   ├── models/                   # Modele domenowe
│   │   │   │   ├── component.py          # Component, ComponentPattern
│   │   │   │   ├── estimate.py           # Estimate, EstimatePhases, Risk
│   │   │   │   ├── multi_model.py        # StageContext, PipelineProgress
│   │   │   │   └── config.py             # AppConfig, OllamaConfig
│   │   │   ├── interfaces/               # Protokoły/abstrakcje
│   │   │   │   ├── ai_client.py
│   │   │   │   ├── database.py
│   │   │   │   └── multi_model.py
│   │   │   └── exceptions.py
│   │   ├── infrastructure/
│   │   │   ├── ai/
│   │   │   │   └── ollama_client.py      # Klient Ollama (text, vision, embeddings)
│   │   │   ├── multi_model/              # 4-etapowy pipeline Multi-Model
│   │   │   │   ├── orchestrator.py       # Koordynator etapów
│   │   │   │   ├── stage1_technical_analysis.py
│   │   │   │   ├── stage2_structural_decomposition.py
│   │   │   │   ├── stage3_hours_estimation.py
│   │   │   │   └── stage4_risk_optimization.py
│   │   │   ├── parsers/
│   │   │   │   ├── excel_parser.py       # Parser Exceli (komponenty)
│   │   │   │   ├── pdf_parser.py         # Parser PDF (specyfikacje)
│   │   │   │   └── component_parser.py
│   │   │   ├── learning/
│   │   │   │   ├── pattern_learner.py    # Uczenie wzorców komponentów
│   │   │   │   └── bundle_learner.py     # Uczenie relacji komponent→sub
│   │   │   ├── embeddings/
│   │   │   │   └── pgvector_service.py   # Vector search (podobne projekty)
│   │   │   ├── database/
│   │   │   │   └── postgres_client.py    # PostgreSQL + pgvector
│   │   │   └── factory.py                # Dependency injection
│   │   └── presentation/
│   │       ├── app.py                    # <── GŁÓWNY PLIK STREAMLIT
│   │       ├── components/               # UI komponenty
│   │       │   ├── file_uploader.py
│   │       │   ├── results_display.py
│   │       │   ├── multi_model_results.py
│   │       │   ├── progress_tracker.py
│   │       │   ├── sidebar.py
│   │       │   ├── learning.py
│   │       │   ├── pattern_analysis.py
│   │       │   └── project_history.py
│   │       └── state/
│   │           └── session_manager.py
│   │
│   └── (doc-converter modules)           # Osobna aplikacja - konwerter dokumentów
│       ├── infrastructure/
│       │   ├── ocr/tesseract_ocr.py      # OCR (Tesseract)
│       │   ├── audio/whisper_client.py   # ASR (Whisper)
│       │   └── audio/pyannote_client.py  # Speaker diarization
│       └── ...
│
├── doc-converter/
│   └── app/converter.py                  # Monolityczny konwerter (~1000+ LOC)
│
├── whisper-rocm/server.py                # Serwer Whisper (AMD ROCm)
├── pyannote/server.py                    # Serwer Pyannote
└── tests/
    ├── unit/
    └── integration/
```

---

## 2. Główny Plik Streamlit: `src/cad/presentation/app.py`

### Struktura UI:
```
📋 Menu (sidebar radio):
├── 📊 Dashboard          → Statystyki projektów/wzorców
├── 🆕 Nowy projekt       → Główny flow estymacji
├── 📚 Historia i Uczenie → Historia, feedback, wzorce, bundles, export
└── 🛠️ Admin             → Czyszczenie danych, przeliczanie embeddingów
```

### Kluczowe funkcje:
| Funkcja | Linia | Opis |
|---------|-------|------|
| `init_app()` | 44 | Inicjalizacja DI: DB, AI, parsery, pipeline |
| `main()` | 179 | Entry point + routing |
| `render_new_project_page()` | 305 | **Flow estymacji** - upload + analiza AI |
| `is_description_poor()` | 121, 283 | Walidacja jakości opisu |

### Flow "Nowy projekt":
1. User wpisuje opis + upload PDF/Excel
2. **Pre-check** (opcjonalnie) → `pipeline.precheck_requirements()` → Project Brain
3. **Analiza AI** → `pipeline.estimate_from_description()`
   - Single-model LUB Multi-model (4 etapy)
4. Wyświetlenie wyników + lista komponentów

---

## 3. Moduły "które robią pracę"

### 3.1 Ollama Client (`src/cad/infrastructure/ai/ollama_client.py`)

```python
class OllamaClient:
    """Implementuje: AIClient, VisionAIClient, EmbeddingClient"""

    def generate_text(prompt, model, json_mode, timeout) -> str
        # POST /api/generate

    def analyze_image(prompt, images_base64, model) -> str
        # POST /api/generate z images[]

    def generate_embedding(text, model) -> list[float]
        # POST /api/embeddings

    def list_available_models() -> list[str]
        # GET /api/tags (cached 5min)
```

**Konfiguracja** (z `.env` lub `AppConfig`):
- `OLLAMA_URL` = `http://127.0.0.1:11434`
- `text_model` = np. `llama3:latest`
- `vision_model` = np. `llava:latest`
- `embed_model` = np. `nomic-embed-text:latest`

---

### 3.2 Multi-Model Pipeline (`src/cad/infrastructure/multi_model/`)

4-etapowy pipeline estymacji:

| Etap | Plik | Model (konfigurowalny) | Output |
|------|------|------------------------|--------|
| **Stage 1** | `stage1_technical_analysis.py` | `deepseek-coder` | `TechnicalAnalysis`: complexity, materials, standards, challenges |
| **Stage 2** | `stage2_structural_decomposition.py` | `llama3` | `StructuralDecomposition`: root_components, total_count, max_depth |
| **Stage 3** | `stage3_hours_estimation.py` | `llama3` | `estimated_components[]` z godzinami (layout, detail, doc) |
| **Stage 4** | `stage4_risk_optimization.py` | `llama3` | `risks[], suggestions[], assumptions[], warnings[]` |

**Orchestrator** (`orchestrator.py`):
```python
def execute_pipeline(context, stage1_model, stage2_model, ...) -> Estimate:
    # 1. TechnicalAnalysisStage.analyze()
    # 2. StructuralDecompositionStage.decompose()
    # 3. HoursEstimationStage.estimate_hours()
    # 4. RiskOptimizationStage.analyze_risks()
    # → _build_estimate()
```

---

### 3.3 OCR (`src/infrastructure/ocr/tesseract_ocr.py`)

```python
class TesseractOCR:
    def extract_text(image_bytes, language="pol+eng", preprocess=True) -> str:
        # 1. PIL.Image.open()
        # 2. Adaptive preprocessing (Otsu thresholding dla niskiej jakości)
        # 3. pytesseract.image_to_string()
```

**Zależności**: `pytesseract`, `opencv-python`, `Pillow`, `numpy`

---

### 3.4 Audio/Whisper (`src/infrastructure/audio/whisper_client.py`)

```python
class WhisperASRClient:
    def __init__(base_url="http://localhost:9000")

    def transcribe(audio_bytes, language=None, timeout=300) -> list[AudioSegment]:
        # POST /asr z files={"audio_file": ...}
        # Zwraca: [AudioSegment(start, end, text), ...]
```

**Serwer Whisper** (`whisper-rocm/server.py`) - osobny proces.

---

### 3.5 Doc-Converter (`doc-converter/app/converter.py`)

Monolityczna aplikacja Streamlit (~1000+ LOC) do konwersji dokumentów:
- PDF → text (pdfplumber + OCR fallback)
- DOCX, PPTX → text
- Audio → transkrypcja (Whisper + Pyannote diarization)
- Obrazy → OCR/Vision
- Email (.eml, .msg) → text

**Konfiguracja** (env vars):
```
OLLAMA_URL=http://127.0.0.1:11434
WHISPER_URL=http://127.0.0.1:9000
PYANNOTE_URL=http://127.0.0.1:8000
```

---

## 4. Przykładowy Flow: Upload → Wynik

```
┌─────────────────────────────────────────────────────────────────────┐
│ USER: Wpisuje opis projektu + upload PDF/Excel                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ app.py:render_new_project_page()                                    │
│ ├── render_file_uploader() → files["pdfs"], files["excel"]         │
│ └── render_text_input() → description                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ "🤖 Analizuj z AI" button
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ EstimationPipeline.estimate_from_description()                      │
│ ├── excel_parser.parse() → components[]                             │
│ ├── pdf_parser.extract_text() → pdf_texts[]                        │
│ ├── pgvector.find_similar_projects() → similar_projects[]          │
│ └── [ROUTING]                                                       │
│     ├── use_multi_model=True → _estimate_multi_model()             │
│     └── use_multi_model=False → _estimate_single_model()           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
┌───────────────────────┐               ┌───────────────────────┐
│ SINGLE-MODEL          │               │ MULTI-MODEL (4 stages)│
│ 1. Build prompt       │               │ 1. TechnicalAnalysis  │
│ 2. ai.generate_text() │               │ 2. Decomposition      │
│ 3. Parse JSON         │               │ 3. HoursEstimation    │
│ 4. Enrich patterns    │               │ 4. RiskOptimization   │
│ 5. Scale if too low   │               │ 5. Build Estimate     │
└───────────┬───────────┘               └───────────┬───────────┘
            │                                       │
            └─────────────────┬─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Estimate object:                                                    │
│ ├── components: [Component(name, hours_3d_layout, hours_3d_detail,  │
│ │                          hours_2d, confidence, ...)]              │
│ ├── phases: EstimatePhases(layout, detail, documentation)          │
│ ├── risks: [Risk(description, impact, mitigation)]                 │
│ ├── overall_confidence: float                                       │
│ └── generation_metadata: {multi_model, similar_projects, ...}      │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ UI: render_multi_model_results(estimate, hourly_rate)               │
│ ├── Podsumowanie godzin (layout/detail/doc)                         │
│ ├── Wykres Gantt / breakdown                                        │
│ ├── Ryzyka i sugestie                                               │
│ └── Lista komponentów (render_components_list)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Ocena: Przenieść vs Przepisać

### Mocne strony (PRZENIEŚĆ):
- **Architektura DDD** - czyste separation of concerns (domain/infrastructure/presentation)
- **Multi-model pipeline** - dobrze przemyślany 4-etapowy flow z walidacją
- **Pattern learning** - mechanizm uczenia z historycznych danych
- **Vector search** (pgvector) - semantyczne dopasowanie podobnych projektów
- **Blending strategies** - inteligentne łączenie wzorców z AI

### Słabe strony (do refaktoru):
- **Duplikacja `is_description_poor()`** - dwie identyczne funkcje w tym samym pliku
- **Doc-converter jako monolit** - 1000+ LOC w jednym pliku
- **Hardcoded credentials** - `password == "polmic"` w admin
- **Brak testów** - katalog tests/ prawie pusty
- **Mieszanie języków** - PL/EN w promptach i komentarzach

### Rekomendacja:

| Moduł | Decyzja | Uzasadnienie |
|-------|---------|--------------|
| `src/cad/` | **PRZENIEŚĆ** | Dobra architektura, warto zachować |
| `multi_model/` | **PRZENIEŚĆ** | Kluczowa logika, dobrze zaprojektowana |
| `ollama_client.py` | **PRZENIEŚĆ** | Prosty, dobrze działa |
| `doc-converter/` | **PRZEPISAĆ** | Monolit, trudny do utrzymania |
| `presentation/app.py` | **REFAKTOR** | Za duży, duplikacje, wydzielić pages |

---

## 6. Sugerowane etapy migracji

1. **Testy jednostkowe** dla `domain/models/` (Component, Estimate)
2. **Refaktor `app.py`** - wydzielić pages do osobnych plików
3. **Usunąć duplikaty** (is_description_poor)
4. **Config z env/secrets** zamiast hardcoded
5. **Przepisać doc-converter** na modułową architekturę
6. **Dodać typing** tam gdzie brakuje
7. **CI/CD** z testami

