# CAD Estimator Pro - Analiza "Przenieść vs Przepisać"

> **WERDYKT: 90% PRZENIEŚĆ + 10% REFAKTOR**
>
> Kod jest dobrej jakości (DDD, Clean Architecture). Nie wymaga przepisywania od zera.
> Wymaga tylko kilku refaktorów i poprawek bezpieczeństwa.

---

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

## 6. KONKRETNA DECYZJA: Co z czym?

### ✅ PRZENIEŚĆ (bez zmian lub minimalne poprawki)

| Plik/Moduł | Stan | Dlaczego OK |
|------------|------|-------------|
| `src/cad/domain/models/` | ✅ Gotowe | Czyste dataclassy, immutable, dobrze typowane |
| `src/cad/domain/interfaces/` | ✅ Gotowe | Protocol-based, łatwo podmienić implementacje |
| `src/cad/infrastructure/multi_model/` | ✅ Gotowe | 4-etapowy pipeline, walidacje, logging |
| `src/cad/infrastructure/ai/ollama_client.py` | ✅ Gotowe | Prosty, działa, cache modeli |
| `src/cad/infrastructure/parsers/` | ✅ Gotowe | Excel/PDF parsery, error handling |
| `src/cad/infrastructure/learning/` | ✅ Gotowe | Pattern/Bundle learner z blending |
| `src/cad/infrastructure/embeddings/` | ✅ Gotowe | pgvector search działa |
| `src/cad/application/estimation_pipeline.py` | ✅ Gotowe | Główna orkiestracja, multi-strategy matching |

### ⚠️ REFAKTOR (poprawki, nie przepisywanie)

| Plik | Problem | Rozwiązanie |
|------|---------|-------------|
| `src/cad/presentation/app.py` | 2x `is_description_poor()` (linie 121-176 i 283-304) | Usunąć duplikat (linia 283-304) |
| `src/cad/presentation/app.py` | Hardcoded `password == "polmic"` (linia 753) | Przenieść do env: `ADMIN_PASSWORD` |
| `src/cad/presentation/app.py` | 816 linii, wszystkie pages w jednym pliku | Wydzielić do `pages/dashboard.py`, `pages/new_project.py`, etc. |
| `docker-compose.yml` | Hasło DB jawne w pliku | Przenieść do `.env` lub secrets |

### ❌ NIE PRZENOSIĆ (jeśli nie jest potrzebne)

| Moduł | Dlaczego |
|-------|----------|
| `doc-converter/app/converter.py` | Monolit 1000+ LOC, osobna aplikacja, ma już refaktorowaną wersję w `src/` |
| `whisper-rocm/`, `pyannote/` | Zewnętrzne serwery, nie są częścią CAD Estimator |

---

## 7. ZADANIA DO WYKONANIA (w kolejności)

### Faza 1: Krytyczne poprawki (1-2h)

```
[ ] 1. Usunąć duplikat is_description_poor() z app.py (linie 283-304)
[ ] 2. Przenieść hasło admina do env:
      - app.py linia 753: password == os.getenv("CAD_ADMIN_PASSWORD", "change_me")
      - docker-compose: dodać CAD_ADMIN_PASSWORD do environment
[ ] 3. Przenieść hasła DB do .env (już są częściowo, sprawdzić)
```

### Faza 2: Refaktor app.py (2-4h)

```
[ ] 4. Wydzielić pages do osobnych plików:
      src/cad/presentation/
      ├── app.py              # tylko routing + init_app()
      ├── pages/
      │   ├── __init__.py
      │   ├── dashboard.py    # render_dashboard_page()
      │   ├── new_project.py  # render_new_project_page()
      │   ├── history.py      # render_history_page()
      │   └── admin.py        # render_admin_page()
      └── utils/
          └── validators.py   # is_description_poor()
```

### Faza 3: Testy (4-8h)

```
[ ] 5. Testy jednostkowe dla domain/models/:
      - test_component.py
      - test_estimate.py
      - test_multi_model.py
[ ] 6. Testy integracyjne dla pipeline:
      - test_estimation_pipeline.py (mock AI)
      - test_multi_model_orchestrator.py
[ ] 7. Testy dla parsers:
      - test_excel_parser.py
      - test_pdf_parser.py
```

### Faza 4: Opcjonalne ulepszenia (4-8h)

```
[ ] 8. Dodać mypy strict mode (pyproject.toml ma już konfigurację)
[ ] 9. CI/CD pipeline (GitHub Actions):
      - lint (ruff)
      - type check (mypy)
      - tests (pytest)
[ ] 10. Dokumentacja API (docstringi są, ale można dodać mkdocs)
```

---

## 8. Zależności (requirements)

### CAD Estimator (`cad/requirements.txt`)
```
streamlit>=1.28.0       # UI framework
psycopg2-binary>=2.9.9  # PostgreSQL + pgvector
pandas>=2.1.0           # Data processing
numpy>=1.24.0           # Numerics
PyPDF2>=3.0.0           # PDF parsing
openpyxl>=3.1.0         # Excel parsing
Pillow>=10.0.0          # Image processing
rapidfuzz>=3.5.0        # Fuzzy string matching
requests>=2.31.0        # HTTP (Ollama API)
plotly>=5.18.0          # Charts
pydantic>=2.5.0         # Config validation
pydantic-settings>=2.1.0
```

### Brakujące (do dodania jeśli potrzebne)
```
pytest>=7.4.0           # Testing
pytest-cov>=4.1.0       # Coverage
black>=23.7.0           # Formatting
ruff>=0.0.286           # Linting
mypy>=1.5.0             # Type checking
```

---

## 9. Docker Stack

```yaml
# Serwisy dla CAD Estimator Pro:
ollama:         # LLM backend (AMD ROCm)
  - port: 11434
  - models: llama3, deepseek-coder, nomic-embed-text

cad-postgres:   # PostgreSQL + pgvector
  - port: 5432
  - db: cad_estimator

cad-panel:      # Streamlit UI
  - port: 8501
  - mounts: src/cad → /app/src

# Opcjonalne (dla doc-converter):
whisper:        # ASR
pyannote:       # Speaker diarization
doc-converter:  # Document processing UI
```

---

## 10. Podsumowanie architektoniczne

```
┌────────────────────────────────────────────────────────────────────┐
│                        CAD ESTIMATOR PRO                           │
├────────────────────────────────────────────────────────────────────┤
│  PRESENTATION (Streamlit)                                          │
│  ├── app.py (routing)                                              │
│  ├── components/ (UI widgets)                                      │
│  └── state/session_manager.py                                      │
├────────────────────────────────────────────────────────────────────┤
│  APPLICATION (Use Cases)                                           │
│  ├── estimation_pipeline.py (main orchestrator)                    │
│  └── batch_importer.py                                             │
├────────────────────────────────────────────────────────────────────┤
│  DOMAIN (Business Logic)                                           │
│  ├── models/ (Component, Estimate, Risk, etc.)                     │
│  ├── interfaces/ (AIClient, DatabaseClient, etc.)                  │
│  └── exceptions.py                                                 │
├────────────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE (External Services)                                │
│  ├── ai/ollama_client.py → Ollama API                              │
│  ├── multi_model/ → 4-stage pipeline                               │
│  ├── database/postgres_client.py → PostgreSQL                      │
│  ├── embeddings/pgvector_service.py → Vector search                │
│  ├── parsers/ → Excel, PDF                                         │
│  └── learning/ → Pattern/Bundle learner                            │
└────────────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
    ┌─────────┐         ┌─────────┐         ┌───────────┐
    │ Ollama  │         │PostgreSQL│        │ PDF/Excel │
    │ (LLM)   │         │+pgvector│         │  Files    │
    └─────────┘         └─────────┘         └───────────┘
```

**WNIOSEK:** Architektura jest czysta (DDD + Clean Architecture).
Kod wymaga tylko drobnych poprawek, nie przepisywania od zera.

