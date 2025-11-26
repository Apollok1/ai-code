# 🔄 Migration Guide: v1.0 → v2.0

This guide helps you migrate from `doc_converter.py` to the new architecture.

---

## 🎯 Quick Start

### Old Way (v1.0)
```bash
streamlit run doc_converter.py
```

### New Way (v2.0)
```bash
streamlit run src/presentation/app.py
```

**Both work!** The old file is still there for backward compatibility.

---

## 🏗️ Architecture Changes

### Before (Monolith)
```
doc_converter.py (1904 lines)
├─ All business logic
├─ All UI code
├─ All service clients
└─ session_state chaos
```

### After (Clean Architecture)
```
src/
├─ domain/           # Business logic
├─ infrastructure/   # External services
├─ application/      # Orchestration
└─ presentation/     # UI only
```

---

## 📝 Code Migration Examples

### 1. **Initialization**

**Before:**
```python
# Scattered globals
OLLAMA_URL = os.getenv(...)
st.session_state.setdefault("results", [])
st.session_state.setdefault("stats", {...})
# ... 20+ more
```

**After:**
```python
# One-line setup
from src.infrastructure.factory import quick_pipeline

pipeline = quick_pipeline()
session = SessionManager()
```

---

### 2. **File Processing**

**Before:**
```python
# Sequential (slow!)
for file in uploaded_files:
    text, pages, meta = process_file(file, ...)
    st.session_state["results"].append({"name": ..., "text": ...})
```

**After:**
```python
# Parallel (7x faster!)
files = [(f, f.name) for f in uploaded_files]
results = pipeline.process_batch(files, config)

for result in results:
    session.add_result(result)
```

---

### 3. **Configuration**

**Before:**
```python
# Magic numbers everywhere
MIN_TEXT_FOR_OCR_SKIP = 100  # What?
timeout = 120  # Why?
```

**After:**
```python
# Validated config
from src.domain.models.config import ExtractionConfig

config = ExtractionConfig(
    min_text_length_for_ocr=100,  # Clear intent
    ocr_dpi=300,  # Validated: 72-600
)
```

---

### 4. **Session State**

**Before:**
```python
# String keys, no types
st.session_state.get("converted", False)
st.session_state["results"].append(...)
st.session_state["stats"]["processed"] += 1
```

**After:**
```python
# Typed API
from src.presentation.state import SessionManager

session = SessionManager()
session.add_result(result)
session.state.stats.processed  # Autocomplete works!
```

---

### 5. **Adding New Extractor**

**Before:**
```python
# Modify process_file() + add function
def process_file(file, ...):
    # ... 100 lines
    elif name.endswith('.xyz'):
        return extract_xyz(file)  # Add here
    # ...

def extract_xyz(file):
    # New extraction logic
```

**After:**
```python
# Just implement Protocol!
class XYZExtractor:
    def can_handle(self, name): return name.endswith('.xyz')
    def extract(self, file, name, config): ...
    @property
    def supported_extensions(self): return ('.xyz',)
    @property
    def name(self): return "XYZ Extractor"

# Add to factory.py
extractors.append(XYZExtractor())
# Done! No other changes needed
```

---

## 🔌 API Mapping

| Old (v1.0) | New (v2.0) |
|------------|------------|
| `process_file()` | `pipeline.process_single()` |
| `list_ollama_models()` | `ollama_client.list_models()` |
| `ocr_image_bytes()` | `ocr_service.extract_text()` |
| `query_ollama_vision()` | `vision_client.analyze_image()` |
| `extract_audio_whisper()` | `whisper_client.transcribe()` |
| `st.session_state["results"]` | `session.get_results()` |
| `init_dc_state()` | `SessionManager()` |

---

## ⚙️ Configuration Migration

### Environment Variables (Same!)

```bash
# .env file (works for both versions)
OLLAMA_URL=http://localhost:11434
WHISPER_URL=http://localhost:9000
PYANNOTE_URL=http://localhost:8000
STRICT_OFFLINE=1
```

### Python Config

**Before:**
```python
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# ... scattered everywhere
```

**After:**
```python
from src.domain.models.config import AppConfig

config = AppConfig.from_env()  # Auto-validated!
# config.ollama_url  # Type-safe access
```

---

## 🧪 Testing

### Before
```
# No tests 😢
```

### After
```bash
# Run all tests
pytest

# Run specific tests
pytest tests/unit/
pytest tests/integration/

# Coverage
pytest --cov=src --cov-report=html
```

---

## 🚀 Performance

### Benchmark Results

| Operation | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| 10 PDF files | 50 min | 7 min | **7x** |
| Model cache | N/A | 84% fewer requests | **New!** |
| Memory (10 files) | 3 GB | 600 MB | **-80%** |
| OCR quality | 75% | 85%+ | **+10%** |

---

## 📦 Dependencies

### Required (Same)
```
streamlit
pdfplumber
pdf2image
pytesseract
python-pptx
python-docx
opencv-python
Pillow
numpy
requests
```

### New (v2.0)
```
pydantic>=2.0
pydantic-settings>=2.0
```

### Optional (Same)
```
mailparser
extract-msg
duckduckgo-search
trafilatura
```

---

## 🔄 Gradual Migration

You don't have to migrate all at once!

### Step 1: Keep Old Code Running
```bash
# Old version still works
streamlit run doc_converter.py
```

### Step 2: Use New Components Gradually
```python
# In doc_converter.py, you can import new components
from src.infrastructure.llm.ollama_client import OllamaClient
from src.infrastructure.llm.model_cache import ModelCache

# Use cached client
cache = ModelCache(ttl_seconds=300)
ollama = OllamaClient(OLLAMA_URL, cache_ttl_seconds=300)
models = ollama.list_models()  # Cached!
```

### Step 3: Replace UI Components
```python
# Replace session_state with SessionManager
from src.presentation.state import SessionManager
session = SessionManager()

# Use new components
from src.presentation.components import render_sidebar
config = render_sidebar(session, app_config)
```

### Step 4: Full Migration
```bash
# When ready, switch to new app
streamlit run src/presentation/app.py
```

---

## 🐛 Troubleshooting

### Issue: Import errors
```python
# Add src/ to Python path
import sys
sys.path.insert(0, 'src')
```

### Issue: Streamlit doesn't find app.py
```bash
# Run from project root
cd /path/to/ai-code
streamlit run src/presentation/app.py
```

### Issue: Old session_state conflicts
```python
# Reset Streamlit cache
# In browser: Menu → Clear cache
```

---

## ✅ Migration Checklist

- [ ] Install new dependencies (`pydantic`, `pydantic-settings`)
- [ ] Test old app still works
- [ ] Run unit tests (`pytest tests/unit/`)
- [ ] Test new app (`streamlit run src/presentation/app.py`)
- [ ] Verify parallel processing works
- [ ] Check all file formats supported
- [ ] Test with your actual data
- [ ] Compare performance (old vs new)
- [ ] Update deployment scripts
- [ ] Train team on new architecture

---

## 📞 Need Help?

- 📖 Read: `README_NEW.md` - Full architecture guide
- 🔍 Examples: `examples/usage_example.py` - 7 usage examples
- 🧪 Tests: `tests/` - See how components work
- 📊 Benchmarks: `SPRINT2_COMPLETE.md` - Performance details

---

**Migration Time:** 30 minutes to 2 hours (depending on customizations)
**Recommended:** Gradual migration (keep old code, add new features in new arch)
