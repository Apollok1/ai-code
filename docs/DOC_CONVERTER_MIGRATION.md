# Doc-Converter Migration Guide

**Problem:** There are TWO versions of doc-converter in the codebase

**Date:** 2025-12-15

---

## 📊 **Current Situation**

### **Version 1: Monolithic (Currently Running)**

```
Location: /doc-converter/app/converter.py
Size: 78 KB (monolithic file)
Status: ✅ Currently deployed and working
Import Style: All in one file
```

**Pros:**
- ✅ Working right now
- ✅ Simple deployment

**Cons:**
- ❌ Hard to maintain (78KB single file!)
- ❌ No separation of concerns
- ❌ Difficult to test
- ❌ Hard to extend

---

### **Version 2: Refactored Architecture (Not Deployed)**

```
Location: /src/
Structure:
  src/
  ├── domain/           # Business logic & models
  ├── presentation/     # Streamlit UI
  ├── infrastructure/   # External services (Ollama, Whisper, etc.)
  └── application/      # Use cases & pipelines

Size: ~5.5 KB per file (well-organized)
Status: ⚠️ NOT deployed (import issues)
Import Style: from domain., from presentation., etc.
```

**Pros:**
- ✅ Clean architecture
- ✅ Easy to maintain
- ✅ Testable
- ✅ Extensible
- ✅ Follows best practices

**Cons:**
- ❌ Requires PYTHONPATH configuration
- ❌ Not currently deployed

---

## 🔧 **The Import Problem**

The refactored version uses imports like:

```python
# In src/presentation/components/sidebar.py:
from domain.models.config import AppConfig              # ❌ Fails without PYTHONPATH
from presentation.state.session_manager import ...      # ❌ Fails without PYTHONPATH
```

These imports expect `PYTHONPATH=/app/src` to be set in the Docker container.

**Current Docker setup (old version):**
```yaml
volumes:
  - ${AI_CODE_PATH}/doc-converter/app:/app/app:ro  # ← Mounts OLD monolithic version
```

**Files with this pattern:** 45+ files

---

## ✅ **Solution Implemented**

Created two new files:

### **1. Dockerfile.refactored**
```dockerfile
# Sets PYTHONPATH=/app/src
ENV PYTHONPATH=/app/src

# Copies /src instead of /doc-converter/app
COPY ../src/ ./src/

# Runs new app
CMD ["streamlit", "run", "/app/src/presentation/app.py", ...]
```

### **2. docker-compose.doc-converter-refactored.yml**
```yaml
# Override file for refactored version
services:
  doc-converter:
    build:
      dockerfile: doc-converter/Dockerfile.refactored
    environment:
      - PYTHONPATH=/app/src
    volumes:
      - ${AI_CODE_PATH}/src:/app/src:ro  # ← Mounts NEW refactored version
```

---

## 🚀 **How to Switch to Refactored Version**

### **Option 1: Permanent Switch (Recommended)**

1. **Update main docker-compose file:**

```bash
cd /home/michal/moj-asystent-ai

# Edit docker-compose.direct-mount.yml
# Change doc-converter section to use refactored version
```

2. **Rebuild and restart:**

```bash
docker-compose down doc-converter
docker-compose build doc-converter --no-cache
docker-compose up -d doc-converter
docker-compose logs -f doc-converter
```

### **Option 2: Test with Override File**

```bash
cd /home/michal/moj-asystent-ai

# Use both compose files (second overrides first)
docker-compose \
  -f docker-compose.direct-mount.yml \
  -f docker-compose.doc-converter-refactored.yml \
  up -d doc-converter

# Check logs
docker-compose logs -f doc-converter
```

### **Option 3: Keep Old Version (Status Quo)**

Do nothing - continue using monolithic `converter.py`.

---

## 🧪 **Testing the Refactored Version**

After deploying refactored version:

1. **Open browser:**
   ```
   http://localhost:8502
   ```

2. **Check sidebar loads:**
   - Should see: ⚙️ Configuration
   - Sections: OCR, Vision, Audio, Performance

3. **Test file conversion:**
   - Upload a PDF or audio file
   - Check conversion works
   - Verify output appears

4. **Check logs for errors:**
   ```bash
   docker-compose logs -f doc-converter | grep -i error
   ```

   **Expected (no import errors):**
   ```
   INFO - Starting Document Converter Pro v2.0
   INFO - Configuration loaded successfully
   INFO - Pipeline initialized
   ```

---

## 📋 **Checklist for Migration**

- [ ] Backup current working version (just in case)
- [ ] Build refactored Dockerfile
- [ ] Test with override docker-compose file
- [ ] Verify imports work (no ModuleNotFoundError)
- [ ] Test file conversion (PDF, audio, images)
- [ ] Test all features (OCR, vision, diarization, summarization)
- [ ] Check performance (should be similar or better)
- [ ] If all works ✅ → switch permanently
- [ ] If issues ❌ → revert to old version

---

## 🔄 **Rollback Plan**

If refactored version has issues:

```bash
cd /home/michal/moj-asystent-ai

# Stop refactored version
docker-compose down doc-converter

# Switch back to old docker-compose (without override)
docker-compose -f docker-compose.direct-mount.yml up -d doc-converter

# Verify old version works
docker-compose logs -f doc-converter
```

---

## 📁 **File Structure Comparison**

### **Old Version**
```
doc-converter/
└── app/
    └── converter.py (78 KB - everything in one file)
```

### **New Version**
```
src/
├── domain/
│   ├── models/
│   │   └── config.py            # Configuration models
│   └── interfaces/
│       ├── extractor.py         # Interface definitions
│       └── audio_service.py
├── presentation/
│   ├── app.py                   # Main Streamlit app (5.5 KB)
│   ├── components/
│   │   └── sidebar.py           # Sidebar component (clean!)
│   └── state/
│       └── session_manager.py   # Session management
├── infrastructure/
│   ├── extractors/              # PDF, DOCX, Audio, etc.
│   ├── llm/                     # Ollama client
│   ├── audio/                   # Whisper, Pyannote
│   └── ocr/                     # Tesseract
└── application/
    └── pipeline.py              # Main processing pipeline
```

---

## 💡 **Why Migrate?**

| Aspect | Old (Monolithic) | New (Refactored) |
|--------|------------------|------------------|
| **Maintainability** | ❌ Very hard | ✅ Easy |
| **Testability** | ❌ Difficult | ✅ Unit testable |
| **Extensibility** | ❌ Hard to add features | ✅ Simple to extend |
| **Code Quality** | ❌ 78KB single file | ✅ Clean, organized |
| **Performance** | ✅ Good | ✅ Same or better |
| **Deployment** | ✅ Simple | ⚠️ Needs PYTHONPATH |
| **Debugging** | ❌ Hard to trace | ✅ Clear stack traces |

**Recommendation:** Migrate to refactored version for long-term maintainability.

---

## 🐛 **Troubleshooting**

### **Issue: ModuleNotFoundError: No module named 'domain'**

**Cause:** PYTHONPATH not set correctly

**Fix:**
```bash
# In Dockerfile or docker-compose:
ENV PYTHONPATH=/app/src
```

### **Issue: Import errors after migration**

**Cause:** Cached .pyc files

**Fix:**
```bash
# Clear Python cache
docker-compose exec doc-converter find /app -name "*.pyc" -delete
docker-compose exec doc-converter find /app -name "__pycache__" -type d -exec rm -rf {} +

# Restart
docker-compose restart doc-converter
```

### **Issue: Streamlit doesn't reload on code changes**

**Cause:** fileWatcherType setting

**Fix:**
```yaml
# In docker-compose command:
"--server.fileWatcherType=auto"  # Enable auto-reload
```

---

## 📞 **Support**

If migration issues occur:
1. Check logs: `docker-compose logs -f doc-converter`
2. Verify PYTHONPATH: `docker-compose exec doc-converter env | grep PYTHONPATH`
3. Test imports: `docker-compose exec doc-converter python -c "from domain.models.config import AppConfig"`
4. Rollback if needed (see Rollback Plan above)

---

**Status:** Ready to migrate ✅
**Files Created:**
- ✅ `doc-converter/Dockerfile.refactored`
- ✅ `docker-compose.doc-converter-refactored.yml`
- ✅ This migration guide

**Next Step:** Test with Option 2 (override file) before permanent switch.
