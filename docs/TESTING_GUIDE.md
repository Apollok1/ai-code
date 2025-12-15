# 🧪 INSTRUKCJA TESTOWANIA - Krok po kroku

**Data:** 2025-12-15
**Status:** Wszystkie serwisy uruchomione ✅

---

## 📋 **CHECKLIST TESTOWANIA**

### ✅ **1. DOC-CONVERTER (Port 8502) - REFACTORED VERSION**

#### **Test 1.1: Sprawdź czy interface się ładuje**

```bash
# W przeglądarce:
http://localhost:8502
```

**✅ Oczekiwany wynik:**
- [ ] Widzisz interface Streamlit
- [ ] Sidebar ma sekcje: ⚙️ Configuration, 📝 OCR, 👁️ Vision, 🎤 Audio, ⚡ Performance
- [ ] Brak błędów importu (sprawdź DevTools Console - F12)

**❌ Jeśli nie działa:**
```bash
docker logs doc-converter | grep -i error
```

---

#### **Test 1.2: Upload PDF z tekstem**

1. Otwórz http://localhost:8502
2. W sidebar ustaw:
   - OCR Language: `pol+eng`
   - Max Pages: `20`
3. Upload prosty PDF (np. faktura, dokument)
4. Kliknij "Convert"

**✅ Oczekiwany wynik:**
- [ ] Progress bar się pokazuje
- [ ] Konwersja zajmuje kilka sekund
- [ ] Wyświetla się wyekstrahowany tekst
- [ ] Można pobrać plik tekstowy

**🔍 Sprawdź logi:**
```bash
docker logs doc-converter --tail=30
```

Powinno być:
```
INFO - Starting Document Converter Pro v2.0
INFO - Configuration loaded successfully
INFO - Processing file: your_file.pdf
INFO - Extraction complete
```

---

#### **Test 1.3: Upload audio (MP3/WAV)**

1. Upload plik audio (np. nagranie głosowe)
2. Ustaw w sidebar:
   - Enable Speaker Diarization: ✅
   - Enable Meeting Summaries: ✅
3. Kliknij "Convert"

**✅ Oczekiwany wynik:**
- [ ] Transkrypcja się wykonuje (Whisper)
- [ ] Pokazuje speakers (Speaker 0, Speaker 1, etc.)
- [ ] Generuje podsumowanie (jeśli włączone)
- [ ] Czas: ~30 sekund dla 1 minuty audio

**🔍 Sprawdź czy Whisper działa:**
```bash
curl http://localhost:9000/transcribe -X POST \
  -F "audio_file=@test.mp3" \
  -F "language=pl" \
  -F "model=medium"
```

**❌ Jeśli błąd 422 z Pyannote:**
```bash
docker logs pyannote --tail=20
# Sprawdź czy HF_TOKEN jest ustawiony
```

---

#### **Test 1.4: Upload obrazu**

1. Upload zdjęcie/screenshot
2. W sidebar ustaw:
   - Enable Vision Models: ✅
   - Image Processing Mode: `describe`
3. Kliknij "Convert"

**✅ Oczekiwany wynik:**
- [ ] Vision model (qwen2.5vl:7b) analizuje obraz
- [ ] Generuje opis po polsku
- [ ] Czas: ~5-10 sekund

---

### ✅ **2. CAD-PANEL (Port 8501) - Estimator 10/10**

#### **Test 2.1: Sprawdź interface**

```bash
# W przeglądarce:
http://localhost:8501
```

**✅ Oczekiwany wynik:**
- [ ] Widzisz "🚀 CAD Estimator Pro"
- [ ] Sidebar ma sekcje: 🤖 Modele AI, 🎯 Pipeline Estymacji
- [ ] Menu na górze sidebara: 📋 Menu
- [ ] Brak błędów Python (sprawdź logi poniżej)

**🔍 Sprawdź logi:**
```bash
docker logs cad-panel --tail=50
```

Powinno być:
```
INFO - CAD Estimator Pro starting...
INFO - Configuration loaded successfully
INFO - Multi-model orchestrator initialized
```

**❌ Jeśli błąd:**
```bash
# Sprawdź import errors
docker logs cad-panel | grep -i "modulenotfounderror\|importerror"

# Restart
docker restart cad-panel
sleep 10
docker logs cad-panel --tail=20
```

---

#### **Test 2.2: Single-Model estymacja (szybki test)**

1. Otwórz http://localhost:8501
2. W sidebar:
   - **WYŁĄCZ** "Multi-Model Pipeline"
   - Model tekstowy: `qwen2.5:7b`
3. Przejdź do "🆕 Nowy projekt"
4. Wybierz dział: `131 - Automotive`
5. Wpisz opis:
   ```
   Rama stalowa pod przenośnik taśmowy, długość 5m, ciężar 500kg,
   konstrukcja spawana ze stali S235JR
   ```
6. Kliknij "Generuj Estymację"

**✅ Oczekiwany wynik:**
- [ ] Estymacja trwa 10-20 sekund
- [ ] Pokazuje komponenty (Frame, Supports, Welds, etc.)
- [ ] Każdy komponent ma godziny: 3D Layout, 3D Detail, 2D
- [ ] Suma godzin: ~30-60h (zależy od modelu)
- [ ] Confidence: 0.6-0.8

**🔍 Sprawdź logi:**
```bash
docker logs cad-panel | grep -E "Stage|Estimate"
```

---

#### **Test 2.3: Multi-Model Pipeline (pełny test - 10/10!)**

1. W sidebar:
   - **WŁĄCZ** "Multi-Model Pipeline (4 etapy)"
   - Rozwiń "⚙️ Wybór modeli per etap"
   - Ustaw:
     - 1️⃣ Technical Analysis: `qwen2.5:14b`
     - 2️⃣ Structural Decomposition (CRITICAL): `qwen2.5:14b` ⭐
     - 3️⃣ Hours Estimation: `qwen2.5:7b`
     - 4️⃣ Risk Analysis: `qwen2.5:14b`

2. Wpisz bardziej złożony opis:
   ```
   Kompletny system przenośnika taśmowego z napędem elektrycznym 3kW,
   sterowaniem PLC Siemens S7-1200, zasilaniem 400V, konstrukcją stalową
   ocynkowaną, długością 8m, wydajnością 1000kg/h, z systemem bezpieczeństwa
   (emergency stop, light curtains)
   ```

3. Kliknij "Generuj Estymację"

**✅ Oczekiwany wynik:**
- [ ] **Stage 1** (Technical Analysis): ~15 sekund
  - Complexity: high
  - Materials: S235JR, galvanized steel, elektryka
  - Standards: ISO, EN

- [ ] **Stage 2** (Structural Decomposition): ~20 sekund
  - Hierarchia komponentów (Main Assembly → Sub-assemblies → Parts)
  - Component count: 20-40 komponentów
  - Depth: 2-4 poziomy
  - ⚠️ **Ten etap jest KRYTYCZNY** - sprawdź czy hierarchia ma sens!

- [ ] **Stage 3** (Hours Estimation): ~15 sekund
  - Każdy komponent ma estymację godzin
  - Total hours: ~150-250h dla tego projektu
  - Pattern matching z bazy (jeśli są podobne projekty)

- [ ] **Stage 4** (Risk Analysis): ~20 sekund
  - Lista ryzyk (Medium/High severity)
  - Suggestions (jak zoptymalizować)
  - Assumptions (co założono)
  - Warnings (na co uważać)

**Suma czasu:** ~70 sekund dla pełnego pipeline

**🔍 Sprawdź walidację:**
```bash
docker logs cad-panel | grep -i "validation"
```

Powinno być:
```
INFO - ✓ Stage 1 validation passed
INFO - ✓ Stage 2 validation passed (CRITICAL stage validated)
INFO - ✓ Stage 3 validation passed (total=XXX.Xh)
```

**❌ Jeśli błąd ValidationError:**
```bash
docker logs cad-panel | tail -50
# Model zwrócił niepoprawny JSON lub brakujące pola
```

---

#### **Test 2.4: Sprawdź Stage 2 szczegółowo (10/10 check)**

Po uruchomieniu multi-model pipeline:

**Sprawdź hierarchię komponentów:**
- [ ] Są główne assemblies (np. "Frame Assembly", "Drive System")
- [ ] Każdy główny ma sub-assemblies
- [ ] Sub-assemblies mają konkretne części
- [ ] Liczby komponentów są realistyczne (nie 1000!)
- [ ] Nazwy mają sens techniczny (nie "Component 1", "Part A")

**Przykład dobrej hierarchii:**
```
Frame Assembly
├── Main Beam (qty: 2)
├── Support Structure (qty: 4)
│   ├── Bracket (qty: 8)
│   └── Fasteners (qty: 32)
└── Welded Joints (qty: 16)

Drive System
├── Electric Motor 3kW (qty: 1)
├── Gearbox (qty: 1)
└── Coupling (qty: 1)
```

**🔍 Logi Stage 2:**
```bash
docker logs cad-panel | grep "Stage 2"
```

---

### ✅ **3. OLLAMA (LLM Backend)**

#### **Test 3.1: Sprawdź modele**

```bash
# Lista dostępnych modeli
curl http://localhost:11434/api/tags | jq '.models[].name'
```

**✅ Oczekiwany wynik:**
```json
"qwen2.5:7b"
"qwen2.5:14b"
"qwen2.5:32b"
"qwen2.5vl:7b"
"llama3:8b"
"nomic-embed-text"
```

**Jeśli brakuje modeli:**
```bash
docker exec -it ollama ollama pull qwen2.5:14b
docker exec -it ollama ollama pull qwen2.5vl:7b
```

---

#### **Test 3.2: Test generacji**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "What is 2+2?",
  "stream": false
}'
```

**✅ Oczekiwany wynik:**
```json
{
  "model": "qwen2.5:7b",
  "response": "2+2 equals 4.",
  ...
}
```

---

### ✅ **4. WHISPER (Audio Transcription)**

```bash
# Test endpoint
curl http://localhost:9000/health
```

**✅ Oczekiwany wynik:**
```json
{"status": "healthy"}
```

**Test transkrypcji** (jeśli masz plik audio):
```bash
curl -X POST http://localhost:9000/transcribe \
  -F "audio_file=@test.mp3" \
  -F "language=pl" \
  -F "model=medium"
```

---

### ✅ **5. PYANNOTE (Speaker Diarization)**

```bash
# Test health
curl http://localhost:8001/health
```

**✅ Oczekiwany wynik:**
```json
{"status": "healthy"}
```

---

### ✅ **6. CAD POSTGRES (Database)**

```bash
# Test połączenia
docker exec -it cad-postgres psql -U cad_user -d cad_estimator -c "SELECT version();"
```

**✅ Oczekiwany wynik:**
```
PostgreSQL 16.x with pgvector
```

---

## 🎯 **PODSUMOWANIE - CHECKLIST**

Po zakończeniu wszystkich testów:

### **Doc-Converter (Refactored v2.0)**
- [ ] Interface działa (http://localhost:8502)
- [ ] PDF conversion działa
- [ ] Audio transcription działa
- [ ] Vision analysis działa
- [ ] Brak import errors w logach
- [ ] **STATUS: PRODUCTION READY ✅**

### **CAD-Panel (10/10 Version)**
- [ ] Interface działa (http://localhost:8501)
- [ ] Menu na górze sidebara ✅
- [ ] Single-model estymacja działa
- [ ] Multi-model pipeline działa (4 etapy)
- [ ] Stage 2 używa 14b (CRITICAL) ✅
- [ ] Walidacja między etapami działa ✅
- [ ] Brak validation errors w logach
- [ ] **STATUS: PRODUCTION READY 10/10 ✅**

### **Backend Services**
- [ ] Ollama działa (http://localhost:11434)
- [ ] Whisper działa (http://localhost:9000)
- [ ] Pyannote działa (http://localhost:8001)
- [ ] PostgreSQL działa (localhost:5432)

---

## 📊 **OCZEKIWANE WYNIKI - BENCHMARK**

### **Doc-Converter Performance:**
- PDF (10 stron): ~5-10 sekund
- Audio (1 minuta): ~30 sekund
- Image analysis: ~5-10 sekund

### **CAD-Panel Performance:**
- Single-model (prosty projekt): ~10-20 sekund
- Multi-model (złożony projekt):
  - Stage 1 (14b): ~15 sekund
  - Stage 2 (14b): ~20 sekund ⭐ CRITICAL
  - Stage 3 (7b): ~15 sekund
  - Stage 4 (14b): ~20 sekund
  - **Total: ~70 sekund**

### **Accuracy (CAD-Panel):**
- Hours estimation error: < 20% (cel)
- Component count error: < 3 components (cel)
- Stage 2 decomposition: logiczna hierarchia ✅

---

## ❌ **TROUBLESHOOTING**

### **Problem: 502 Bad Gateway**
```bash
# Sprawdź logi
docker logs [service-name] --tail=50

# Restart service
docker restart [service-name]
```

### **Problem: Import errors**
```bash
# Doc-converter
docker logs doc-converter | grep -i "modulenotfounderror"
# Powinno być: brak błędów (PYTHONPATH=/app/src działa)

# CAD-panel
docker logs cad-panel | grep -i "importerror"
# Powinno być: brak błędów (PYTHONPATH=/app/src działa)
```

### **Problem: Slow performance**
```bash
# Sprawdź GPU utilization (dla ROCm)
rocm-smi

# Sprawdź RAM
free -h

# Sprawdź czy modele są w pamięci
docker exec -it ollama ps aux | grep ollama
```

### **Problem: ValidationError w Stage 2**
```bash
# Stage 2 zwrócił nieprawidłową strukturę
docker logs cad-panel | grep "Stage 2"

# Rozwiązanie: model może potrzebować lepszego promptu
# lub zwiększ temperaturę/zmniejsz top_p w konfiguracji
```

---

## 🚀 **NASTĘPNE KROKI**

Po zakończeniu testów:

1. **Zbierz dane benchmarkowe:**
   - Uruchom 10-20 testów estymacji
   - Zapisz: czas, dokładność, błędy
   - Porównaj single vs multi-model

2. **Dostosuj prompty** (jeśli potrzeba):
   - Zobacz: `/home/user/ai-code/src/cad/infrastructure/multi_model/stage*.py`
   - Metody: `_build_*_prompt()`

3. **Uruchom benchmarki** (gdy masz dane historyczne):
   ```bash
   cd ~/ai-code
   python -m cad.scripts.run_benchmark --stage1 qwen2.5:14b --stage2 qwen2.5:14b
   ```

4. **Monitoruj produkcję:**
   ```bash
   # Sprawdzaj logi regularnie
   docker logs cad-panel | grep -i "validation\|error"
   docker logs doc-converter | grep -i "error"
   ```

---

## 📝 **UWAGI KOŃCOWE**

### **Co zostało zmienione (10/10 improvements):**
1. ✅ Stage 2 → 14b (CRITICAL fix)
2. ✅ Menu na górę sidebara
3. ✅ Walidacja między etapami (sanity checks)
4. ✅ Doc-converter → refactored architecture
5. ✅ Benchmarking framework dodany

### **Co należy monitorować:**
- Stage 2 decomposition quality (najważniejsze!)
- Hours estimation accuracy
- Validation errors
- Performance (czas wykonania)

---

**Autor:** Claude Code
**Data:** 2025-12-15
**Wersja:** Production 10/10 ✅
