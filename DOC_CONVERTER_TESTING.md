# 🧪 DOC-CONVERTER - Przewodnik Testowania

## 📋 SZYBKI START

### 1. Sprawdź czy wszystko działa

```bash
cd /home/michal/moj-asystent-ai

# Sprawdź status wszystkich kontenerów
docker compose ps
```

**Powinno pokazać:**
```
NAME             STATUS          PORTS
doc-converter    Up (healthy)    8502
whisper          Up (healthy)    9000
pyannote         Up (healthy)    8001
ollama           Up (healthy)    11434
```

---

## 🔍 DIAGNOSTYKA KROK PO KROKU

### Krok 1: Sprawdź doc-converter

```bash
# Logi doc-converter
docker compose logs doc-converter --tail=50

# Health check
curl http://localhost:8502/_stcore/health

# Otwórz w przeglądarce
firefox http://localhost:8502
```

**Co powinno być:**
- ✅ `External URL: http://localhost:8502`
- ✅ `You can now view your Streamlit app`

### Krok 2: Sprawdź whisper (transkrypcja audio)

```bash
# Logi whisper
docker compose logs whisper --tail=30

# Health check
curl http://localhost:9000/docs

# Test API
curl http://localhost:9000/
```

**Odpowiedź:** `{"message":"Whisper ASR API"}`

### Krok 3: Sprawdź pyannote (rozpoznawanie mówców)

```bash
# Logi pyannote
docker compose logs pyannote --tail=50

# Health check
curl http://localhost:8001/health
```

**Jeśli działa:**
```json
{"status":"ok","model_loaded":true}
```

**Jeśli timeout/nie działa:**
```
curl: (7) Failed to connect
lub
curl: (28) Operation timed out
```

### Krok 4: Sprawdź ollama (AI models)

```bash
# Health check
curl http://localhost:11434/api/tags

# Lista modeli
docker compose exec ollama ollama list
```

---

## ⚠️ PYANNOTE TIMEOUT - CO ZROBIĆ?

### Co to jest pyannote?

**Pyannote** = rozpoznaje KTO mówi w pliku audio (speaker diarization)
- Whisper → transkrybuje CO powiedziano
- Pyannote → rozpoznaje KTO to powiedział

### Dlaczego timeout?

1. **Model się ładuje** (pierwsze uruchomienie 1-3 minuty)
2. **Brak HF_TOKEN** (Hugging Face token)
3. **Za mało RAM/GPU**
4. **Port zablokowany**

### ROZWIĄZANIE 1: Poczekaj na model

```bash
# Obserwuj logi
docker compose logs -f pyannote

# Szukaj:
# ✅ "Model loaded successfully"
# ❌ "Model loading failed"
# 🔄 "Loading model..." (czekaj 1-3 min)
```

**Jeśli widzisz "Loading model...":**
```bash
# Poczekaj 2-3 minuty, potem sprawdź znowu
sleep 180
curl http://localhost:8001/health
```

### ROZWIĄZANIE 2: Sprawdź HF_TOKEN

```bash
cd /home/michal/moj-asystent-ai

# Sprawdź czy .env ma token
cat .env | grep HF_TOKEN

# Jeśli brak:
nano .env
```

**Dodaj:**
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

**Gdzie wziąć token?**
1. https://huggingface.co/settings/tokens
2. Create new token → Role: Read
3. Skopiuj i wklej do .env

**Restart pyannote:**
```bash
docker compose restart pyannote
docker compose logs -f pyannote
```

### ROZWIĄZANIE 3: Wyłącz pyannote (jeśli nie potrzebujesz)

**Doc-converter działa BEZ pyannote!** Różnica:

| Z pyannote | Bez pyannote |
|------------|--------------|
| Transkrypcja + KTO mówi | Tylko transkrypcja |
| `Speaker 1: tekst`<br>`Speaker 2: tekst` | Cały tekst bez podziału |

**Jak wyłączyć:**

```bash
cd /home/michal/moj-asystent-ai
nano docker-compose.yml
```

**Zakomentuj sekcję pyannote:**
```yaml
#  pyannote:
#    build: ./pyannote
#    container_name: pyannote
#    ...
```

**Usuń depends_on w doc-converter:**
```yaml
doc-converter:
  depends_on:
    ollama:
      condition: service_healthy
    whisper:
      condition: service_started
    # pyannote:                    ← ZAKOMENTUJ
    #   condition: service_started ← ZAKOMENTUJ
```

**Restart:**
```bash
docker compose up -d
```

### ROZWIĄZANIE 4: Zwiększ timeout

```bash
nano docker-compose.yml
```

**Zmień healthcheck pyannote:**
```yaml
pyannote:
  healthcheck:
    test: ["CMD-SHELL", "curl -fsS http://localhost:8000/health"]
    interval: 60s          # było 30s
    timeout: 30s           # było 10s
    retries: 20            # było 10
    start_period: 300s     # było 120s (daj 5 minut!)
```

---

## 🎯 TESTOWANIE DOC-CONVERTER

### Test 1: Upload PDF

1. **Otwórz:** http://localhost:8502
2. **Kliknij:** "Upload files"
3. **Wybierz:** Dowolny PDF
4. **Sprawdź:** Czy wyświetla tekst

**Przykład testowego PDF:**
```bash
# Stwórz testowy PDF z tekstu
echo "To jest test PDF dla doc-converter" > test.txt
nano test.txt
# Zapisz coś więcej

# Albo użyj istniejącego PDF
```

### Test 2: Upload obrazu (OCR)

1. **Plik:** Zrzut ekranu lub zdjęcie z tekstem
2. **Upload** do doc-converter
3. **Sprawdź:** Czy OCR rozpoznał tekst

### Test 3: Upload audio (Whisper + Pyannote)

**Jeśli masz plik .mp3 / .wav:**
```bash
# Upload w doc-converter
# Sprawdź czy:
# ✅ Transkrypcja działa (Whisper)
# ✅ Podział na mówców (Pyannote) - jeśli działa
```

### Test 4: Generowanie podsumowania (Ollama)

1. **Upload dokumentu**
2. **Kliknij:** "Generate Summary" (jeśli dostępne)
3. **Sprawdź:** Czy Ollama generuje podsumowanie

**Jeśli błąd - sprawdź Ollama:**
```bash
docker compose exec ollama ollama list

# Jeśli brak modeli - pobierz:
docker compose exec ollama ollama pull llama2
docker compose exec ollama ollama pull mistral
```

---

## 📊 SPRAWDZANIE LOGÓW

### Wszystkie logi na raz:

```bash
cd /home/michal/moj-asystent-ai

# Ostatnie 50 linii z każdego
docker compose logs --tail=50

# Live monitoring (Ctrl+C aby wyjść)
docker compose logs -f

# Tylko doc-converter
docker compose logs -f doc-converter
```

### Szukanie błędów:

```bash
# Szukaj ERROR
docker compose logs | grep -i error

# Szukaj TIMEOUT
docker compose logs | grep -i timeout

# Szukaj FAILED
docker compose logs | grep -i failed
```

---

## 🔧 TYPOWE PROBLEMY

### Problem: "Connection refused"

```bash
# Sprawdź czy kontener działa
docker compose ps doc-converter

# Sprawdź porty
netstat -tulpn | grep 8502

# Restart
docker compose restart doc-converter
```

### Problem: "Whisper timeout"

```bash
# Zwiększ pamięć dla whisper
nano docker-compose.yml

# Dodaj:
whisper:
  deploy:
    resources:
      limits:
        memory: 4G
      reservations:
        memory: 2G
```

### Problem: "Ollama model not found"

```bash
# Lista modeli
docker compose exec ollama ollama list

# Pobierz model
docker compose exec ollama ollama pull llama2:7b

# Sprawdź czy działa
docker compose exec ollama ollama run llama2 "test"
```

---

## ✅ CHECKLIST PRZED TESTEM

- [ ] Docker Compose uruchomiony: `docker compose ps`
- [ ] Doc-converter UP (healthy)
- [ ] Whisper UP (healthy)
- [ ] Ollama UP (healthy)
- [ ] Pyannote UP (healthy) - opcjonalnie
- [ ] Port 8502 dostępny: `curl localhost:8502`
- [ ] Przegladarka otwarta: http://localhost:8502
- [ ] Przygotowane pliki testowe (PDF, zdjęcie, audio)

---

## 🚀 SZYBKI TEST (30 sekund)

```bash
cd /home/michal/moj-asystent-ai

# 1. Sprawdź status
docker compose ps | grep -E "doc-converter|whisper|ollama|pyannote"

# 2. Health checks
curl -s http://localhost:8502/_stcore/health && echo "✅ Doc-converter OK"
curl -s http://localhost:9000/ && echo "✅ Whisper OK"
curl -s http://localhost:11434/api/tags && echo "✅ Ollama OK"
curl -s http://localhost:8001/health && echo "✅ Pyannote OK" || echo "⚠️  Pyannote timeout (opcjonalny)"

# 3. Otwórz w przeglądarce
echo "Otwórz: http://localhost:8502"
```

**Gotowe!** 🎉
