# 🚀 Deployment: Direct Mounting (BEZ kopiowania!)

## 📋 Koncepcja

Zamiast kopiować pliki z `ai-code` do `moj-asystent-ai`, Docker **montuje katalogi bezpośrednio** z repozytorium Git.

### ✅ Zalety:
- **Brak kopiowania** - oszczędność czasu i miejsca
- **Zmiany widoczne od razu** - edytujesz w ai-code, Streamlit auto-reload wykrywa
- **Prostsza struktura** - jeden katalog z kodem
- **Bezpieczniejsze** - brak ryzyka niezsynchronizowanych kopii

### 📁 Struktura:

```
/home/michal/
├── ai-code/                           # ← TUTAJ EDYTUJESZ KOD
│   ├── doc-converter/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       └── converter.py
│   ├── cad/
│   │   ├── Dockerfile
│   │   └── app/
│   │       └── main.py
│   ├── docker-compose.direct-mount.yml
│   ├── deploy-direct.sh
│   └── .env.example
│
└── moj-asystent-ai/                   # ← TUTAJ DZIAŁA DOCKER
    ├── docker-compose.yml             # (skopiowane z direct-mount.yml)
    ├── .env                           # konfiguracja ścieżek
    ├── outputs/                       # dane aplikacji
    ├── ollama_data/
    └── cad/
        └── postgres-data/
```

Docker montuje:
- `ai-code/doc-converter/app` → `/app/app` w kontenerze
- `ai-code/cad/app` → `/app` w kontenerze

## 🔧 Setup (TYLKO RAZ)

### 1. Sklonuj repozytorium (jeśli jeszcze nie masz):

```bash
cd /home/michal
git clone https://github.com/Apollok1/ai-code.git
cd ai-code
```

### 2. Utwórz katalog projektu:

```bash
mkdir -p /home/michal/moj-asystent-ai
cd /home/michal/moj-asystent-ai
```

### 3. Skopiuj konfigurację (tylko raz):

```bash
# Docker Compose
cp /home/michal/ai-code/docker-compose.direct-mount.yml ./docker-compose.yml

# Konfiguracja środowiskowa
cp /home/michal/ai-code/.env.example ./.env
```

### 4. Edytuj `.env`:

```bash
nano .env
```

**WAŻNE:** Ustaw poprawną ścieżkę do ai-code:

```bash
AI_CODE_PATH=/home/michal/ai-code
ANYTHINGLLM_API_KEY=twój_klucz
HF_TOKEN=twój_token
```

### 5. Utwórz katalogi dla danych:

```bash
mkdir -p outputs ollama_data cad/postgres-data storage/chroma
```

### 6. Pierwsze uruchomienie:

```bash
cd /home/michal/ai-code
chmod +x deploy-direct.sh
REBUILD=1 ./deploy-direct.sh
```

## 🚀 Codzienne użycie

### Aktualizacja i restart:

```bash
cd /home/michal/ai-code
./deploy-direct.sh
```

**Co się dzieje:**
1. ✅ `git pull` - pobiera najnowszy kod
2. ✅ `docker compose restart` - restartuje kontenery
3. ✅ Sprawdza health check
4. ✅ Pokazuje logi

### Przebudowa (po zmianach w Dockerfile/requirements):

```bash
REBUILD=1 ./deploy-direct.sh
```

### Tylko restart (bez git pull):

```bash
cd /home/michal/moj-asystent-ai
docker compose restart doc-converter cad-panel
```

## 🔄 Workflow programisty

### 1. Edycja kodu:

```bash
cd /home/michal/ai-code
nano doc-converter/app/converter.py
```

### 2. Sprawdź zmiany:

```bash
git status
git diff
```

### 3. Commit i push:

```bash
git add .
git commit -m "update: poprawka w doc-converter"
git push
```

### 4. Wdróż na serwerze:

```bash
# Na serwerze
cd /home/michal/ai-code
./deploy-direct.sh
```

**GOTOWE!** Zmiany są od razu widoczne.

## 📊 Monitorowanie

### Sprawdź status:

```bash
cd /home/michal/moj-asystent-ai
docker compose ps
```

### Logi na żywo:

```bash
# Doc-converter
docker compose logs -f doc-converter

# CAD-panel
docker compose logs -f cad-panel

# Wszystkie usługi
docker compose logs -f
```

### Health check:

```bash
curl http://localhost:8502/_stcore/health  # doc-converter
curl http://localhost:8501/_stcore/health  # cad-panel
```

## 🐛 Troubleshooting

### Problem: "Brak dostępu do plików"

**Przyczyna:** Niepoprawna ścieżka w `.env`

**Rozwiązanie:**
```bash
cd /home/michal/moj-asystent-ai
nano .env

# Ustaw:
AI_CODE_PATH=/home/michal/ai-code
```

### Problem: "Kod się nie aktualizuje"

**Przyczyna:** Docker cache lub brak restartu

**Rozwiązanie:**
```bash
cd /home/michal/ai-code
REBUILD=1 ./deploy-direct.sh
```

### Problem: "Kontener nie startuje po zmianach"

**Sprawdź logi:**
```bash
cd /home/michal/moj-asystent-ai
docker compose logs doc-converter
```

**Sprawdź czy plik istnieje:**
```bash
ls -la /home/michal/ai-code/doc-converter/app/converter.py
```

### Problem: "Permission denied"

**Zmień uprawnienia:**
```bash
chmod +x /home/michal/ai-code/deploy-direct.sh
```

## 🔐 Uprawnienia

Docker montuje pliki jako **read-only** (`:ro`), więc kontenery nie mogą modyfikować kodu źródłowego. To bezpieczne!

Katalogi do zapisu (outputs, data) są montowane bez `:ro`.

## 🎯 Porównanie ze starą metodą

| Funkcja | Stara metoda (kopiowanie) | Nowa metoda (direct mount) |
|---------|---------------------------|----------------------------|
| Aktualizacja kodu | `git pull` + `rsync` + restart | `git pull` + restart |
| Czas deployu | ~15-30s | ~5-10s |
| Miejsce na dysku | 2x więcej (kopia) | Tylko raz |
| Ryzyko błędów | Możliwe niezsynchronizowane kopie | Jedna wersja kodu |
| Auto-reload | Wymaga restartu | Streamlit wykrywa zmiany |

## 📝 Notatki

- **Auto-reload Streamlit:** W docker-compose ustawiony `--server.fileWatcherType=auto`, więc Streamlit wykrywa zmiany w plikach `.py`
- **Volume mounting:** Katalogi montowane jako read-only (`:ro`) - kontenery nie mogą modyfikować kodu
- **Build context:** Dockerfile nadal budowany z kontekstu `ai-code/doc-converter`, ale `app/` montowane live
- **Dane:** Katalogi `outputs/`, `postgres-data/` itp. pozostają w `moj-asystent-ai/`
