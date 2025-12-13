#!/bin/bash
set -euo pipefail

# ==============================================================================
# === USTAWIENIA
# ==============================================================================
# Katalogi robocze Git
CODE_REPO_DIR="/home/michal/ai-code"
DEPLOY_REPO_DIR="/home/michal/ai-code-deploy"

# Katalog docelowy, gdzie działa Docker Compose
PROJECT_DIR="/home/michal/moj-asystent-ai"

# Logi
LOG_FILE="/home/michal/deploy-ai.log"

# ==============================================================================
# === STEROWANIE
# ==============================================================================
ACTION="${ACTION:-deploy}"
REBUILD="${REBUILD:-0}"

# ==============================================================================
# === FUNKCJE POMOCNICZE
# ==============================================================================
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }
err() { echo "[$(ts)] ERROR: $*" | tee -a "$LOG_FILE" >&2; }

# --- Funkcja dla akcji 'deploy' (pobieranie i kopiowanie) ---
run_deploy() {
  log "Rozpoczynam DEPLOY..."

  log "Aktualizacja repozytorium z kodem..."
  (cd "$CODE_REPO_DIR" && git pull)

  log "Kopiowanie doc-converter do katalogu projektu..."
  # Upewnij się, że katalog docelowy istnieje
  mkdir -p "$PROJECT_DIR"

  # Kopiuj całą strukturę doc-converter
  if [ -d "$CODE_REPO_DIR/doc-converter" ]; then
    log "-> Kopiuję całą strukturę doc-converter..."
    rsync -av --delete \
      "$CODE_REPO_DIR/doc-converter/" \
      "$PROJECT_DIR/doc-converter/"
    log "-> Skopiowano: doc-converter/ (Dockerfile, requirements.txt, app/)"
  else
    err "Brak katalogu doc-converter w repozytorium!"
    exit 1
  fi

  log "Kopiowanie cad-panel do katalogu projektu..."
  # Upewnij się, że katalogi docelowe istnieją
  mkdir -p "$PROJECT_DIR/cad/app"

  # Kopiuj plik cad-panel
  if [ -f "$CODE_REPO_DIR/cad_main.py" ]; then
    cp "$CODE_REPO_DIR/cad_main.py" "$PROJECT_DIR/cad/app/main.py"
    log "-> Skopiowano: cad_main.py -> cad/app/main.py"
  else
    log "-> Uwaga: brak pliku cad_main.py (pomijam)"
  fi

  log "Pliki aplikacji zaktualizowane."

  if [ "$REBUILD" = "1" ]; then
    log "Przebudowa i restartowanie usług aplikacyjnych..."
    (cd "$PROJECT_DIR" && docker compose up -d --build --force-recreate --no-deps doc-converter cad-panel)
  else
    log "Restartowanie usług aplikacyjnych..."
    (cd "$PROJECT_DIR" && docker compose up -d --force-recreate --no-deps doc-converter cad-panel)
  fi

  log "Sprawdzanie statusu usług..."
  sleep 5
  DOC_STATUS=$(curl -sI http://localhost:8502/_stcore/health 2>/dev/null | head -n1 || echo "Brak odpowiedzi")
  CAD_STATUS=$(curl -sI http://localhost:8501/_stcore/health 2>/dev/null | head -n1 || echo "Brak odpowiedzi")
  log "Status doc-converter (port 8502): ${DOC_STATUS}"
  log "Status cad-panel (port 8501):     ${CAD_STATUS}"

  log "Sprawdzanie logów (ostatnie 20 linii)..."
  (cd "$PROJECT_DIR" && docker compose logs --tail=20 doc-converter)
}

# --- Funkcja dla akcji 'push' (wysyłanie) ---
run_push() {
  log "Rozpoczynam PUSH na GitHub..."

  log "Kopiowanie plików z projektu do repozytoriów Git..."

  # Kopiuj całą strukturę doc-converter z powrotem do repo
  if [ -d "$PROJECT_DIR/doc-converter" ]; then
    rsync -av --delete \
      "$PROJECT_DIR/doc-converter/" \
      "$CODE_REPO_DIR/doc-converter/"
    log "-> Skopiowano doc-converter/ do repozytorium"
  fi

  # Kopiuj cad_main.py
  if [ -f "$PROJECT_DIR/cad/app/main.py" ]; then
    cp "$PROJECT_DIR/cad/app/main.py" "$CODE_REPO_DIR/cad_main.py"
    log "-> Skopiowano cad/app/main.py -> cad_main.py"
  fi

  log "Pliki aplikacji przygotowane do wysłania."

  log "Wysyłanie zmian z repozytorium kodu..."
  if ! (cd "$CODE_REPO_DIR" && git diff-index --quiet HEAD --); then
    (cd "$CODE_REPO_DIR" && git add . && git commit -m "update: $(date +'%Y-%m-%d %H:%M')" && git push)
    log "Zmiany w kodzie wysłane."
  else
    log "Brak zmian w repozytorium kodu."
  fi
}

# ==============================================================================
# === GŁÓWNA LOGIKA
# ==============================================================================
log "=== Rozpoczynam akcję: ${ACTION} ==="

if [ "$ACTION" = "deploy" ]; then
  run_deploy
elif [ "$ACTION" = "push" ]; then
  run_push
else
  err "Nieznana akcja: ${ACTION}. Dostępne: 'deploy', 'push'."
  exit 1
fi

log "=== Akcja '${ACTION}' zakończona ==="
echo "" >> "$LOG_FILE"
