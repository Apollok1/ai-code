#!/bin/bash
set -euo pipefail

# ==============================================================================
# DEPLOY SCRIPT - DIRECT MOUNTING (bez kopiowania!)
# ==============================================================================
# Ten skrypt używa docker-compose z direct mounting - kod jest montowany
# bezpośrednio z ai-code, więc nie ma potrzeby kopiowania plików.
# ==============================================================================

# Kolory dla logów
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo -e "${GREEN}[$(ts)]${NC} $*"; }
info() { echo -e "${BLUE}[$(ts)]${NC} ℹ️  $*"; }
warn() { echo -e "${YELLOW}[$(ts)]${NC} ⚠️  $*"; }
err() { echo -e "${RED}[$(ts)]${NC} ❌ ERROR: $*" >&2; }

# ==============================================================================
# USTAWIENIA
# ==============================================================================
PROJECT_DIR="${PROJECT_DIR:-/home/user/ai-code}"
AI_CODE_PATH="${AI_CODE_PATH:-/home/user/ai-code}"
ACTION="${ACTION:-update}"
REBUILD="${REBUILD:-0}"

# ==============================================================================
# FUNKCJE
# ==============================================================================

check_requirements() {
    log "Sprawdzanie wymagań..."

    # Sprawdź czy istnieje katalog z kodem
    if [ ! -d "$AI_CODE_PATH" ]; then
        err "Katalog $AI_CODE_PATH nie istnieje!"
        exit 1
    fi

    # Sprawdź czy istnieje .env w projekcie
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        warn "Brak pliku .env w $PROJECT_DIR"
        warn "Tworzę .env z .env.example..."
        cp "$AI_CODE_PATH/.env.example" "$PROJECT_DIR/.env"
        warn "WAŻNE: Uzupełnij plik $PROJECT_DIR/.env swoimi danymi!"
    fi

    # Sprawdź czy docker-compose.yml istnieje
    if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
        warn "Brak docker-compose.yml w $PROJECT_DIR"
        info "Kopiuję docker-compose.direct-mount.yml..."
        cp "$AI_CODE_PATH/docker-compose.direct-mount.yml" "$PROJECT_DIR/docker-compose.yml"
    fi

    info "Wymagania spełnione ✓"
}

update_code() {
    log "Aktualizacja kodu z GitHub..."

    cd "$AI_CODE_PATH"

    # Zapisz aktualną gałąź
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    info "Gałąź: $CURRENT_BRANCH"

    # Pull
    if git pull; then
        info "Kod zaktualizowany ✓"
    else
        err "Błąd podczas git pull!"
        exit 1
    fi

    # Pokaż ostatni commit
    LAST_COMMIT=$(git log -1 --pretty=format:"%h - %s (%cr)")
    info "Ostatni commit: $LAST_COMMIT"
}

restart_services() {
    log "Restartowanie usług Docker..."

    cd "$PROJECT_DIR"

    if [ "$REBUILD" = "1" ]; then
        warn "REBUILD=1 - Przebudowa obrazów Docker..."
        docker compose up -d --build --force-recreate doc-converter cad-panel
    else
        info "Restart kontenerów (bez przebudowy)..."
        docker compose restart doc-converter cad-panel
    fi

    info "Usługi zrestartowane ✓"
}

check_health() {
    log "Sprawdzanie statusu usług..."

    sleep 3

    # Doc-converter
    if curl -sf http://localhost:8502/_stcore/health > /dev/null 2>&1; then
        info "✅ doc-converter (8502) - DZIAŁA"
    else
        warn "⚠️  doc-converter (8502) - BRAK ODPOWIEDZI"
    fi

    # CAD-panel
    if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        info "✅ cad-panel (8501) - DZIAŁA"
    else
        warn "⚠️  cad-panel (8501) - BRAK ODPOWIEDZI"
    fi
}

show_logs() {
    log "Ostatnie logi z usług:"

    cd "$PROJECT_DIR"

    echo ""
    info "=== DOC-CONVERTER ==="
    docker compose logs --tail=10 doc-converter

    echo ""
    info "=== CAD-PANEL ==="
    docker compose logs --tail=10 cad-panel
}

show_info() {
    cat << EOF

${GREEN}╔════════════════════════════════════════════════════════════════╗
║  DEPLOYMENT ZAKOŃCZONY - DIRECT MOUNTING                       ║
╚════════════════════════════════════════════════════════════════╝${NC}

${BLUE}📁 Struktura:${NC}
   Kod:        $AI_CODE_PATH
   Docker:     $PROJECT_DIR

${BLUE}🔗 Montowanie:${NC}
   - doc-converter/app → zamontowane BEZPOŚREDNIO z ai-code
   - cad/app          → zamontowane BEZPOŚREDNIO z ai-code

${BLUE}💡 Jak to działa:${NC}
   1. Edytujesz kod w: $AI_CODE_PATH
   2. Zmiany są widoczne OD RAZU w kontenerach
   3. Streamlit auto-reload wykrywa zmiany
   4. NIE TRZEBA kopiować plików!

${BLUE}🚀 Kolejne aktualizacje:${NC}
   cd $AI_CODE_PATH
   git pull
   ./deploy-direct.sh                    # Restart usług

   lub z przebudową (po zmianie Dockerfile/requirements):
   REBUILD=1 ./deploy-direct.sh

${BLUE}📊 Status:${NC}
   - doc-converter: http://localhost:8502
   - cad-panel:     http://localhost:8501
   - anythingllm:   http://localhost:3001
   - ollama:        http://localhost:11434

${BLUE}📝 Logi:${NC}
   cd $PROJECT_DIR
   docker compose logs -f doc-converter
   docker compose logs -f cad-panel

EOF
}

# ==============================================================================
# GŁÓWNA LOGIKA
# ==============================================================================

main() {
    log "╔════════════════════════════════════════════════════════════════╗"
    log "║  DEPLOY - DIRECT MOUNTING (bez kopiowania!)                    ║"
    log "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    check_requirements
    update_code
    restart_services
    check_health
    show_logs
    show_info
}

# Uruchom
main "$@"
