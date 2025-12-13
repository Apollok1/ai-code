#!/bin/bash
# Quick test script for doc-converter

set -euo pipefail

# Kolory
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         DOC-CONVERTER - Quick Health Check                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

PROJECT_DIR="${PROJECT_DIR:-/home/michal/moj-asystent-ai}"

cd "$PROJECT_DIR" 2>/dev/null || {
    echo -e "${RED}❌ Katalog $PROJECT_DIR nie istnieje!${NC}"
    exit 1
}

# Funkcje pomocnicze
check_service() {
    local name=$1
    local url=$2

    if curl -sf "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name${NC} - OK"
        return 0
    else
        echo -e "${RED}❌ $name${NC} - FAILED (timeout lub down)"
        return 1
    fi
}

check_container() {
    local name=$1

    if docker compose ps "$name" 2>/dev/null | grep -q "Up"; then
        echo -e "${GREEN}✅ Container: $name${NC} - UP"
        return 0
    else
        echo -e "${RED}❌ Container: $name${NC} - DOWN"
        return 1
    fi
}

# ============================================================================
# 1. KONTENERY
# ============================================================================
echo -e "${BLUE}[1/4] Sprawdzanie kontenerów...${NC}"
check_container "doc-converter" || CONV_DOWN=1
check_container "whisper" || WHISPER_DOWN=1
check_container "ollama" || OLLAMA_DOWN=1
check_container "pyannote" || PYANNOTE_DOWN=1
echo ""

# ============================================================================
# 2. HEALTH CHECKS
# ============================================================================
echo -e "${BLUE}[2/4] Sprawdzanie health endpoints...${NC}"
check_service "Doc-converter" "http://localhost:8502/_stcore/health" || CONV_FAIL=1
check_service "Whisper" "http://localhost:9000/" || WHISPER_FAIL=1
check_service "Ollama" "http://localhost:11434/api/tags" || OLLAMA_FAIL=1

# Pyannote (opcjonalny)
if check_service "Pyannote" "http://localhost:8001/health"; then
    PYANNOTE_OK=1
else
    echo -e "${YELLOW}⚠️  Pyannote${NC} - Opcjonalny (rozpoznawanie mówców)"
fi
echo ""

# ============================================================================
# 3. PORTY
# ============================================================================
echo -e "${BLUE}[3/4] Sprawdzanie portów...${NC}"
netstat -tuln 2>/dev/null | grep -q ":8502" && echo -e "${GREEN}✅ Port 8502${NC} (doc-converter)" || echo -e "${RED}❌ Port 8502${NC} nie nasłuchuje"
netstat -tuln 2>/dev/null | grep -q ":9000" && echo -e "${GREEN}✅ Port 9000${NC} (whisper)" || echo -e "${RED}❌ Port 9000${NC} nie nasłuchuje"
netstat -tuln 2>/dev/null | grep -q ":11434" && echo -e "${GREEN}✅ Port 11434${NC} (ollama)" || echo -e "${RED}❌ Port 11434${NC} nie nasłuchuje"
netstat -tuln 2>/dev/null | grep -q ":8001" && echo -e "${GREEN}✅ Port 8001${NC} (pyannote)" || echo -e "${YELLOW}⚠️  Port 8001${NC} (pyannote opcjonalny)"
echo ""

# ============================================================================
# 4. OLLAMA MODELS
# ============================================================================
echo -e "${BLUE}[4/4] Sprawdzanie modeli Ollama...${NC}"
MODELS=$(docker compose exec -T ollama ollama list 2>/dev/null | tail -n +2 | wc -l)
if [ "$MODELS" -gt 0 ]; then
    echo -e "${GREEN}✅ Ollama models:${NC} $MODELS modeli zainstalowanych"
    docker compose exec -T ollama ollama list | tail -n +2 | head -5
else
    echo -e "${YELLOW}⚠️  Brak modeli Ollama${NC}"
    echo -e "   Pobierz model: ${BLUE}docker compose exec ollama ollama pull llama2${NC}"
fi
echo ""

# ============================================================================
# PODSUMOWANIE
# ============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        PODSUMOWANIE                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

if [ -z "${CONV_FAIL:-}" ] && [ -z "${WHISPER_FAIL:-}" ] && [ -z "${OLLAMA_FAIL:-}" ]; then
    echo -e "${GREEN}✅ DOC-CONVERTER GOTOWY DO TESTOWANIA!${NC}"
    echo ""
    echo -e "${GREEN}🌐 Otwórz w przeglądarce:${NC} ${BLUE}http://localhost:8502${NC}"
    echo ""

    if [ -n "${PYANNOTE_OK:-}" ]; then
        echo -e "${GREEN}✅ Pyannote działa${NC} - rozpoznawanie mówców dostępne"
    else
        echo -e "${YELLOW}⚠️  Pyannote nie działa${NC} - tylko transkrypcja bez podziału na mówców"
        echo -e "   Sprawdź: ${BLUE}docker compose logs pyannote${NC}"
    fi

else
    echo -e "${RED}❌ PROBLEMY WYKRYTE!${NC}"
    echo ""

    if [ -n "${CONV_FAIL:-}" ]; then
        echo -e "${RED}• Doc-converter nie odpowiada${NC}"
        echo -e "  Fix: ${BLUE}docker compose restart doc-converter${NC}"
        echo -e "  Logi: ${BLUE}docker compose logs doc-converter${NC}"
    fi

    if [ -n "${WHISPER_FAIL:-}" ]; then
        echo -e "${RED}• Whisper nie odpowiada${NC}"
        echo -e "  Fix: ${BLUE}docker compose restart whisper${NC}"
        echo -e "  Logi: ${BLUE}docker compose logs whisper${NC}"
    fi

    if [ -n "${OLLAMA_FAIL:-}" ]; then
        echo -e "${RED}• Ollama nie odpowiada${NC}"
        echo -e "  Fix: ${BLUE}docker compose restart ollama${NC}"
        echo -e "  Logi: ${BLUE}docker compose logs ollama${NC}"
    fi
fi

echo ""
echo -e "${BLUE}📚 Pełna dokumentacja:${NC} DOC_CONVERTER_TESTING.md"
echo ""
