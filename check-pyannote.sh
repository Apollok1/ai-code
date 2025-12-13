#!/bin/bash
# Diagnostyka pyannote i naprawy

echo "=== PYANNOTE DIAGNOSTYKA ==="

cd /home/michal/moj-asystent-ai

echo "1. Status kontenera:"
docker compose ps pyannote

echo ""
echo "2. Logi pyannote (ostatnie 50 linii):"
docker compose logs pyannote --tail=50

echo ""
echo "3. Health check:"
curl -s http://localhost:8001/health

echo ""
echo "4. Sprawdź czy HF_TOKEN jest ustawiony:"
docker compose exec pyannote env | grep HF_TOKEN

echo ""
echo "=== CO ZROBIĆ ==="
echo "Jeśli widzisz błąd 'HF_TOKEN' - dodaj do .env:"
echo "HF_TOKEN=twoj_token_z_huggingface"
echo ""
echo "Jeśli timeout - zwiększ w docker-compose.yml:"
echo "  start_period: 300s (5 minut)"
