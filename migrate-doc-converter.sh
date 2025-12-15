#!/bin/bash
#
# Doc-Converter Migration Script
# Safely migrates from monolithic to refactored version
#
# Usage: ./migrate-doc-converter.sh
#

set -e  # Exit on error

echo "============================================================================"
echo "DOC-CONVERTER MIGRATION TO REFACTORED VERSION"
echo "============================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "docker-compose.direct-mount.yml" ]; then
    echo -e "${RED}ERROR: Run this script from moj-asystent-ai directory${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1/5: Pulling latest changes from ai-code...${NC}"
cd ${AI_CODE_PATH:-/home/michal/ai-code}
git pull
cd -

echo -e "${GREEN}✓ Code updated${NC}"
echo ""

echo -e "${YELLOW}Step 2/5: Stopping old doc-converter...${NC}"
docker compose down doc-converter

echo -e "${GREEN}✓ Old version stopped${NC}"
echo ""

echo -e "${YELLOW}Step 3/5: Building refactored version (this may take a few minutes)...${NC}"
docker compose build doc-converter --no-cache

echo -e "${GREEN}✓ Build complete${NC}"
echo ""

echo -e "${YELLOW}Step 4/5: Starting refactored doc-converter...${NC}"
docker compose up -d doc-converter

echo -e "${GREEN}✓ Service started${NC}"
echo ""

echo -e "${YELLOW}Step 5/5: Checking logs for errors...${NC}"
echo ""
echo "Waiting 10 seconds for service to initialize..."
sleep 10

echo ""
echo "Last 30 log lines:"
echo "---"
docker compose logs --tail=30 doc-converter

echo ""
echo "============================================================================"

# Check for import errors
if docker compose logs doc-converter | grep -q "ModuleNotFoundError"; then
    echo -e "${RED}✗ MIGRATION FAILED - Import errors detected${NC}"
    echo ""
    echo "Checking for specific errors:"
    docker compose logs doc-converter | grep -i "error" | tail -10
    echo ""
    echo "To rollback:"
    echo "  1. Stop refactored version: docker compose down doc-converter"
    echo "  2. Restore old docker-compose.yml from git"
    echo "  3. Rebuild and restart: docker compose up -d doc-converter"
    exit 1
fi

# Check if service is healthy
if ! docker compose ps doc-converter | grep -q "Up"; then
    echo -e "${RED}✗ MIGRATION FAILED - Service not running${NC}"
    echo ""
    echo "Full logs:"
    docker compose logs doc-converter
    exit 1
fi

echo -e "${GREEN}✓ MIGRATION SUCCESSFUL!${NC}"
echo ""
echo "Doc-Converter is now running the refactored version"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:8502 in your browser"
echo "  2. Test file conversion (PDF, audio, images)"
echo "  3. Monitor logs: docker compose logs -f doc-converter"
echo ""
echo "If you encounter issues:"
echo "  - Check logs: docker compose logs -f doc-converter"
echo "  - Report in GitHub issues"
echo ""
echo "============================================================================"
