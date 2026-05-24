#!/usr/bin/env bash
# ┌──────────────────────────────────────────────┐
# │  BIST Platform — Health Check Script         │
# │  Kullanim: bash scripts/healthcheck.sh       │
# └──────────────────────────────────────────────┘
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

check() {
    local label="$1"
    local url="$2"
    local expected_code="${3:-200}"

    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null || echo "000")
    if [ "$status" = "$expected_code" ]; then
        echo -e "  ${GREEN}[✓]${NC} $label (HTTP $status)"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}[✗]${NC} $label (beklenen: $expected_code, alınan: $status)"
        FAIL=$((FAIL + 1))
    fi
}

API="${1:-http://localhost:8000}"

echo ""
echo "═══════════════════════════════════════════"
echo "  BIST Platform Health Check"
echo "  Hedef: $API"
echo "  Tarih: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════"
echo ""

# API kontrolleri
echo "── API Endpoint Kontrolleri ──"
check "Health"              "$API/api/health"
check "Readiness"           "$API/api/ready"
check "Symbols"             "$API/api/symbols"
check "Dashboard"           "$API/api/dashboard/overview?timeframe=1d"
check "Market Radar"        "$API/api/market/radar?timeframe=1d"
check "Latest Signals"      "$API/api/signals/latest?limit=5"
check "Latest Portfolios"   "$API/api/portfolios/latest?timeframe=1d"
check "API Docs"            "$API/docs"

# Container kontrolleri
echo ""
echo "── Docker Container Durumu ──"
if command -v docker &> /dev/null && docker compose ps &> /dev/null 2>&1; then
    docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || \
        echo -e "  ${YELLOW}[!]${NC} docker compose ps calismadi"
else
    echo -e "  ${YELLOW}[!]${NC} Docker bulunamadi veya compose dosyasi yok"
fi

# Disk ve RAM
echo ""
echo "── Sistem Kaynaklari ──"
echo "  Disk: $(df -h / | awk 'NR==2{print $3 " / " $2 " (" $5 ")"}')"
echo "  RAM:  $(free -h | awk 'NR==2{print $3 " / " $2}')"
echo "  Swap: $(free -h | awk 'NR==3{print $3 " / " $2}')"

echo ""
echo "═══════════════════════════════════════════"
echo -e "  Sonuc: ${GREEN}$PASS basarili${NC}, ${RED}$FAIL basarisiz${NC}"
echo "═══════════════════════════════════════════"

exit $FAIL
