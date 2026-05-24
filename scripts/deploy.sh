#!/usr/bin/env bash
# ┌──────────────────────────────────────────────────────┐
# │  BIST Platform — VDS Otomatik Deployment Script      │
# │  Kullanim: bash scripts/deploy.sh                    │
# │  Sifirdan bir Ubuntu VDS'te calistirilir.            │
# └──────────────────────────────────────────────────────┘
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[X]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[i]${NC} $1"; }

# ─── Config ────────────────────────────────────────────
APP_DIR="${APP_DIR:-/opt/bist-platform}"
GIT_REPO="${GIT_REPO:-}"  # git clone URL (bos birakilirsa mevcut dizin kullanilir)

# ─── Root Check ────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    err "Bu script root olarak calistirilmalidir: sudo bash scripts/deploy.sh"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  BIST Platform — VDS Otomatik Deployment"
echo "  Tarih: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════"
echo ""

# ─── Step 1: Sistem Paketleri ─────────────────────────
log "1/8 Sistem paketleri kontrol ediliyor..."
export DEBIAN_FRONTEND=noninteractive
apt update -qq && apt upgrade -y -qq

for pkg in curl git ufw htop; do
    if ! command -v "$pkg" &> /dev/null && ! dpkg -l "$pkg" &> /dev/null 2>&1; then
        apt install -y -qq "$pkg"
    fi
done

# ─── Step 2: Firewall ──────────────────────────────────
log "2/8 Firewall (UFW) yapilandiriliyor..."
ufw --force reset > /dev/null 2>&1
ufw default deny incoming > /dev/null
ufw default allow outgoing > /dev/null
ufw allow 22/tcp > /dev/null
ufw allow 80/tcp > /dev/null
ufw allow 443/tcp > /dev/null
ufw --force enable > /dev/null
log "  UFW aktif: SSH(22), HTTP(80), HTTPS(443)"

# ─── Step 3: Docker ────────────────────────────────────
log "3/8 Docker kontrol ediliyor..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker --now
fi
log "  Docker: $(docker --version)"

# ─── Step 4: Proje Dizini ──────────────────────────────
log "4/8 Proje dizini hazirlaniyor..."
if [ -n "$GIT_REPO" ]; then
    if [ -d "$APP_DIR" ]; then
        warn "  $APP_DIR zaten var, git pull yapiliyor..."
        cd "$APP_DIR" && git pull
    else
        mkdir -p "$APP_DIR"
        git clone "$GIT_REPO" "$APP_DIR"
    fi
elif [ ! -d "$APP_DIR" ]; then
    err "APP_DIR=$APP_DIR bulunamadi ve GIT_REPO bos. Projeyi manuel kopyalayin."
fi
cd "$APP_DIR"

# ─── Step 5: .env Kontrolü ─────────────────────────────
log "5/8 .env dosyasi kontrol ediliyor..."
if [ ! -f ".env" ]; then
    if [ -f ".env.production.example" ]; then
        cp .env.production.example .env
        warn "  .env.production.example -> .env kopyalandi"
        warn "  LUTFEN .env dosyasini duzenleyin! (nano .env)"
        warn "  BIST_SECRET_KEY ve POSTGRES_PASSWORD mutlaka degistirilmeli!"
    elif [ -f ".env.example" ]; then
        cp .env.example .env
        warn "  .env.example -> .env kopyalandi"
        warn "  LUTFEN .env dosyasini duzenleyin!"
    fi
fi

# Temel kontroller
if grep -q "change-me-before-production\|BURAYA_GUVENLI_SIFRE_YAZ\|BURAYA_64_KARAKTERLIK" .env 2>/dev/null; then
    err "  .env dosyasinda varsayilan sifreler/anahtarlar bulundu! Lutfen duzenleyin."
fi

# ─── Step 6: Docker Compose ────────────────────────────
log "6/8 Docker Compose build ve baslatiliyor..."
docker compose down --remove-orphans 2>/dev/null || true
docker compose build --pull
docker compose up -d

log "  Servisler baslatildi, 10 saniye bekleniyor..."
sleep 10

# ─── Step 7: Database Migration ────────────────────────
log "7/8 Veritabani hazirlaniyor..."
docker compose run --rm backend python -m alembic upgrade head 2>/dev/null || \
    warn "  Migration basarisiz (veritabani daha hazir degil, 15 sn bekleniyor...)"
sleep 15
docker compose run --rm backend python -m alembic upgrade head

# Seed (sadece ilk kurulumda)
docker compose run --rm backend python scripts/seed_symbols.py 2>/dev/null || true
docker compose run --rm backend python scripts/seed_indices.py 2>/dev/null || true

# ─── Step 8: Health Check ──────────────────────────────
log "8/8 Health check yapiliyor..."
sleep 2
if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
    log "  Backend saglikli!"
else
    warn "  Backend henuz yanit vermiyor. docker compose logs backend ile kontrol edin."
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo -e "  ${GREEN}Deployment tamamlandi!${NC}"
echo ""
echo "  Sirakiler:"
echo "  1. docker compose ps (servis durumu)"
echo "  2. bash scripts/healthcheck.sh (saglik kontrolu)"
echo "  3. docker compose logs -f (log takibi)"
echo "  4. Veri cekme: docker compose run --rm backend python scripts/run_pipeline.py \\"
echo "       --source yfinance --timeframe 1d --horizons short,medium,long"
echo "═══════════════════════════════════════════════════"
