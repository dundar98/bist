# 🚀 BIST Sinyal Platformu — VDS Deployment Rehberi

Bu rehber, daha önce hiç sunucu yönetimi yapmamış birinin bile bu projeyi
sıfırdan bir VDS (Virtual Dedicated Server) üzerinde çalışır hale getirebilmesi
için adım adım yazılmıştır.

---

## İçindekiler

1. [VDS Nedir, Nasıl Alınır?](#1-vds-nedir-nasıl-alınır)
2. [Sunucuya Bağlanma ve İlk Kurulum](#2-sunucuya-bağlanma-ve-i̇lk-kurulum)
3. [Projeyi Sunucuya Taşıma](#3-projeyi-sunucuya-taşıma)
4. [Production Yapılandırması](#4-production-yapılandırması)
5. [Sistemi Başlatma](#5-sistemi-başlatma)
6. [Domain ve HTTPS (SSL)](#6-domain-ve-https-ssl)
7. [Veritabanı İlk Yükleme (Seed)](#7-veritabanı-i̇lk-yükleme-seed)
8. [Günlük Otomatik İşler](#8-günlük-otomatik-i̇şler)
9. [Yedekleme (Backup)](#9-yedekleme-backup)
10. [Güncelleme (Update)](#10-güncelleme-update)
11. [İzleme ve Sorun Giderme](#11-i̇zleme-ve-sorun-giderme)
12. [Güvenlik Kontrol Listesi](#12-güvenlik-kontrol-listesi)

---

## 1. VDS Nedir, Nasıl Alınır?

**VDS (Virtual Dedicated Server)**, sadece senin kullanımına ayrılmış sanal bir
bilgisayardır. 7/24 çalışır, internet erişimi vardır ve üzerinde istediğin
yazılımı çalıştırabilirsin.

### Önerilen VDS Özellikleri

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB |
| Disk | 30 GB SSD | 50+ GB SSD |
| İşletim Sistemi | Ubuntu 24.04 LTS | Ubuntu 24.04 LTS |
| Trafik | 2 TB | Sınırsız |

### Nereden Alınır? (Türkiye'den popüler seçenekler)

| Sağlayıcı | Başlangıç Fiyatı | Aylık (yaklaşık) |
|-----------|-----------------|-------------------|
| **Hetzner** | CX22 (2 vCPU, 4 GB) | ~€4.50 |
| **Netcup** | RS 1000 (4 vCPU, 8 GB) | ~€5.50 |
| **Contabo** | Cloud VPS S (4 vCPU, 8 GB) | ~€5.99 |
| **DigitalOcean** | Basic (2 vCPU, 4 GB) | ~$24 |
| **Vultr** | High Frequency (2 vCPU, 4 GB) | ~$24 |

> 💡 **Öneri:** Hetzner veya Contabo fiyat/performans açısından en iyisidir.
> Hetzner'ın Almanya/Nürnberg lokasyonu Türkiye'ye ping olarak da iyidir (~50ms).

### Satın Alma Adımları (Hetzner örneği)

1. [hetzner.com/cloud](https://www.hetzner.com/cloud) adresine git
2. "Add Server" tıklayın
3. İşletim sistemi: **Ubuntu 24.04**
4. Plan: **CX22** (başlangıç için yeterli, sonra yükseltebilirsiniz)
5. Lokasyon: **Nürnberg** (Türkiye'ye en yakın)
6. SSH Key ekleyin (yoksa şimdilik atlayın, şifre ile bağlanırsınız)
7. Sunucu adı: `bist-platform` (veya istediğiniz)
8. "Create & Buy Now" tıklayın
9. Size root şifresi e-posta ile gelecek

---

## 2. Sunucuya Bağlanma ve İlk Kurulum

### Windows'tan Bağlanma

PowerShell'i açın ve şu komutu yazın:

```powershell
ssh root@SUNUCU_IP_ADRESI
```

İlk bağlanmada "Are you sure you want to continue connecting?" sorusuna `yes`
yazın. Şifrenizi girin (Hetzner size e-posta ile gönderdi).

### İlk Güvenlik Ayarları

Bağlandıktan sonra sırasıyla şu komutları çalıştırın:

```bash
# 1. Sistemi güncelle
apt update && apt upgrade -y

# 2. Firewall kur (UFW) ve yapılandır
apt install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP (web)
ufw allow 443/tcp     # HTTPS (web)
ufw --force enable
ufw status verbose    # Kontrol et

# 3. Docker kur
curl -fsSL https://get.docker.com | sh
systemctl enable docker --now

# 4. Kullanıcını docker grubuna ekle (root'tan çalışmamak için)
usermod -aG docker root

# 5. Docker Compose (standalone plugin)
apt install -y docker-compose-v2
# Alternatif olarak plugin:
# mkdir -p /usr/local/lib/docker/cli-plugins
# curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" -o /usr/local/lib/docker/cli-plugins/docker-compose
# chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# 6. Git kur
apt install -y git

# 7. Sistem saatini Türkiye'ye ayarla
timedatectl set-timezone Europe/Istanbul
timedatectl status

# 8. Swap dosyası oluştur (RAM + 2GB önerilir)
fallocate -l 6G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

> ✅ Bu noktada sunucun güvenli, Docker ve Git yüklü durumda.

---

## 3. Projeyi Sunucuya Taşıma

### Yöntem A: Git ile Çekme (Önerilen)

```bash
# Sunucuda proje dizinini oluştur
mkdir -p /opt/bist-platform
cd /opt/bist-platform

# GitHub'dan projeyi çek (private repo ise token gerekir)
# Public repo:
git clone https://github.com/KULLANICI_ADIN/bist100-trading.git .

# VEYA private repo için GitHub Personal Access Token ile:
# git clone https://TOKEN@github.com/KULLANICI_ADIN/bist100-trading.git .
```

### Yöntem B: SCP ile Dosya Kopyalama (GitHub yoksa)

Kendi bilgisayarında PowerShell'de:

```powershell
# Proje dizinindeyken:
scp -r . root@SUNUCU_IP:/opt/bist-platform/
```

> ⚠️ Bu yöntem `node_modules/`, `__pycache__/`, `.git/` gibi gereksiz
> dosyaları da kopyalar. Git ile klonlamak daha temizdir.

---

## 4. Production Yapılandırması

### 4.1. `.env` Dosyasını Hazırlama

Sunucuda, proje dizininde:

```bash
cd /opt/bist-platform

# Örnek .env dosyasını kopyala
cp .env.example .env

# Düzenle (nano ile)
nano .env
```

**Production `.env` içeriği (BUNLARI KESİNLİKLE DEĞİŞTİR):**

```ini
BIST_APP_NAME="BIST Signal Platform"
BIST_ENVIRONMENT=production
BIST_DATABASE_URL=postgresql+psycopg://bist:GUVENLI_BIR_SIFRE@postgres:5432/bist
BIST_CONTAINER_DATABASE_URL=postgresql+psycopg://bist:GUVENLI_BIR_SIFRE@postgres:5432/bist
BIST_SECRET_KEY=RANDOM_64_KARAKTERLIK_GUVENLI_BIR_ANAHTAR
BIST_ACCESS_TOKEN_EXPIRE_MINUTES=60
BIST_CORS_ORIGINS=https://SENIN_DOMAININ.com,https://www.SENIN_DOMAININ.com
BIST_ALERT_WEBHOOK_URL=

POSTGRES_DB=bist
POSTGRES_USER=bist
POSTGRES_PASSWORD=GUVENLI_BIR_SIFRE
APP_DOMAIN=SENIN_DOMAININ.com
ACME_EMAIL=SENIN_EMAILIN@gmail.com
BIST_BACKUP_DIR=backups
```

> 🔐 **ÖNEMLİ:** `BIST_SECRET_KEY` için güçlü bir rastgele değer üret:
> ```bash
> python3 -c "import secrets; print(secrets.token_urlsafe(48))"
> ```

### 4.2. Domain DNS Ayarları

Eğer bir domain adın varsa (yoksa 6. bölüme geç):

1. Domain yönetim panelinde (Cloudflare, Namesilo, vb.) DNS kayıtlarını ekle:
   - **A kaydı:** `@` → `SUNUCU_IP` 
   - **A kaydı:** `www` → `SUNUCU_IP`
   - TTL: Auto veya 3600

2. DNS'in yayılması 5-60 dakika sürebilir. Kontrol için:
   ```bash
   nslookup SENIN_DOMAININ.com
   ```

---

## 5. Sistemi Başlatma

Proje dizininde tüm servisleri başlat:

```bash
cd /opt/bist-platform
docker compose up -d
```

Bu komut şunları yapar:
- PostgreSQL, backend, frontend, worker, proxy (Caddy) container'larını oluşturur
- İlk çalıştırmada imajları indirir (~2-3 dakika)
- `-d` parametresi arka planda çalıştırır

```bash
# Çalışan container'ları gör
docker compose ps

# Çıktı şuna benzer olmalı:
# NAME                    STATUS              PORTS
# repo-inspect-postgres-1  Up (healthy)       0.0.0.0:5432->5432/tcp
# repo-inspect-backend-1   Up                 0.0.0.0:8000->8000/tcp
# repo-inspect-frontend-1  Up                 0.0.0.0:3000->80/tcp
# repo-inspect-worker-1    Up
# repo-inspect-proxy-1     Up                 0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
```

### Hemen test et:

```bash
# Health check
curl http://localhost:8000/api/health

# Domain yoksa IP ile:
# Tarayıcıda: http://SUNUCU_IP
```

---

## 6. Domain ve HTTPS (SSL)

### Domain Yoksa (Sadece IP ile Kullanım)

`.env` dosyasında:
```ini
APP_DOMAIN=SUNUCU_IP
ACME_EMAIL=admin@example.com
```

⚠️ IP adresiyle HTTPS çalışmaz. Sadece HTTP kullanabilirsin. Caddy bu
durumda kendini düzgün yapılandıracaktır.

### Domain Varsa (HTTPS Otomatik)

Caddy, konfigürasyonda **hiçbir şey yapmana gerek kalmadan** Let's Encrypt
üzerinden ücretsiz SSL sertifikası alır.

Yapman gereken tek şey `.env` dosyasında:
```ini
APP_DOMAIN=sendomain.com
ACME_EMAIL=gercek@email.com
```

Ve Caddyfile zaten doğru yapılandırılmış. Container'ları yeniden başlat:

```bash
docker compose down
docker compose up -d
```

1-2 dakika içinde siten `https://sendomain.com` adresinde çalışır hale gelir.
Sertifika otomatik yenilenir.

---

## 7. Veritabanı İlk Yükleme (Seed)

Sistem ilk kez kurulduğunda veritabanı boştur. Aşağıdaki komutları sırayla
çalıştır:

```bash
cd /opt/bist-platform

# 1. Veritabanı tablolarını oluştur
docker compose run --rm backend python -m alembic upgrade head

# 2. BIST100 hisse listesini yükle
docker compose run --rm backend python scripts/seed_symbols.py

# 3. Endeks verilerini yükle (XU100, XU030, vb.)
docker compose run --rm backend python scripts/seed_indices.py

# 4. Endeks fiyat geçmişini yükle
docker compose run --rm backend python scripts/update_indices.py --lookback-days 365

# 5. İlk admin kullanıcısını oluştur
docker compose run --rm backend python scripts/create_user.py \
  --username admin \
  --email admin@sendomain.com \
  --password "GUCLU_BIR_SIFRE" \
  --role admin \
  --full-name "Admin User"

# 6. İlk fiyat verilerini çek (ilk çekim uzun sürer ~5-10 dk)
docker compose run --rm backend python scripts/update_prices.py \
  --source yfinance \
  --timeframe 1d \
  --limit 50 \
  --lookback-days 180

# 7. Feature hesapla
docker compose run --rm backend python scripts/update_features.py \
  --timeframe 1d \
  --limit 50 \
  --lookback-bars 260

# 8. İlk portföy/sinyal snapshot'ını oluştur
docker compose run --rm backend python scripts/build_portfolio.py \
  --timeframe 1d \
  --horizon medium \
  --max-positions 10 \
  --min-score 55
```

> ✅ Bu adımlardan sonra `https://sendomain.com/api/signals/latest` adresinde
> sinyalleri görmeye başlamalısın.

---

## 8. Günlük Otomatik İşler

Worker container'ı zaten 24 saatte bir pipeline'ı çalıştıracak şekilde
yapılandırılmıştır (`scripts/run_worker.py`).

### Worker'ın çalıştığını kontrol et:

```bash
# Worker loglarını gör
docker compose logs worker --tail=50

# Worker'ı yeniden başlat
docker compose restart worker
```

### Manuel pipeline çalıştırma (test için):

```bash
docker compose run --rm backend python scripts/run_pipeline.py \
  --source yfinance \
  --timeframe 1d \
  --horizons short,medium,long \
  --use-market-regime
```

### Cron ile alternatif zamanlama (daha hassas kontrol için)

Eğer worker yerine cron kullanmak istersen:

```bash
# Crontab'ı düzenle
crontab -e

# Aşağıdaki satırları ekle:
# Her iş günü 09:45 - Kısa vade fiyat güncelleme
45 9 * * 1-5 cd /opt/bist-platform && docker compose run --rm backend python scripts/run_pipeline.py --source yfinance --timeframe 1d --horizons short --use-market-regime >> /var/log/bist-cron.log 2>&1

# Her iş günü 18:30 - Tam günlük pipeline
30 18 * * 1-5 cd /opt/bist-platform && docker compose run --rm backend python scripts/run_pipeline.py --source yfinance --timeframe 1d --horizons short,medium,long --use-market-regime --train-ml >> /var/log/bist-cron.log 2>&1
```

---

## 9. Yedekleme (Backup)

### Otomatik Yedekleme (Önerilen)

```bash
# Backup script'ini çalıştırılabilir yap
chmod +x /opt/bist-platform/scripts/backup_database.py

# Crontab'a ekle (her gün 03:00)
crontab -e
```

```cron
# Günlük database yedeği (her gün saat 03:00)
0 3 * * * cd /opt/bist-platform && docker compose run --rm backend python scripts/backup_database.py >> /var/log/bist-backup.log 2>&1
```

### Manuel Yedekleme

```bash
cd /opt/bist-platform
docker compose run --rm backend python scripts/backup_database.py
```

### Yedeği Başka Bir Sunucuya Kopyalama

```bash
# Backup dizinindeki en son yedeği bul
ls -t backups/

# Kendi bilgisayarına çek (PowerShell'de)
scp root@SUNUCU_IP:/opt/bist-platform/backups/EN_SON_DOSYA ./
```

### Yedekten Geri Yükleme

```bash
# Container içinde
docker compose exec postgres pg_restore \
  -U bist -d bist -c backups/DOSYA_ADI.dump
```

---

## 10. Güncelleme (Update)

Yeni kod değişikliği yaptığında:

```bash
cd /opt/bist-platform

# 1. Son kodu çek
git pull

# 2. Container'ları yeniden build et ve başlat
docker compose up -d --build backend frontend worker

# 3. Veritabanı migration varsa çalıştır
docker compose run --rm backend python -m alembic upgrade head

# 4. Eski imajları temizle (disk tasarrufu)
docker system prune -f
```

---

## 11. İzleme ve Sorun Giderme

### Logları Görüntüleme

```bash
# Tüm servislerin son logları
docker compose logs --tail=100

# Sadece backend logları (canlı izle)
docker compose logs -f backend

# Sadece worker logları
docker compose logs worker --tail=50
```

### Health Check

```bash
# Backend sağlık kontrolü
curl http://localhost:8000/api/health

# Backend readiness (database bağlı mı?)
curl http://localhost:8000/api/ready

# Dashboard verisi
curl http://localhost:8000/api/dashboard/overview
```

### Container'ları Yönetme

```bash
# Tüm container'ları durdur
docker compose down

# Sadece belirli bir servisi yeniden başlat
docker compose restart backend

# Tüm servisleri durdurup yeniden başlat
docker compose down && docker compose up -d

# Container içinde komut çalıştır
docker compose exec backend python scripts/check_validator.py
```

### Yaygın Sorunlar

| Sorun | Çözüm |
|-------|-------|
| Container başlamıyor | `docker compose logs postgres` ile logları kontrol et |
| Database'e bağlanamıyor | `.env` dosyasındaki `POSTGRES_PASSWORD` doğru mu? |
| Caddy sertifika hatası | Domain DNS'inin sunucu IP'sine işaret ettiğinden emin ol |
| Worker çalışmıyor | `docker compose logs worker` loglarını kontrol et |
| Disk dolu | `docker system prune -a -f` ile temizlik yap |
| YFinance veri çekmiyor | BIST kapalı olabilir, hafta içi mesai saatlerinde dene |

### Kaynak Kullanımı Kontrolü

```bash
# Docker container'larının kaynak kullanımı
docker stats --no-stream

# Sunucu genel durumu
htop    # yoksa: apt install -y htop
df -h   # disk kullanımı
free -h # RAM kullanımı
```

---

## 12. Güvenlik Kontrol Listesi

Production'a çıkmadan önce her maddeyi kontrol et:

- [ ] `.env` dosyasındaki `BIST_SECRET_KEY` değiştirildi (rastgele 64 karakter)
- [ ] `.env` dosyasındaki `POSTGRES_PASSWORD` değiştirildi (güçlü şifre)
- [ ] Admin kullanıcısı oluşturuldu ve güçlü şifre verildi
- [ ] `.env` dosyası `.gitignore`'da (yanlışlıkla commit'lenmez)
- [ ] UFW firewall aktif ve sadece 22, 80, 443 portları açık
- [ ] SSH root girişi için şifre yerine SSH key kullanılıyor (öneri)
- [ ] Domain varsa HTTPS çalışıyor (Caddy otomatik halleder)
- [ ] Database yedeği otomatik alınıyor (cron ayarlandı)
- [ ] `5432` portu dış dünyaya kapalı (PostgreSQL sadece Docker ağından erişilir)
- [ ] Başarısız giriş denemelerine karşı fail2ban kurulu (opsiyonel)

### SSH Key ile Bağlanma (Şifreden Daha Güvenli)

Kendi bilgisayarında PowerShell'de:

```powershell
# SSH key oluştur (yoksa)
ssh-keygen -t ed25519 -C "email@example.com"

# Public key'i sunucuya kopyala
cat ~/.ssh/id_ed25519.pub | ssh root@SUNUCU_IP "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

Sunucuda `/etc/ssh/sshd_config` dosyasında şu satırı değiştir:
```
PasswordAuthentication no
```

```bash
systemctl restart sshd
```

> Artık sunucuya sadece SSH key ile bağlanılabilir, şifre ile giriş kapalıdır.

---

## 🎯 Hızlı Başlangıç Özeti (Tecrübeliler İçin)

```bash
# 1. Sunucuyu hazırla
apt update && apt upgrade -y && apt install -y ufw git
ufw allow 22,80,443/tcp && ufw --force enable
curl -fsSL https://get.docker.com | sh

# 2. Projeyi çek
cd /opt && git clone REPO_URL bist-platform && cd bist-platform

# 3. Yapılandır ve başlat
cp .env.example .env && nano .env  # Şifreleri değiştir!
docker compose up -d

# 4. Veritabanını hazırla
docker compose run --rm backend python -m alembic upgrade head
docker compose run --rm backend python scripts/seed_symbols.py
docker compose run --rm backend python scripts/seed_indices.py
docker compose run --rm backend python scripts/create_user.py --username admin --email a@a.com --password "GUCLU_SIFRE" --role admin

# 5. İlk veriyi çek
docker compose run --rm backend python scripts/run_pipeline.py --source yfinance --timeframe 1d --horizons short,medium,long --use-market-regime

# Bitti! https://SENIN_DOMAININ
```

---

## 📦 Sistem Mimarisi (Production)

```
İnternet
    │
    ▼
┌──────────────────────┐
│   Caddy (proxy)      │  ← Port 80/443 dış dünyaya açık
│   Otomatik HTTPS      │    Tek giriş noktası
└──────┬───────────────┘
       │
       ├──────────────► frontend (Nginx + React)  ← Port 80 (iç ağ)
       │
       └──────────────► backend (FastAPI)         ← Port 8000 (iç ağ)
                              │
                              ├──► PostgreSQL     ← Port 5432 (iç ağ)
                              │
                              └──► Worker         ← 24 saatte bir pipeline
```

---

*Son güncelleme: 24 Mayıs 2026*
*Hazırlayan: Deep Code (AI Assistant)*
