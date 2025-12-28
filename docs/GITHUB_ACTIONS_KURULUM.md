# BIST100 GitHub Actions Kurulum Rehberi

## 🚀 Hızlı Kurulum (5 Dakika)

### Adım 1: GitHub Repository Oluşturma

1. https://github.com/new adresine gidin
2. Repository adı: `bist100-trading` (veya istediğiniz)
3. **Private** seçin (kodunuz gizli kalır)
4. "Create repository" tıklayın

### Adım 2: Secrets Ekleme (E-posta Bilgileri)

1. Repository sayfanızda: **Settings** → **Secrets and variables** → **Actions**
2. **"New repository secret"** tıklayın ve şunları ekleyin:

| Secret Adı | Değer |
|------------|-------|
| `BIST_EMAIL_SENDER` | `dundarmd89@gmail.com` |
| `BIST_EMAIL_PASSWORD` | `fgvbngtfbgnxolkb` |
| `BIST_EMAIL_RECIPIENTS` | `dundarmd89@gmail.com` |

⚠️ **Önemli**: Birden fazla alıcı için virgülle ayırın: `email1@x.com,email2@y.com`

### Adım 3: Kodu GitHub'a Push Etme

PowerShell'de şu komutları çalıştırın:

```powershell
cd c:\Users\dunda\OneDrive\Masaüstü\kod\bist

# Git başlat
git init

# Tüm dosyaları ekle
git add .

# İlk commit
git commit -m "Initial commit: BIST100 Trading System"

# GitHub remote ekle (kendi repo URL'inizi yazın)
git remote add origin https://github.com/KULLANICI_ADINIZ/bist100-trading.git

# Push
git branch -M main
git push -u origin main
```

### Adım 4: Actions'ı Kontrol Etme

1. GitHub'da repository'nize gidin
2. **Actions** sekmesine tıklayın
3. "BIST100 Daily Scan" workflow'u göreceksiniz

### Adım 5: Manuel Test

1. **Actions** → **BIST100 Daily Scan** → **Run workflow**
2. "Run workflow" butonuna tıklayın
3. Çalışmasını izleyin (3-5 dakika)
4. E-posta geldi mi kontrol edin!

---

## 📅 Çalışma Zamanlaması

- **Her gün saat 09:00** (Türkiye saati)
- **Sadece hafta içi** (Pazartesi-Cuma)
- Borsa kapalı günlerde veri olmaz ama hata vermez

---

## ❓ Sorun Giderme

### "Workflow not running"
- Actions sekmesinde workflow'un enabled olduğundan emin olun
- İlk push'tan sonra otomatik aktif olur

### "Email not sent"
- Secrets doğru girildi mi kontrol edin
- Secret isimlerinde typo var mı?

### "Model not found"
- İlk çalışmada model otomatik eğitilir
- 3-5 dakika bekleyin

---

## 🔧 Özelleştirme

`.github/workflows/daily_scan.yml` dosyasında:

- **Saat değiştirme**: `cron: '0 6 * * 1-5'` → `0 6` = 06:00 UTC = 09:00 Türkiye
- **Hafta sonu dahil**: `1-5` → `*` yapın
- **Farklı saat**: https://crontab.guru/ kullanın
