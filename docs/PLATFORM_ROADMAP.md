# BIST Signal Platform Roadmap

Bu belge mevcut araştırma prototipini database merkezli, seçici sinyal üreten ve web arayüzü olan bir platforma dönüştürmek için uygulanacak yolu özetler.

## Faz 1: Platform Altyapısı

- FastAPI uygulaması
- PostgreSQL bağlantısı
- SQLAlchemy modelleri
- Alembic migration altyapısı
- Sağlık ve readiness endpointleri
- BIST sembol seed scripti

### Yerel Başlatma

```bash
python -m alembic upgrade head
python scripts/seed_symbols.py
uvicorn app.main:app --reload
```

Sağlık kontrolü:

```text
GET /api/health
GET /api/ready
GET /api/symbols
```

## Faz 2: Veri Boru Hattı

- BIST sembollerini `symbols` tablosuna taşıma
- Günlük ve intraday OHLCV verisini incremental güncelleme
- Veri kalite kontrolleri
- Feature değerlerini kalıcı hale getirme

İlk fiyat güncelleme:

```bash
python scripts/update_prices.py --timeframe 1d --limit 10
```

Geliştirme ortamında ağ kullanmadan boru hattını denemek için:

```bash
python scripts/update_prices.py --source synthetic --timeframe 1d --symbols THYAO,GARAN --lookback-days 30
```

Fiyat API'si:

```text
GET /api/symbols/THYAO/prices?timeframe=1d&limit=100
```

Feature hesaplama:

```bash
python scripts/update_features.py --timeframe 1d --symbols THYAO,GARAN --lookback-bars 260
```

Feature API'si:

```text
GET /api/symbols/THYAO/features?timeframe=1d&limit=20
```

Tek komutla günlük pipeline:

```bash
python scripts/run_pipeline.py --source yfinance --timeframe 1d --limit 50 --lookback-days 180 --horizons short,medium,long
```

## Faz 3: Skor ve Sepet Motoru

```bash
python scripts/build_portfolio.py --timeframe 1d --max-positions 10 --min-score 55
```

```text
GET /api/signals/latest?timeframe=1d
GET /api/portfolios/latest?timeframe=1d
```

## Faz 3: Skor ve Sinyal Motoru

- Model skorunu teknik skorlarla birleştirme
- Hacim, trend, relatif güç ve risk skorları
- Final skoru 0-100 aralığında üretme
- Günlük ana aday sayısını 5-10 ile sınırlayan sepet motoru

## Faz 4: Model Eğitimi

- LightGBM/XGBoost baseline
- Mevcut LSTM/Transformer modellerini registry yapısına bağlama
- Walk-forward validation
- Model run ve prediction kayıtları

## Faz 5: Web Sitesi

- Next.js dashboard
- Kullanıcı kaydı ve giriş
- Admin kullanıcı/rol yönetimi
- Sinyal, sepet, performans ve hisse detay ekranları

## Faz 6: Performans Takibi

- Her sinyalin 1/3/5/10/20 günlük sonucunu ölçme
- Stop/target gerçekleşme takibi
- Strateji ve vade bazlı başarı metrikleri
