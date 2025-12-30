import yfinance as yf
from datetime import date, timedelta

symbol = "THYAO.IS"
end = date.today()
start = end - timedelta(days=30)

print(f"Fetching 1h data for {symbol} from {start} to {end}...")
ticker = yf.Ticker(symbol)
df = ticker.history(start=start.isoformat(), end=end.isoformat(), interval="1h")

if df.empty:
    print("❌ No data found with start/end!")
    print("Trying without dates (last 30d period)...")
    df = ticker.history(period="1mo", interval="1h")

print(f"Result: {len(df)} rows")
if not df.empty:
    print(df.head())
else:
    print("❌ Still no data!")
