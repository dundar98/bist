import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.bist100_validator import BIST100Validator

validator = BIST100Validator()
symbols = validator.get_all_symbols()

print(f"Total symbols in BIST100Validator: {len(symbols)}")
if symbols:
    print(f"First 10 symbols: {symbols[:10]}")
else:
    print("❌ NO SYMBOLS FOUND!")
