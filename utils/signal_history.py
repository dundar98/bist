import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class SignalHistoryTracker:
    """
    Tracks historical signals to provide 'memory' and performance analytics.
    Saves and loads signal history from a JSON file.
    """
    
    def __init__(self, history_file: str = "output/signal_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_signal(self, symbol: str, signal_type: str, price: float, probability: float, timeframe: str):
        """Record a new signal if it's not a duplicate of the last one for this symbol."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal": signal_type,
            "price": price,
            "probability": probability,
            "timeframe": timeframe
        }
        self.history.append(entry)
        self._persist()

    def _persist(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def get_previous_signals(self, symbol: str, limit: int = 3) -> List[Dict]:
        """Get last N signals for a specific symbol."""
        return [h for h in self.history if h['symbol'] == symbol][-limit:]

    def get_signal_performance(self, symbol: str, current_price: float) -> Optional[str]:
        """Calculates return if a previous BUY signal exists for this symbol."""
        last_buy = next((h for h in reversed(self.history) if h['symbol'] == symbol and h['signal'] == 'BUY'), None)
        if last_buy:
            entry_price = last_buy['price']
            ret = (current_price - entry_price) / entry_price
            date_str = datetime.fromisoformat(last_buy['timestamp']).strftime('%Y-%m-%d')
            return f"Önceki AL: {entry_price:.2f} TL ({date_str}), Mevcut Getiri: %{ret*100:+.1f}"
        return None
