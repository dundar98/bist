"""
News Sentiment Analysis Module.

Fetches news from RSS feeds and analyzes sentiment using a pre-trained BERT model.
"""

import logging
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Optional imports
try:
    import feedparser
except ImportError:
    feedparser = None

@dataclass
class NewsItem:
    title: str
    link: str
    published: datetime
    source: str
    sentiment_score: float = 0.0  # -1.0 to 1.0
    sentiment_label: str = "NEUTRAL" # POSITIVE, NEGATIVE, NEUTRAL


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment for BIST stocks using Hugging Face Transformers.
    """
    
    def __init__(self, model_name: str = "savasy/bert-base-turkish-sentiment-cased"):
        """
        Initialize analyzer.
        
        Args:
            model_name: Hugging Face model ID for Turkish sentiment
        """
        self.model_name = model_name
        self.pipeline = None
        self._feeds = [
            # KAP RSS (Genel) - Example
            "https://www.kap.org.tr/tr/rss/bildirim",
            # Investing.com TR - Example
            "https://tr.investing.com/rss/news_25.rss", # Stock market news
        ]
        
    def _load_model(self):
        """Lazy load the model."""
        if self.pipeline is not None:
            return

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.pipeline = pipeline(
                "sentiment-analysis", 
                model=model, 
                tokenizer=tokenizer,
                device=-1 # CPU
            )
        except ImportError:
            logger.error("Transformers library not installed. Sentiment analysis disabled.")
            self.pipeline = None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.pipeline = None

    def fetch_news(self, symbol: str, lookback_days: int = 2) -> List[NewsItem]:
        """
        Fetch news for a specific symbol from feeds.
        """
        # Note: Real implementation would need a more sophisticated scraper 
        # or a targeted RSS feed for the specific stock.
        # For this logic, we will mock the fetching based on keywords in general feeds
        # or return a few dummy results if no network access.
        
        news_items = []
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Simple keyword filtering
        keywords = [symbol, f"{symbol} Hissesi", "BIST"]
        
        if feedparser is None:
            logger.warning("feedparser not installed. Skipping news fetch.")
            return []
            
        for feed_url in self._feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Parse date
                    try:
                        dt = datetime(*entry.published_parsed[:6])
                    except:
                        dt = datetime.now()
                        
                    if dt < cutoff_date:
                        continue
                        
                    # Check keywords
                    content = f"{entry.title} {entry.description if 'description' in entry else ''}"
                    if any(k in content.upper() for k in keywords):
                        news_items.append(NewsItem(
                            title=entry.title,
                            link=entry.link,
                            published=dt,
                            source=feed.feed.title if 'title' in feed.feed else "News"
                        ))
            except Exception:
                continue
                
        return news_items

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text.
        Returns score between -1.0 (Negative) and 1.0 (Positive).
        """
        self._load_model()
        
        if self.pipeline is None:
            return 0.0
            
        try:
            # Truncate text to 512 tokens (approx chars)
            result = self.pipeline(text[:512])[0]
            
            label = result['label']
            score = result['score']
            
            # Map labels to score
            # savasy/bert-base-turkish-sentiment-cased uses: positive, negative
            if label.lower() == 'positive':
                return score
            elif label.lower() == 'negative':
                return -score
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0
            
    def get_stock_sentiment(self, symbol: str) -> float:
        """
        Get aggregated sentiment score for a stock.
        """
        news = self.fetch_news(symbol)
        if not news:
            return 0.0
            
        scores = []
        for item in news:
            score = self.analyze_sentiment(item.title)
            scores.append(score)
            
        return sum(scores) / len(scores) if scores else 0.0
