#!/usr/bin/env python3
"""
Improved News Scraper with free alternative sources.
Uses multiple free crypto news sources as fallbacks.
"""

import json
import logging
import requests
import feedparser
from typing import List, Dict, Any
from datetime import datetime, timezone
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class ImprovedNewsScraper:
    """
    Multi-source crypto news scraper using free sources.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cache_dir = Path("cache/news")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Free crypto news RSS sources (updated URLs for 2025)
        self.free_sources = {
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'decrypt': 'https://decrypt.co/feed',
            'bitcoinmagazine': 'https://bitcoinmagazine.com/feed',
            'cryptoslate': 'https://cryptoslate.com/feed/',
            'cryptopotato': 'https://cryptopotato.com/feed/',
            'coingape': 'https://coingape.com/feed/',
            'ambcrypto': 'https://ambcrypto.com/feed/',
            'cryptonews': 'https://cryptonews.com/news/feed/',
            'beincrypto': 'https://beincrypto.com/feed/'
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
    def fetch_from_rss(self, source_name: str, url: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """Fetch news from RSS feed."""
        try:
            self.logger.info(f"Fetching RSS from {source_name}: {url}")
            
            feed = feedparser.parse(url)
            if not feed.entries:
                self.logger.warning(f"No entries found in RSS feed for {source_name}")
                return []
            
            news_items = []
            for entry in feed.entries[:max_items]:
                # Parse timestamp
                published_time = getattr(entry, 'published_parsed', None)
                if published_time:
                    timestamp = datetime(*published_time[:6], tzinfo=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)
                
                news_item = {
                    'title': getattr(entry, 'title', 'No title'),
                    'url': getattr(entry, 'link', ''),
                    'source': source_name,
                    'published_at': timestamp.isoformat(),
                    'summary': getattr(entry, 'summary', '')[:200],
                    'content': getattr(entry, 'title', '') + ' ' + getattr(entry, 'summary', '')
                }
                news_items.append(news_item)
            
            self.logger.info(f"Fetched {len(news_items)} items from {source_name}")
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error fetching RSS from {source_name}: {e}")
            return []
    
    def fetch_crypto_news(self, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch crypto news from multiple free sources.
        """
        all_news = []
        items_per_source = max(1, limit // len(self.free_sources))
        
        for source_name, rss_url in self.free_sources.items():
            try:
                source_news = self.fetch_from_rss(source_name, rss_url, items_per_source)
                all_news.extend(source_news)
                
                if len(all_news) >= limit:
                    break
                    
                # Small delay between requests
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error processing source {source_name}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        all_news.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        
        # Return limited results
        result = all_news[:limit]
        self.logger.info(f"Total news items fetched: {len(result)} from {len([s for s in self.free_sources.keys()])} sources")
        
        return result

def test_improved_news_scraper():
    """Test the improved news scraper."""
    print("🧪 Testing Improved News Scraper...")
    
    scraper = ImprovedNewsScraper()
    news_items = scraper.fetch_crypto_news(limit=15)
    
    print(f"📰 Fetched {len(news_items)} news items")
    
    if news_items:
        print("\n📊 Sample news items:")
        for i, item in enumerate(news_items[:5]):
            print(f"  {i+1}. [{item['source']}] {item['title'][:60]}...")
            print(f"     URL: {item['url']}")
            print(f"     Published: {item['published_at']}")
            print()
    
    return len(news_items) > 0

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(name)s - %(message)s')
    
    success = test_improved_news_scraper()
    print(f"✅ News scraper test: {'PASSED' if success else 'FAILED'}")
