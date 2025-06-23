#!/usr/bin/env python3
"""
Simplified News Scraper
=======================

Este archivo reemplaza el CryptoPanic scraper con fuentes alternativas
que están funcionando correctamente.
"""

import logging
from typing import List, Dict, Any
from tools.alternative_news import AlternativeNewsProvider

logger = logging.getLogger(__name__)

# Initialize the alternative news provider
_news_provider = AlternativeNewsProvider()

def fetch_news(limit: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch news using alternative sources only.
    
    Args:
        limit: Number of news items to fetch
        
    Returns:
        List of news items
    """
    logger.info(f"Fetching {limit} news items from alternative sources...")
    try:
        news_items = _news_provider.get_alternative_news(limit=limit)
        logger.info(f"✅ Successfully fetched {len(news_items)} news items")
        return news_items
    except Exception as e:
        logger.error(f"❌ Failed to fetch news from alternative sources: {e}")
        return []

if __name__ == "__main__":
    # Test the simplified news scraper
    print("🧪 Testing Simplified News Scraper")
    print("=" * 50)
    
    news = fetch_news(limit=5)
    
    if news:
        print(f"✅ Fetched {len(news)} news items:")
        for i, item in enumerate(news, 1):
            print(f"{i}. {item.get('title', 'No Title')[:60]}...")
            print(f"   Source: {item.get('source', 'Unknown')} | {item.get('published_at', 'No date')[:10]}")
    else:
        print("❌ No news items fetched")
    
    print("\n🎉 Simplified news scraper test completed!")
