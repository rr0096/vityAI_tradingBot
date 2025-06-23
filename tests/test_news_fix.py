#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 Quick Test: Simplified News System")
print("=" * 50)

try:
    # Test alternative news directly
    from tools.alternative_news import get_news_alternative
    print("✅ Testing alternative news...")
    news = get_news_alternative(limit=3)
    print(f"✅ Got {len(news)} news items from alternative sources")
    
    # Test new simplified news scraper  
    from tools.news_scraper import fetch_news
    print("✅ Testing simplified news scraper...")
    news2 = fetch_news(limit=3)
    print(f"✅ Got {len(news2)} news items from simplified scraper")
    
    print("🎉 All systems working! CryptoPanic eliminated successfully.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
