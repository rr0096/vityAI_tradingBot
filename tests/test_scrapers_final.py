#!/usr/bin/env python3
"""
Test final para verificar que Twitter y News scrapers funcionan correctamente
"""

import sys
import logging
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_twitter_scraper():
    """Test Twitter scraper functionality"""
    logger = logging.getLogger(__name__)
    logger.info("🐦 Testing Twitter scraper...")
    
    try:
        from tools.twitter_scraper import TwitterScraper
        scraper = TwitterScraper()
        
        # Test scraping
        results = scraper._run(usernames=["VitalikButerin"], limit_per_user=2, use_cache=False)
        
        if results and len(results) > 0:
            total_posts = sum(len(posts) for posts in results.values())
            logger.info(f"✅ Twitter scraper: {total_posts} posts")
            return True
        else:
            logger.warning("⚠️ Twitter scraper returned no posts")
            return False
    except Exception as e:
        logger.error(f"❌ Twitter scraper error: {e}")
        return False

def test_news_scraper():
    """Test improved news scraper functionality"""
    logger = logging.getLogger(__name__)
    logger.info("📰 Testing News scraper...")
    
    try:
        from tools.improved_news_scraper import ImprovedNewsScraper
        scraper = ImprovedNewsScraper()
        
        # Test scraping
        news = scraper.fetch_crypto_news(limit=5)  # Use correct method
        
        if news and len(news) > 0:
            logger.info(f"✅ News scraper: {len(news)} articles")
            return True
        else:
            logger.warning("⚠️ News scraper returned no articles")
            return False
    except Exception as e:
        logger.error(f"❌ News scraper error: {e}")
        return False

def test_sentiment_agent():
    """Test sentiment agent integration"""
    logger = logging.getLogger(__name__)
    logger.info("🎯 Testing sentiment agent integration...")
    
    try:
        from agents.sentiment_enhanced import EnhancedSentimentAnalyst
        from config.modern_models import ModelManager
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Create sentiment agent
        EnhancedSentimentAnalyst(
            model_manager=model_manager,
            symbol="BTC"
        )
        
        logger.info("✅ Sentiment agent created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Sentiment integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Final Integration Tests...")
    
    tests = [
        ("Twitter Scraper", test_twitter_scraper),
        ("News Scraper", test_news_scraper),
        ("Sentiment Agent", test_sentiment_agent)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                logger.warning(f"⚠️ {test_name} test had issues")
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
    
    logger.info("\n" + "="*50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
        logger.info("🔧 Some components may need additional configuration")

if __name__ == "__main__":
    main()
