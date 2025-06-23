#!/usr/bin/env python3
"""
Test simple para verificar que los scrapers funcionan después de la migración
"""

import sys
import logging
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_scrapers_working():
    """Test que los scrapers funcionan correctamente después de la migración"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Scrapers After Migration...")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Twitter scraper
    try:
        total_tests += 1
        logger.info("🐦 Testing Twitter scraper...")
        
        from tools.twitter_scraper import TwitterScraper
        scraper = TwitterScraper()
        
        results = scraper._run(
            usernames=['VitalikButerin', 'naval'], 
            limit_per_user=3,
            use_cache=False
        )
        
        post_count = sum(len(posts) for posts in results.values())
        logger.info(f"✅ Twitter scraper: {post_count} posts obtained")
        
        if post_count > 0:
            success_count += 1
            # Show sample
            for username, posts in results.items():
                if posts:
                    sample = posts[0]
                    logger.info(f"   Sample from {username}: [{sample.get('source')}] {sample.get('text', '')[:80]}...")
                    break
        else:
            logger.warning("⚠️ No posts obtained from Twitter scraper")
            
    except Exception as e:
        logger.error(f"❌ Twitter scraper test failed: {e}")
    
    # Test 2: News scraper  
    try:
        total_tests += 1
        logger.info("📰 Testing News scraper...")
        
        from tools.news_scraper import NewsScraper
        news_scraper = NewsScraper()
        
        news_results = news_scraper._run()
        news_count = len(news_results) if isinstance(news_results, list) else 0
        
        logger.info(f"✅ News scraper: {news_count} articles obtained")
        
        if news_count > 0:
            success_count += 1
            # Show sample
            if news_results and len(news_results) > 0:
                sample = news_results[0]
                title = sample.get('title', sample.get('text', 'No title'))
                logger.info(f"   Sample news: {title[:80]}...")
        else:
            logger.warning("⚠️ No news obtained from News scraper")
            
    except Exception as e:
        logger.error(f"❌ News scraper test failed: {e}")
    
    # Test 3: Integration test - sentiment agent with scrapers
    try:
        total_tests += 1
        logger.info("🎯 Testing sentiment agent integration...")
        
        # Try to import and create sentiment agent
        from agents.sentiment_enhanced import SentimentEnhanced
        sentiment_agent = SentimentEnhanced()
        
        # Test twitter data fetch
        twitter_data = sentiment_agent._get_twitter_data(['BTC'])
        twitter_data_count = len(twitter_data) if twitter_data else 0
        
        # Test news data fetch
        news_data = sentiment_agent._get_news_data(['BTC'])
        news_data_count = len(news_data) if news_data else 0
        
        total_data = twitter_data_count + news_data_count
        logger.info(f"✅ Sentiment integration: {total_data} data points")
        logger.info(f"   - Twitter: {twitter_data_count} posts")
        logger.info(f"   - News: {news_data_count} articles")
        
        if total_data > 0:
            success_count += 1
        else:
            logger.warning("⚠️ No data obtained through sentiment agent")
            
    except Exception as e:
        logger.error(f"❌ Sentiment integration test failed: {e}")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Twitter scraper migration: SUCCESS")
        logger.info("✅ News scraper update: SUCCESS")
        logger.info("✅ System integration: SUCCESS")
        logger.info("\n🚀 The system is ready for Week 3 of the roadmap!")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - success_count} tests failed")
        logger.info("🔧 Some components may need additional configuration")
        return False

if __name__ == "__main__":
    print("🧪 Testing FenixTradingBot Scrapers After Migration...")
    print("="*60)
    
    success = test_scrapers_working()
    
    if success:
        print("\n🎯 MIGRATION SUCCESSFUL!")
        print("📋 Next Steps:")
        print("   1. ✅ Twitter scraper working with real data")
        print("   2. ✅ News scraper working with RSS feeds") 
        print("   3. ✅ Sentiment pipeline integrated")
        print("   4. 🚀 Ready for Week 3: Multi-model consensus")
    else:
        print("\n⚠️ Some issues detected. Check logs above.")
        sys.exit(1)
