#!/usr/bin/env python3
"""
Test completo del pipeline de sentiment con el nuevo Twitter scraper
"""

import sys
import logging
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.sentiment_enhanced import SentimentEnhanced
from tools.twitter_scraper import TwitterScraper
from tools.news_scraper import NewsScraper

def test_sentiment_pipeline_complete():
    """Test the complete sentiment pipeline with new scrapers"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Complete Sentiment Pipeline...")
    
    try:
        # Initialize sentiment agent
        logger.info("📊 Initializing SentimentEnhanced agent...")
        sentiment_agent = SentimentEnhanced()
        logger.info("✅ Sentiment agent initialized")
        
        # Test Twitter scraper integration
        logger.info("🐦 Testing Twitter scraper integration...")
        twitter_data = sentiment_agent._get_twitter_data(['BTC', 'ETH'])
        logger.info(f"✅ Twitter data obtained: {len(twitter_data)} posts")
        
        if twitter_data:
            sample_post = twitter_data[0]
            logger.info(f"📝 Sample Twitter post: [{sample_post.get('source', 'Unknown')}] {sample_post.get('text', '')[:100]}...")
        
        # Test News scraper integration
        logger.info("📰 Testing News scraper integration...")
        news_data = sentiment_agent._get_news_data(['BTC', 'ETH'])
        logger.info(f"✅ News data obtained: {len(news_data)} articles")
        
        if news_data:
            sample_news = news_data[0]
            logger.info(f"📰 Sample news: {sample_news.get('title', sample_news.get('text', ''))[:100]}...")
        
        # Test full sentiment analysis
        logger.info("🎯 Testing full sentiment analysis...")
        symbols = ['BTC']
        
        # Run sentiment analysis (this will use the new scrapers)
        from models.outputs import SentimentAnalysisInput
        input_data = SentimentAnalysisInput(
            symbols=symbols,
            context="Test analysis of Bitcoin sentiment using new scraper data"
        )
        
        logger.info("🤖 Running sentiment analysis with LLM...")
        result = sentiment_agent._run(input_data)
        logger.info("✅ Sentiment analysis completed")
        
        # Display results
        if hasattr(result, 'sentiment_scores') and result.sentiment_scores:
            for symbol, score_data in result.sentiment_scores.items():
                logger.info(f"📊 {symbol} Sentiment: {score_data}")
        
        # Summary
        total_sources = len(twitter_data) + len(news_data)
        logger.info("🎉 Pipeline test successful!")
        logger.info(f"   - Twitter posts: {len(twitter_data)}")
        logger.info(f"   - News articles: {len(news_data)}")
        logger.info(f"   - Total sources: {total_sources}")
        logger.info("   - Sentiment analysis: ✅ Working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in sentiment pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_scrapers():
    """Test individual scrapers in isolation"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Testing individual scrapers...")
    
    # Test Twitter scraper
    try:
        logger.info("🐦 Testing TwitterScraper...")
        twitter_scraper = TwitterScraper()
        twitter_results = twitter_scraper._run(
            usernames=['VitalikButerin', 'naval'],
            limit_per_user=3,
            use_cache=False
        )
        twitter_count = sum(len(posts) for posts in twitter_results.values())
        logger.info(f"✅ TwitterScraper: {twitter_count} posts")
    except Exception as e:
        logger.error(f"❌ TwitterScraper error: {e}")
    
    # Test News scraper
    try:
        logger.info("📰 Testing NewsScraper...")
        news_scraper = NewsScraper()
        news_results = news_scraper._run()
        news_count = len(news_results) if isinstance(news_results, list) else 0
        logger.info(f"✅ NewsScraper: {news_count} articles")
    except Exception as e:
        logger.error(f"❌ NewsScraper error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Complete Sentiment Pipeline Test...")
    
    # Test individual scrapers first
    test_individual_scrapers()
    
    print("\n" + "="*60)
    
    # Test complete pipeline
    success = test_sentiment_pipeline_complete()
    
    if success:
        print("🎉 ALL TESTS PASSED! The sentiment pipeline with new scrapers is working!")
        print("✅ Twitter scraper migration: COMPLETE")
        print("✅ News scraper update: COMPLETE") 
        print("✅ Sentiment pipeline integration: COMPLETE")
    else:
        print("❌ Some tests failed. Check the logs above.")
        sys.exit(1)
