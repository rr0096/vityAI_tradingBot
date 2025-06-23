#!/usr/bin/env python3
"""
Test script para verificar que el Twitter scraper funciona correctamente
"""

import sys
import logging
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.twitter_scraper import TwitterScraper

def test_twitter_scraper():
    """Test basic functionality of the Twitter scraper"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Twitter Scraper Alternative...")
    
    try:
        # Initialize scraper
        scraper = TwitterScraper()
        logger.info("✅ Scraper initialized successfully")
        
        # Test with a few crypto influencers
        test_users = ["VitalikButerin", "naval", "balajis"]
        logger.info(f"📡 Testing scraping for users: {test_users}")
        
        # Run scraper
        results = scraper._run(
            usernames=test_users,
            limit_per_user=5,
            use_cache=False,  # Force fresh data
            include_alternatives=True
        )
        
        # Validate results
        total_posts = 0
        for username, posts in results.items():
            post_count = len(posts)
            total_posts += post_count
            logger.info(f"📊 {username}: {post_count} posts")
            
            # Show sample post
            if posts:
                sample_post = posts[0]
                logger.info(f"  Sample: [{sample_post.get('source', 'Unknown')}] {sample_post.get('text', '')[:100]}...")
                logger.info(f"  URL: {sample_post.get('url', 'N/A')}")
                logger.info(f"  Timestamp: {sample_post.get('timestamp_utc', 'N/A')}")
        
        logger.info(f"🎯 Total posts scraped: {total_posts}")
        
        if total_posts > 0:
            logger.info("✅ Twitter scraper is working correctly!")
            return True
        else:
            logger.warning("⚠️ No posts were scraped - check feed URLs")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing Twitter scraper: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rss_feeds():
    """Test specific RSS feeds"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing individual RSS feeds...")
    
    scraper = TwitterScraper()
    
    # Test specific RSS feeds
    rss_tests = [
        ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("cointelegraph", "https://cointelegraph.com/rss"),
        ("decrypt", "https://decrypt.co/feed")
    ]
    
    for name, url in rss_tests:
        logger.info(f"📡 Testing {name} RSS feed...")
        try:
            posts = scraper._fetch_rss_content(name, url)
            logger.info(f"  ✅ {name}: {len(posts)} posts fetched")
            if posts:
                logger.info(f"  Sample: {posts[0].get('text', '')[:80]}...")
        except Exception as e:
            logger.warning(f"  ⚠️ {name}: Error - {e}")

if __name__ == "__main__":
    print("🚀 Starting Twitter Scraper Tests...")
    
    # Test RSS feeds first
    test_rss_feeds()
    
    print("\n" + "="*60)
    
    # Test main functionality
    success = test_twitter_scraper()
    
    if success:
        print("🎉 All tests passed! Twitter scraper is working.")
    else:
        print("❌ Some tests failed. Check the logs above.")
        sys.exit(1)
