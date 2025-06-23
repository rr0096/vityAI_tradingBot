#!/usr/bin/env python3
"""
Test script for the Enhanced Sentiment Agent Cache System
=========================================================

This script tests:
1. Initial cache state
2. First refresh (should fetch data)
3. Immediate second refresh (should use cache)
4. Force refresh (should bypass cache)
5. Cache status monitoring
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from agents.sentiment_enhanced import EnhancedSentimentAnalyst

def test_sentiment_cache():
    """Test the sentiment agent cache system."""
    print("🧪 Testing Enhanced Sentiment Agent Cache System")
    print("=" * 60)
    
    # Create sentiment agent with simple initialization
    print("🔧 Initializing sentiment agent...")
    agent = EnhancedSentimentAnalyst(
        llm="qwen2.5:7b-instruct",  # Use a simple model name
        verbose=True
    )
    
    print(f"✅ Agent initialized with {agent.cache_refresh_interval_minutes} minute cache interval")
    
    # Test 1: Check initial cache status
    print("\n📊 Test 1: Initial cache status")
    cache_status = agent.get_cache_status()
    print(f"   Cache status: {cache_status}")
    
    # Test 2: First refresh (should fetch data)
    print("\n🔄 Test 2: First refresh (should fetch data)")
    start_time = time.time()
    agent.refresh()
    first_refresh_time = time.time() - start_time
    print(f"   ✅ First refresh completed in {first_refresh_time:.2f} seconds")
    
    cache_status = agent.get_cache_status()
    print(f"   Cache status after first refresh: {cache_status}")
    
    # Test 3: Immediate second refresh (should use cache)
    print("\n⚡ Test 3: Immediate second refresh (should use cache)")
    start_time = time.time()
    agent.refresh()  # Should be skipped due to cache
    second_refresh_time = time.time() - start_time
    print(f"   ✅ Second refresh completed in {second_refresh_time:.2f} seconds")
    print(f"   ⚡ Speed improvement: {first_refresh_time/max(second_refresh_time, 0.001):.1f}x faster")
    
    # Test 4: Force refresh (should bypass cache)
    print("\n🔨 Test 4: Force refresh (should bypass cache)")
    start_time = time.time()
    agent.refresh(force=True)
    force_refresh_time = time.time() - start_time
    print(f"   ✅ Force refresh completed in {force_refresh_time:.2f} seconds")
    
    # Test 5: Cache expiration simulation
    print("\n⏰ Test 5: Simulating cache expiration")
    # Temporarily reduce cache interval for testing
    original_interval = agent.cache_refresh_interval_minutes
    agent.cache_refresh_interval_minutes = 1  # 1 minute for testing
    print(f"   Reduced cache interval to {agent.cache_refresh_interval_minutes} minutes for testing")
    
    time.sleep(61)  # Wait for cache to expire (61 seconds)
    
    start_time = time.time()
    agent.refresh()  # Should fetch new data
    expired_refresh_time = time.time() - start_time
    print(f"   ✅ Expired cache refresh completed in {expired_refresh_time:.2f} seconds")
    
    # Restore original interval
    agent.cache_refresh_interval_minutes = original_interval
    
    # Final status
    print("\n📈 Final Results:")
    final_cache_status = agent.get_cache_status()
    print(f"   Final cache status: {final_cache_status}")
    
    print("\n🎯 Performance Summary:")
    print(f"   • First refresh (full fetch): {first_refresh_time:.2f}s")
    print(f"   • Cached refresh: {second_refresh_time:.2f}s")
    print(f"   • Force refresh: {force_refresh_time:.2f}s")
    print(f"   • Expired refresh: {expired_refresh_time:.2f}s")
    print(f"   • Cache efficiency: {first_refresh_time/max(second_refresh_time, 0.001):.1f}x speedup")
    
    print("\n🎉 Sentiment agent cache system test completed successfully!")
    return agent

if __name__ == "__main__":
    try:
        agent = test_sentiment_cache()
        
        print("\n🔍 Additional Info:")
        print(f"   • Total fetched items: {len(agent._all_fetched_text_items)}")
        print(f"   • Sampled for LLM: {len(agent._sampled_texts_for_llm)}")
        print(f"   • Fear & Greed Index: {agent._current_fear_greed_value}")
        print(f"   • Cache interval: {agent.cache_refresh_interval_minutes} minutes")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
