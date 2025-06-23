#!/usr/bin/env python3
"""
Quick Cache Test for Sentiment Agent
====================================
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from agents.sentiment_enhanced import EnhancedSentimentAnalyst

def quick_cache_test():
    """Quick test of cache functionality."""
    print("🚀 Quick Sentiment Agent Cache Test")
    print("=" * 40)
    
    # Create agent
    agent = EnhancedSentimentAnalyst(llm="qwen2.5:7b-instruct", verbose=True)
    print(f"✅ Agent created with {agent.cache_refresh_interval_minutes}min cache")
    
    # Check cache status
    status = agent.get_cache_status()
    print(f"📊 Initial cache status: {status['needs_refresh']}")
    
    # First refresh
    print("\n🔄 First refresh...")
    start = time.time()
    agent.refresh()
    first_time = time.time() - start
    print(f"⏱️  Time: {first_time:.2f}s")
    
    # Second refresh (should be cached)
    print("\n⚡ Second refresh (should use cache)...")
    start = time.time()
    agent.refresh()
    second_time = time.time() - start
    print(f"⏱️  Time: {second_time:.2f}s")
    
    # Check final status
    final_status = agent.get_cache_status()
    print(f"\n📈 Results:")
    print(f"   • Speedup: {first_time/max(second_time, 0.001):.1f}x")
    print(f"   • Cache valid: {not final_status['needs_refresh']}")
    print(f"   • Next refresh: {final_status['next_refresh_in']}")
    
    print("\n🎉 Cache test completed!")
    return agent

if __name__ == "__main__":
    try:
        agent = quick_cache_test()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
