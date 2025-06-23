#!/usr/bin/env python3
"""
Test visual analyst after Ollama restart
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_visual_analyst():
    """Test the visual analyst"""
    try:
        from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent
        from collections import deque
        import logging
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        
        print("🧪 Testing Visual Analyst...")
        
        # Create test data
        close_data = [140.0 + i * 0.1 for i in range(50)]
        high_data = [c + 0.5 for c in close_data]
        low_data = [c - 0.3 for c in close_data]
        vol_data = [1000.0 + i * 10.0 for i in range(50)]
        
        close_deque = deque(close_data, maxlen=100)
        high_deque = deque(high_data, maxlen=100)
        low_deque = deque(low_data, maxlen=100)
        vol_deque = deque(vol_data, maxlen=100)
        
        # Create visual analyst (let it auto-configure)
        visual_agent = EnhancedVisualAnalystAgent()
        
        # Run analysis
        print("🔄 Running visual analysis...")
        result = visual_agent.run(
            symbol="SOLUSDT",
            timeframe_str="1m",
            close_buf_deque=close_deque,
            high_buf_deque=high_deque,
            low_buf_deque=low_deque,
            vol_buf_deque=vol_deque
        )
        
        print("✅ Visual Analysis Complete:")
        print(f"   Assessment: {result.overall_visual_assessment}")
        print(f"   Clarity Score: {result.pattern_clarity_score}")
        print(f"   Timeframe: {result.chart_timeframe_analyzed}")
        print(f"   Reasoning: {result.reasoning[:100]}...")
        
        # Check if it's working properly
        if result.pattern_clarity_score is not None and result.pattern_clarity_score > 0.0:
            print("🎉 Visual analyst is working correctly!")
            return True
        elif "Technical error" not in result.reasoning and "LLM query failed" not in result.reasoning:
            print("✅ Visual analyst generated proper output (even if clarity is 0.0)")
            return True
        else:
            print("⚠️  Visual analyst still returning error defaults")
            return False
            
    except Exception as e:
        print(f"❌ Error testing visual analyst: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_visual_analyst()
