#!/usr/bin/env python3

"""
Test script for Bitget chart capture functionality
"""

import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.chart_generator_real import generate_chart_for_visual_agent_real

def test_bitget_capture():
    """Test the Bitget chart capture."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 TESTING BITGET CHART CAPTURE")
    print("=" * 50)
    
    # Test with SOL/USDT
    symbol = "SOLUSDT"
    timeframe = "1"
    
    print(f"📊 Capturing chart for {symbol} ({timeframe}m)")
    print("⏳ This may take 10-15 seconds...")
    
    try:
        chart_b64, filepath = generate_chart_for_visual_agent_real(
            symbol=symbol,
            timeframe=timeframe,
            close_buf=None,
            high_buf=None,
            low_buf=None,
            vol_buf=None,
            tech_metrics={},
            save_chart=True
        )
        
        if chart_b64:
            print(f"✅ SUCCESS! Chart captured")
            print(f"📏 Base64 length: {len(chart_b64)} characters")
            if filepath:
                print(f"💾 Saved to: {filepath}")
            
            # Test if it's actually image data
            try:
                import base64
                decoded = base64.b64decode(chart_b64)
                if len(decoded) > 1000 and decoded.startswith(b'\x89PNG'):
                    print("✅ Valid PNG image detected")
                else:
                    print("⚠️ Image data seems invalid")
            except Exception as e:
                print(f"⚠️ Could not validate image: {e}")
                
        else:
            print("❌ FAILED! No chart data captured")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🔧 Test completed")

if __name__ == "__main__":
    test_bitget_capture()
