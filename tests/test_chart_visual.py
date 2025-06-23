#!/usr/bin/env python3
"""
Test script to validate chart generation with proper visual dimensions.
This test will generate a chart and save it to disk for visual inspection.
"""

import sys
import os
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
from collections import deque
import numpy as np

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.chart_generator import generate_chart_for_visual_agent

def create_sample_price_data(periods=50):
    """Create realistic sample price data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 50000.0
    prices = [base_price]
    
    # Generate price movement with some trends
    for i in range(periods - 1):
        # Random walk with slight upward bias
        change_percent = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
        new_price = prices[-1] * (1 + change_percent)
        prices.append(max(new_price, 1.0))  # Ensure price stays positive
    
    # Create OHLC data
    close_prices = prices
    high_prices = [price * (1 + abs(np.random.normal(0, 0.01))) for price in prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.01))) for price in prices]
    volumes = [np.random.uniform(1000, 10000) for _ in range(periods)]
    
    # Ensure OHLC rules
    for i in range(len(prices)):
        high_prices[i] = max(high_prices[i], close_prices[i], low_prices[i])
        low_prices[i] = min(low_prices[i], close_prices[i], high_prices[i])
    
    return close_prices, high_prices, low_prices, volumes

def test_chart_generation():
    """Test chart generation and validate output."""
    print("🔄 Generating sample price data...")
    
    # Create sample data
    close_prices, high_prices, low_prices, volumes = create_sample_price_data(50)
    
    print(f"📊 Sample data created: {len(close_prices)} price points")
    print(f"   Price range: ${min(close_prices):.2f} - ${max(close_prices):.2f}")
    
    # Generate chart
    print("🎨 Generating chart...")
    try:
        base64_image, saved_filepath = generate_chart_for_visual_agent(
            symbol="BTC/USD",
            timeframe="1h",
            close_buf=close_prices,
            high_buf=high_prices,
            low_buf=low_prices,
            vol_buf=volumes,
            tech_metrics={
                'rsi': 65.5,
                'bb_upper': max(close_prices) * 1.02,
                'bb_lower': min(close_prices) * 0.98,
                'support': min(close_prices) * 1.01,
                'resistance': max(close_prices) * 0.99
            },
            lookback_periods=50,
            save_chart=True
        )
        
        if not base64_image:
            print("❌ Chart generation failed - no image returned")
            return False
            
        print("✅ Chart generated successfully!")
        
        # Decode and analyze the image
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            width, height = image.size
            
            print(f"📏 Image dimensions: {width}x{height} pixels")
            print(f"📁 Image size: {len(image_data):,} bytes ({len(image_data)/1024:.1f} KB)")
            print(f"🎨 Image mode: {image.mode}")
            
            # Validate dimensions
            if width < 100 or height < 100:
                print("❌ Image is too small to be useful")
                return False
            elif width > 2000 or height > 2000:
                print("⚠️  Image is very large, might cause memory issues")
            else:
                print("✅ Image dimensions are reasonable")
            
            # Check aspect ratio
            aspect_ratio = width / height
            print(f"📐 Aspect ratio: {aspect_ratio:.2f}")
            
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                print("⚠️  Unusual aspect ratio detected")
            else:
                print("✅ Aspect ratio looks good")
            
            # Save for manual inspection
            output_path = project_root / "test_chart_output.png"
            with open(output_path, 'wb') as f:
                f.write(image_data)
            print(f"💾 Chart saved to: {output_path}")
            
            if saved_filepath:
                print(f"💾 Also saved to: {saved_filepath}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing image: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Chart generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Chart Visual Test")
    print("=" * 50)
    
    success = test_chart_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Test completed successfully!")
        print("📖 Please check the generated image file to verify visual quality.")
    else:
        print("❌ Test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
