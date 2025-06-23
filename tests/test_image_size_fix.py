#!/usr/bin/env python3
"""
Test script to validate that chart images are now generated with reasonable sizes
from the start without needing extreme resizing.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.chart_generator import EnhancedChartGenerator, FALLBACK_IMAGE_B64
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import base64
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(num_points=50):
    """Generate realistic OHLCV test data"""
    np.random.seed(42)
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=num_points), 
                         end=datetime.now(), freq='1h')[:num_points]
    
    # Generate realistic price data
    base_price = 100.0
    prices = []
    current_price = base_price
    
    for i in range(num_points):
        # Random walk with some trend
        change = np.random.normal(0, 0.02) * current_price
        current_price += change
        
        # Create OHLC for this period
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, 0.01)) * open_price
        low_price = open_price - abs(np.random.normal(0, 0.01)) * open_price
        close_price = low_price + (high_price - low_price) * np.random.random()
        
        prices.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': np.random.randint(1000, 10000)
        })
        
        current_price = close_price
    
    df = pd.DataFrame(prices, index=dates)
    return df

def test_image_size_improvements():
    """Test that images are now generated with reasonable sizes"""
    print("Testing image size improvements...")
    
    # Initialize chart generator
    chart_gen = EnhancedChartGenerator(save_charts_to_disk=True)
    
    # Generate test data with different sizes
    test_cases = [
        ("Small dataset", 20),
        ("Medium dataset", 40),
        ("Large dataset", 50)  # Limit to 50 as per our fix
    ]
    
    for case_name, num_points in test_cases:
        print(f"\n--- {case_name} ({num_points} points) ---")
        
        # Generate test data
        df = generate_test_data(num_points)
        
        # Generate chart
        try:
            chart_b64, saved_path = chart_gen.generate_chart_image(
                symbol="TEST/USDT",
                timeframe="1h",
                close_prices=df['Close'].tolist(),
                high_prices=df['High'].tolist(),
                low_prices=df['Low'].tolist(),
                volumes=df['Volume'].tolist(),
                lookback_periods=num_points
            )
            
            if chart_b64 and chart_b64 != FALLBACK_IMAGE_B64:
                # Decode and check image dimensions
                try:
                    image_data = base64.b64decode(chart_b64)
                    with Image.open(BytesIO(image_data)) as img:
                        width, height = img.size
                        file_size = len(image_data)
                        
                        print("✅ Chart generated successfully")
                        print(f"   Dimensions: {width}x{height} pixels")
                        print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                        
                        # Check if dimensions are reasonable
                        if width > 3000 or height > 2000:
                            print(f"⚠️  Image still quite large: {width}x{height}")
                        elif width < 200 or height < 100:
                            print(f"⚠️  Image too small: {width}x{height}")
                        else:
                            print("✅ Image dimensions are reasonable")
                        
                        # Save test image
                        test_file = f"test_chart_{case_name.lower().replace(' ', '_')}.png"
                        with open(test_file, 'wb') as f:
                            f.write(image_data)
                        print(f"   Saved as: {test_file}")
                        
                except Exception as e:
                    print(f"❌ Error analyzing image: {e}")
            else:
                print("❌ Chart generation failed or returned fallback")
                
        except Exception as e:
            print(f"❌ Error generating chart: {e}")
    
    print("\n=== Summary ===")
    print("Check the generated test images to verify visual quality.")
    print("Images should be:")
    print("- Between 200-2000 pixels wide")
    print("- Between 100-1500 pixels high") 
    print("- Under 500KB in size")
    print("- Visually clear and readable")

if __name__ == "__main__":
    test_image_size_improvements()
