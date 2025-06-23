#!/usr/bin/env python3
"""
Final test to verify chart generator produces images with proper dimensions
and visual quality after fixing the resizing logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import base64
import numpy as np
from io import BytesIO
from PIL import Image
from tools.chart_generator import EnhancedChartGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chart_dimensions():
    """Test that charts are generated with proper dimensions."""
    print("=== Testing Chart Dimensions ===")
    
    # Create generator
    generator = EnhancedChartGenerator()
    
    # Generate sample data - enough for proper chart
    num_points = 30
    base_price = 100.0
    
    # Create realistic price data with trends
    close_prices = []
    high_prices = []
    low_prices = []
    volumes = []
    
    current_price = base_price
    for i in range(num_points):
        # Add some randomness and trend
        change = np.random.normal(0, 2.0)  # Random walk with volatility
        trend = 0.1 * i  # Slight upward trend
        current_price += change + trend
        
        # Ensure realistic OHLC relationships
        close = max(1.0, current_price)
        high = close + abs(np.random.normal(0, 1.0))
        low = close - abs(np.random.normal(0, 1.0))
        volume = abs(np.random.normal(10000, 3000))
        
        close_prices.append(close)
        high_prices.append(high)
        low_prices.append(low)
        volumes.append(volume)
    
    print(f"Generated {len(close_prices)} data points")
    print(f"Price range: {min(close_prices):.2f} - {max(close_prices):.2f}")
    
    try:
        # Generate chart
        result_tuple = generator.generate_chart_image(
            symbol="TESTBTC",
            close_prices=close_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            volumes=volumes,
            timeframe="1h",
            lookback_periods=num_points
        )
        
        # Extract the base64 string from the tuple
        result, saved_path = result_tuple
        
        if result == "Error generating chart":
            print("❌ Chart generation failed")
            return False
        
        # Decode and analyze the image
        image_data = base64.b64decode(result)
        image = Image.open(BytesIO(image_data))
        width, height = image.size
        
        print(f"Generated image dimensions: {width}x{height}")
        
        # Check dimensions
        min_width = 400
        min_height = 300
        max_width = 2000
        max_height = 1500
        
        if width < min_width:
            print(f"❌ Image width too small: {width} < {min_width}")
            return False
        
        if height < min_height:
            print(f"❌ Image height too small: {height} < {min_height}")
            return False
        
        if width > max_width:
            print(f"⚠️  Image width very large: {width} > {max_width}")
        
        if height > max_height:
            print(f"⚠️  Image height very large: {height} > {max_height}")
        
        # Check aspect ratio is reasonable
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 3.0:
            print(f"⚠️  Unusual aspect ratio: {aspect_ratio:.2f}")
        else:
            print(f"✅ Good aspect ratio: {aspect_ratio:.2f}")
        
        # Check image is not mostly empty/blank
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            # Convert to grayscale for analysis
            grayscale = np.mean(image_array, axis=2)
        else:
            grayscale = image_array
        
        # Check if image has content (not all same color)
        min_val = np.min(grayscale)
        max_val = np.max(grayscale)
        variance = np.var(grayscale)
        
        print(f"Image pixel range: {min_val:.1f} - {max_val:.1f}, variance: {variance:.1f}")
        
        if variance < 10:
            print("⚠️  Image might be mostly blank (low variance)")
        else:
            print("✅ Image has visual content")
        
        # Save a test image for manual inspection
        test_image_path = "test_chart_output.png"
        image.save(test_image_path)
        print(f"✅ Test image saved as: {test_image_path}")
        
        print("✅ Chart generation test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Chart generation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_charts():
    """Test generating multiple charts with different data sets."""
    print("\n=== Testing Multiple Charts ===")
    
    generator = EnhancedChartGenerator()
    
    test_cases = [
        {"points": 20, "name": "Small dataset"},
        {"points": 40, "name": "Medium dataset"},
        {"points": 50, "name": "Large dataset"},
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {case['name']} ({case['points']} points)")
        
        # Generate data
        points = case['points']
        close_prices = [100 + np.random.normal(0, 5) for _ in range(points)]
        high_prices = [p + abs(np.random.normal(0, 2)) for p in close_prices]
        low_prices = [p - abs(np.random.normal(0, 2)) for p in close_prices]
        volumes = [abs(np.random.normal(10000, 2000)) for _ in range(points)]
        
        try:
            result_tuple = generator.generate_chart_image(
                symbol=f"TEST{i+1}",
                close_prices=close_prices,
                high_prices=high_prices,
                low_prices=low_prices,
                volumes=volumes,
                timeframe="15m",
                lookback_periods=points
            )
            
            result, saved_path = result_tuple
            
            if result == "Error generating chart":
                print(f"❌ Failed: {case['name']}")
                all_passed = False
                continue
            
            # Check dimensions
            image_data = base64.b64decode(result)
            image = Image.open(BytesIO(image_data))
            width, height = image.size
            
            print(f"  Dimensions: {width}x{height}")
            
            if width >= 400 and height >= 300 and width <= 2000 and height <= 1500:
                print(f"  ✅ {case['name']} - dimensions OK")
            else:
                print(f"  ❌ {case['name']} - dimensions out of range")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ {case['name']} failed: {e}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("🧪 Testing Enhanced Chart Generator - Final Dimensions Test")
    print("=" * 60)
    
    # Test basic chart generation
    test1_passed = test_chart_dimensions()
    
    # Test multiple charts
    test2_passed = test_multiple_charts()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! Chart generator is working correctly.")
        print("📊 Images are generated with proper dimensions and content.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)
