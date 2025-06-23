# examples/enhanced_chart_usage.py
"""
Example demonstrating the enhanced chart generator capabilities.
This shows how to use the new pattern detection and analysis features.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

# Import enhanced chart functionality
try:
    from tools.chart_enhancements import (
        generate_enhanced_chart_for_visual_agent,
        analyze_chart_patterns,
        get_pattern_summary,
        EnhancedPatternDetector,
        ChartPattern,
        PatternType
    )
    print("✅ Enhanced chart generator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import enhanced chart generator: {e}")
    sys.exit(1)

def generate_sample_data(size: int = 100) -> tuple:
    """Generate realistic sample OHLCV data for testing"""
    print(f"📊 Generating {size} sample data points...")
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Generate price movement with trend and volatility
    base_price = 100.0
    price_changes = np.random.normal(0, 0.5, size)
    
    # Add some trend
    trend = np.linspace(0, 10, size)
    close_prices = base_price + np.cumsum(price_changes) + trend
    
    # Generate OHLC from closes
    open_prices = []
    high_prices = []
    low_prices = []
    volumes = []
    
    for i, close in enumerate(close_prices):
        # Open is previous close + small gap
        if i == 0:
            open_price = close + np.random.normal(0, 0.1)
        else:
            open_price = close_prices[i-1] + np.random.normal(0, 0.2)
        
        # High and low around open/close
        high = max(open_price, close) + abs(np.random.normal(0, 0.3))
        low = min(open_price, close) - abs(np.random.normal(0, 0.3))
        
        # Volume with some correlation to price movement
        volume_base = 5000
        volatility_factor = abs(high - low) / close * 1000
        volume = int(volume_base + volatility_factor + np.random.normal(0, 1000))
        volume = max(1000, volume)  # Minimum volume
        
        open_prices.append(open_price)
        high_prices.append(high)
        low_prices.append(low)
        volumes.append(volume)
    
    return close_prices, high_prices, low_prices, volumes

def demonstrate_basic_usage():
    """Demonstrate basic enhanced chart generation"""
    print("\n🎯 === BASIC ENHANCED CHART GENERATION ===")
    
    # Generate sample data
    close_prices, high_prices, low_prices, volumes = generate_sample_data(100)
    
    # Generate enhanced chart
    print("🔄 Generating enhanced chart...")
    chart_b64, filepath, patterns = generate_enhanced_chart_for_visual_agent(
        symbol="BTCUSDT",
        timeframe="1h",
        close_buf=close_prices,
        high_buf=high_prices,
        low_buf=low_prices,
        vol_buf=volumes,
        tech_metrics={'last_price': close_prices[-1], 'volume': volumes[-1]},
        lookback_periods=100,
        save_chart=True,
        detect_patterns=True
    )
    
    # Display results
    print(f"📈 Chart generated: {len(chart_b64)} bytes")
    print(f"💾 Saved to: {filepath if filepath else 'Not saved'}")
    print(f"🔍 Patterns detected: {len(patterns)}")
    
    if patterns:
        print("\n📋 Detected Patterns:")
        for i, pattern in enumerate(patterns, 1):
            signal = "🟢" if pattern.bullish_signal else "🔴" if pattern.bullish_signal is False else "⚪"
            print(f"  {i}. {signal} {pattern.name}")
            print(f"     Confidence: {pattern.confidence:.1%}")
            print(f"     Description: {pattern.description}")
            if pattern.price_level:
                print(f"     Price Level: ${pattern.price_level:.2f}")
            print()
    else:
        print("  No significant patterns detected")
    
    return patterns

def demonstrate_pattern_analysis():
    """Demonstrate standalone pattern analysis"""
    print("\n🔍 === STANDALONE PATTERN ANALYSIS ===")
    
    # Create DataFrame for analysis
    close_prices, high_prices, low_prices, volumes = generate_sample_data(50)
    
    df = pd.DataFrame({
        'Open': [close_prices[i-1] if i > 0 else close_prices[0] for i in range(len(close_prices))],
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
    
    print("📊 Analyzing chart patterns...")
    analysis = analyze_chart_patterns(df)
    
    print(f"📈 Analysis Results:")
    print(f"  Total patterns: {analysis['patterns_detected']}")
    print(f"  Bullish signals: {analysis['bullish_patterns']}")
    print(f"  Bearish signals: {analysis['bearish_patterns']}")
    print(f"  Neutral patterns: {analysis['neutral_patterns']}")
    print(f"  Support levels: {analysis['support_levels']}")
    print(f"  Resistance levels: {analysis['resistance_levels']}")
    
    # Generate pattern summary
    summary = get_pattern_summary(analysis['patterns'])
    print(f"\n📝 Pattern Summary: {summary}")
    
    return analysis

def demonstrate_custom_pattern_detection():
    """Demonstrate custom pattern detection"""
    print("\n🎯 === CUSTOM PATTERN DETECTION ===")
    
    # Create pattern detector
    detector = EnhancedPatternDetector(min_confidence=0.7)
    
    # Create DataFrame with specific patterns
    print("📊 Creating data with specific patterns...")
    
    # Create a Doji pattern (open ≈ close, long wicks)
    doji_data = {
        'Open': [100.0, 101.0, 102.01],  # Last candle: small body
        'High': [100.5, 101.5, 104.0],  # Last candle: long upper wick
        'Low': [99.5, 100.5, 100.0],    # Last candle: long lower wick  
        'Close': [101.0, 102.0, 102.0], # Last candle: close ≈ open
        'Volume': [5000, 5000, 5000]
    }
    
    doji_df = pd.DataFrame(doji_data)
    patterns = detector.detect_candlestick_patterns(doji_df)
    
    print(f"🕯️ Patterns in Doji test data: {len(patterns)}")
    for pattern in patterns:
        print(f"  - {pattern.name}: {pattern.confidence:.1%}")
    
    # Demonstrate support/resistance detection
    print("\n🏗️ Testing Support/Resistance detection...")
    
    # Create data with clear support level
    sr_closes = [100, 99, 100.1, 98.9, 101, 99.8, 100.2, 102, 99.9, 100.1]
    sr_data = {
        'High': [c + 1 for c in sr_closes],
        'Low': [c - 1 for c in sr_closes], 
        'Close': sr_closes,
        'Volume': [5000] * len(sr_closes)
    }
    
    sr_df = pd.DataFrame(sr_data)
    sr_levels = detector.detect_support_resistance_levels(sr_df, lookback=3, min_touches=2)
    
    print(f"📊 Support/Resistance levels found: {len(sr_levels)}")
    for level in sr_levels:
        level_type = "Support" if level.is_support else "Resistance"
        print(f"  - {level_type}: ${level.price:.2f} (Strength: {level.strength:.1%}, Touches: {level.touches})")

def demonstrate_integration_with_existing_code():
    """Show how enhanced charts integrate with existing code"""
    print("\n🔗 === INTEGRATION WITH EXISTING CODE ===")
    
    # This shows backward compatibility
    print("✅ Backward compatibility test...")
    
    # Original function call (still works)
    try:
        from tools.chart_generator import generate_chart_for_visual_agent
        
        close_prices, high_prices, low_prices, volumes = generate_sample_data(50)
        
        # Original call
        original_result = generate_chart_for_visual_agent(
            symbol="BTCUSDT",
            timeframe="1h", 
            close_buf=close_prices,
            high_buf=high_prices,
            low_buf=low_prices,
            vol_buf=volumes,
            save_chart=False
        )
        
        print(f"  Original function: {type(original_result)} with {len(original_result)} elements")
        
        # Enhanced call
        enhanced_result = generate_enhanced_chart_for_visual_agent(
            symbol="BTCUSDT",
            timeframe="1h",
            close_buf=close_prices, 
            high_buf=high_prices,
            low_buf=low_prices,
            vol_buf=volumes,
            save_chart=False,
            detect_patterns=True
        )
        
        print(f"  Enhanced function: {type(enhanced_result)} with {len(enhanced_result)} elements")
        print(f"  Enhanced adds: pattern detection ({len(enhanced_result[2])} patterns)")
        
    except ImportError as e:
        print(f"  ⚠️ Original chart generator not available: {e}")

def main():
    """Main demonstration function"""
    print("🎨 Enhanced Chart Generator Demonstration")
    print("=" * 50)
    
    try:
        # Run demonstrations
        patterns = demonstrate_basic_usage()
        analysis = demonstrate_pattern_analysis() 
        demonstrate_custom_pattern_detection()
        demonstrate_integration_with_existing_code()
        
        print("\n✨ === SUMMARY ===")
        print(f"✅ Enhanced chart generation completed successfully")
        print(f"🔍 Pattern detection working: {len(patterns) if patterns else 0} patterns found")
        print(f"📊 Analysis functionality: {analysis['patterns_detected']} total patterns")
        print(f"🔗 Integration: Backward compatible with existing code")
        print(f"\n📚 For more information, see: docs/CHART_GENERATOR_IMPROVEMENTS.md")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
