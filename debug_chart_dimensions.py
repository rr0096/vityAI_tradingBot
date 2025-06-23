#!/usr/bin/env python3
"""Debug script to test chart generation and examine dimensions."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
import base64

def create_test_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(30) * 2)
    high_prices = close_prices + np.random.uniform(1, 3, 30)
    low_prices = close_prices - np.random.uniform(1, 3, 30)
    open_prices = close_prices + np.random.uniform(-2, 2, 30)
    volumes = np.random.uniform(1000, 10000, 30)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)
    
    return df

def test_basic_mplfinance():
    """Test basic mplfinance plotting to see dimensions."""
    print("=== Testing Basic mplfinance ===")
    
    df = create_test_data()
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
    
    # Test 1: Default settings
    fig, axes = mpf.plot(df, type='candle', volume=True, returnfig=True)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    with Image.open(buf) as img:
        print(f"Test 1 - Default: {img.size[0]}x{img.size[1]} pixels")
    
    plt.close(fig)
    buf.close()
    
    # Test 2: With specific figsize
    fig, axes = mpf.plot(df, type='candle', volume=True, figsize=(12, 8), returnfig=True)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    
    with Image.open(buf) as img:
        print(f"Test 2 - figsize(12,8): {img.size[0]}x{img.size[1]} pixels")
    
    plt.close(fig)
    buf.close()
    
    # Test 3: With figscale
    fig, axes = mpf.plot(df, type='candle', volume=True, figsize=(10, 8), figscale=1.2, returnfig=True)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    
    with Image.open(buf) as img:
        print(f"Test 3 - figscale=1.2: {img.size[0]}x{img.size[1]} pixels")
    
    plt.close(fig)
    buf.close()

def test_with_custom_style():
    """Test with custom styling similar to the chart generator."""
    print("\n=== Testing with Custom Style ===")
    
    df = create_test_data()
    
    # Create custom style similar to chart generator
    up_color = '#26A69A'
    down_color = '#EF5350'
    bg_color = '#131722'
    axes_bg_color = '#1A1E29'
    text_color = '#E0E0E0'
    grid_color = '#404040'
    
    mpf_marketcolors = mpf.make_marketcolors(
        up=up_color, down=down_color, edge='inherit',
        wick={'up': up_color, 'down': down_color},
        volume={'up': up_color, 'down': down_color}, alpha=0.9
    )
    
    style_rc_params = {
        'font.size': 9,
        'axes.labelsize': 8, 
        'axes.titlesize': 11,
        'axes.grid': True, 
        'grid.alpha': 0.15, 
        'grid.color': grid_color,
        'figure.facecolor': bg_color, 
        'axes.facecolor': axes_bg_color,
        'xtick.color': text_color, 
        'ytick.color': text_color,
        'axes.labelcolor': text_color, 
        'axes.titlecolor': text_color,
        'text.color': text_color,
        'figure.dpi': 80,
        'savefig.dpi': 80,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    }
    
    try:
        custom_mpf_style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=mpf_marketcolors,
            rc=style_rc_params,
            y_on_right=False
        )
    except Exception as e:
        print(f"Error creating custom style: {e}")
        custom_mpf_style = 'nightclouds'
    
    # Test with the custom style
    fig, axes = mpf.plot(
        df, type='candle', style=custom_mpf_style,
        volume=True, figsize=(10, 8), returnfig=True,
        figscale=1.0, tight_layout=True
    )
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=80, 
                facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    
    with Image.open(buf) as img:
        print(f"Custom style test: {img.size[0]}x{img.size[1]} pixels")
        
        # Save for visual inspection
        img.save('/Users/giovanniarangio/carpeta sin título 4/fenixtradingbot/debug_chart.png')
        print("Saved debug chart as debug_chart.png")
    
    plt.close(fig)
    buf.close()

if __name__ == "__main__":
    test_basic_mplfinance()
    test_with_custom_style()
    print("\nDebug tests completed.")
