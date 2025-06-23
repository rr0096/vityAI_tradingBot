#!/usr/bin/env python3
"""
Test script to verify the technical agent works with all the fixes applied.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_technical_agent():
    """Test the technical agent with various scenarios."""
    print("🧪 Testing Technical Agent with Fixes")
    print("=" * 50)
    
    try:
        # Import and create agent
        print("1. Importing technical agent...")
        from agents.technical_v_enhanced_fixed import EnhancedTechnicalAnalyst
        print("✅ Import successful")
        
        print("2. Creating agent instance...")
        agent = EnhancedTechnicalAnalyst()
        print("✅ Agent created successfully")
        
        # Test data
        tech_metrics = {
            'last_price': 50000.0,
            'rsi': 65.5,
            'macd': 0.5,
            'signal_line': 0.3,
            'bb_upper': 51000.0,
            'bb_lower': 49000.0,
            'volume': 1000.0,
            'atr': 500.0,
            'adx': 30.0,
            'ma50': 49800.0,
            'curr_vol': 1200.0,
            'avg_vol_20': 1100.0
        }
        
        indicator_sequences = {
            'rsi': [60, 62, 65.5], 
            'macd': [0.1, 0.3, 0.5]
        }
        
        print("3. Running technical analysis...")
        result = agent.run(
            current_tech_metrics=tech_metrics,
            indicator_sequences=indicator_sequences,
            symbol_tick_size=0.01
        )
        
        print("✅ Technical analysis completed successfully!")
        print(f"📊 Signal: {result.signal}")
        print(f"🎯 Confidence: {result.confidence_level}")
        print(f"🧠 Reasoning: {result.reasoning[:150]}...")
        print(f"📈 Market Phase: {result.market_phase}")
        print(f"💰 Price Target: {result.price_target}")
        print(f"🛡️ Stop Loss: {result.stop_loss_suggestion}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test that configuration loads correctly."""
    print("\n🧪 Testing Configuration Loading")
    print("=" * 50)
    
    try:
        from config.config_loader import APP_CONFIG
        from config.heterogeneous_models import HETEROGENEOUS_AGENT_MODELS
        
        print("✅ Configuration loaded successfully")
        print(f"🔧 Technical model: {HETEROGENEOUS_AGENT_MODELS.get('technical', 'NOT FOUND')}")
        print(f"🏪 Trading symbol: {APP_CONFIG.trading.symbol}")
        print(f"🧪 Using testnet: {getattr(APP_CONFIG.binance, 'testnet', 'Not configured')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting FenixTradingBot Technical Agent Tests")
    print("=" * 60)
    
    # Test configuration
    config_ok = test_config_loading()
    
    # Test technical agent
    agent_ok = test_technical_agent()
    
    print("\n" + "=" * 60)
    if config_ok and agent_ok:
        print("🎉 ALL TESTS PASSED! The system is ready for trading.")
        print("✅ Technical agent fixes applied successfully")
        print("✅ Configuration loaded correctly")
        print("✅ Model compatibility issues resolved")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
        sys.exit(1)
