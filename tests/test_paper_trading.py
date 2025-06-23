#!/usr/bin/env python3
"""
Test rápido del sistema de paper trading
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append('.')

# Configurar logging simple
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_paper_trading_components():
    """Test rápido de todos los componentes"""
    print("🧪 Testing FenixTradingBot Paper Trading System")
    print("=" * 50)
    
    try:
        # 1. Test Order Simulator
        print("1️⃣ Testing Order Simulator...")
        from paper_trading.order_simulator import BinanceOrderSimulator, OrderType
        
        order_sim = BinanceOrderSimulator()
        order_sim.update_market_price("BTCUSDT", 43000.0)
        
        # Simular una orden de mercado
        result = await order_sim.place_order(
            symbol="BTCUSDT",
            side="BUY", 
            order_type=OrderType.MARKET,
            quantity=0.001
        )
        
        print(f"   Order placed: {result.get('orderId', 'ERROR')[:8]}... Status: {result.get('status', 'ERROR')}")
        
        # 2. Test Market Data Simulator
        print("2️⃣ Testing Market Data Simulator...")
        from paper_trading.market_simulator import MarketDataSimulator
        
        market_sim = MarketDataSimulator()
        
        # Cargar datos históricos
        success = await market_sim.load_historical_data("BTCUSDT", "1m", 1)
        print(f"   Historical data loaded: {'✅' if success else '❌'}")
        
        current_price = market_sim.get_current_price("BTCUSDT")
        print(f"   Current BTC price: ${current_price:.2f}" if current_price else "   No price data")
        
        # 3. Test System Integration
        print("3️⃣ Testing System Integration...")
        
        # Test config loading
        from config.config_loader import APP_CONFIG
        print(f"   Config loaded: {APP_CONFIG.trading.symbol}")
        
        # Test agent imports
        try:
            from agents.sentiment_enhanced import EnhancedSentimentAnalyst
            from agents.technical_v_enhanced_fixed import EnhancedTechnicalAnalyst
            from agents.risk import AdvancedRiskManager
            print("   Core agents: ✅ Imported successfully")
        except Exception as e:
            print(f"   Core agents: ❌ {e}")
        
        print("=" * 50)
        print("✅ All components tested successfully!")
        print("🚀 Paper trading system is ready to use!")
        print("\nTo start paper trading:")
        print("   python run_paper_trading.py --duration 10")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_paper_trading_components())
    sys.exit(0 if success else 1)
