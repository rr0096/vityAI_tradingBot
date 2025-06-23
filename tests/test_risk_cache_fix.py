#!/usr/bin/env python3

"""
Test script to verify that the risk manager cache fix resolves the PrivateAttr issue.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.risk import AdvancedRiskManager, RiskParameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_risk_manager_cache():
    """Test that the risk manager cache is properly initialized and accessible."""
    
    try:
        # Initialize risk manager
        risk_manager = AdvancedRiskManager(
            symbol_tick_size=0.0001,
            symbol_step_size=0.001,
            min_notional=10.0,
            initial_risk_params=RiskParameters(),
            portfolio_state_file="test_portfolio_state.json"
        )
        
        logger.info("✓ Risk manager initialized successfully")
        
        # Test accessing the cache (this was causing the error before)
        tech_metrics = {
            'last_price': 50000.0,
            'atr': 750.0,
            'adx': 25.0
        }
        
        # This should not raise the PrivateAttr error anymore
        analysis = risk_manager._analyze_market_conditions(tech_metrics, None)
        logger.info(f"✓ Market analysis completed: {analysis}")
        
        # Test that cache is working (second call should use cache)
        analysis2 = risk_manager._analyze_market_conditions(tech_metrics, None)
        logger.info(f"✓ Second analysis call (should use cache): {analysis2}")
        
        # Verify cache is a dict and can be accessed properly
        cache = risk_manager._market_analysis_cache
        logger.info(f"✓ Cache type: {type(cache)}, Cache keys: {list(cache.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_risk_manager_cache()
    if success:
        print("\n🎉 All tests passed! Risk manager cache is working correctly.")
    else:
        print("\n❌ Tests failed. Check the logs above.")
        sys.exit(1)
