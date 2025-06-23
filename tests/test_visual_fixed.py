#!/usr/bin/env python3
"""
Test rápido del Visual Analyst arreglado
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from collections import deque
from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_visual_analyst_fixed():
    """Test del visual analyst con fallback"""
    print("🔍 Testing Fixed Visual Analyst...")
    
    # Datos de prueba
    close_prices = deque([140.1, 140.2, 140.0, 140.3, 140.4] * 20, maxlen=100)
    high_prices = deque([140.3, 140.4, 140.2, 140.5, 140.6] * 20, maxlen=100)
    low_prices = deque([139.9, 140.0, 139.8, 140.1, 140.2] * 20, maxlen=100)
    volumes = deque([1000.0, 1200.0, 800.0, 1500.0, 1100.0] * 20, maxlen=100)
    
    tech_metrics = {
        'rsi': 65.0,
        'sma_20': 140.25,
        'sma_50': 140.15,
        'bb_upper': 140.8,
        'bb_lower': 139.8,
        'close': 140.3,
        'current_price': 140.3
    }
    
    try:
        visual_agent = EnhancedVisualAnalystAgent()
        print("✅ Visual agent creado")
        
        # Test análisis
        result = visual_agent.run(
            symbol="SOLUSDT",
            timeframe_str="1m",
            close_buf_deque=close_prices,
            high_buf_deque=high_prices,
            low_buf_deque=low_prices,
            vol_buf_deque=volumes,
            tech_metrics=tech_metrics
        )
        
        print("✅ Visual Analysis Complete:")
        print(f"   Assessment: {result.overall_visual_assessment}")
        print(f"   Clarity Score: {result.pattern_clarity_score}")
        print(f"   Suggested Action: {result.suggested_action_based_on_visuals}")
        print(f"   Timeframe: {result.chart_timeframe_analyzed}")
        print(f"   Main Elements: {result.main_elements_focused_on}")
        print(f"   Reasoning: {result.reasoning[:150]}...")
        
        if result.pattern_clarity_score and result.pattern_clarity_score > 0.0:
            print("✅ Visual analyst working correctly with fallback!")
        else:
            print("⚠️  Still getting 0.0 clarity score")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error completo: {e}", exc_info=True)

if __name__ == "__main__":
    test_visual_analyst_fixed()
