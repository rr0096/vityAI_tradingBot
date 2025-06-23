#!/usr/bin/env python3
"""
Test completo del flujo visual agent + captura real Bitget
"""

import logging
import sys
import os
from collections import deque

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

def test_visual_agent_with_real_capture():
    """Test completo del agente visual con captura real."""
    
    logger.info("🧪 INICIANDO TEST COMPLETO - VISUAL AGENT + CAPTURA REAL")
    
    try:
        # Importar el agente visual
        from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent
        logger.info("✅ Agente visual importado")
        
        # Crear instancia del agente
        agent = EnhancedVisualAnalystAgent()
        logger.info("✅ Instancia del agente creada")
        
        # Datos de prueba simulados
        symbol = "SOLUSDT"
        timeframe = "1"
        
        # Crear buffers de prueba (simulados)
        close_buf = deque([100.5, 101.2, 99.8, 102.1, 103.5], maxlen=100)
        high_buf = deque([101.0, 102.0, 100.5, 103.0, 104.0], maxlen=100)  
        low_buf = deque([99.5, 100.8, 98.9, 101.5, 102.8], maxlen=100)
        vol_buf = deque([1000.0, 1200.0, 800.0, 1500.0, 1300.0], maxlen=100)
        
        tech_metrics = {
            "rsi": 65.5,
            "ma_20": 101.8,
            "ma_50": 100.2
        }
        
        logger.info(f"🎯 Analizando {symbol} con timeframe {timeframe}")
        
        # Llamar al agente visual
        result = agent.run(
            symbol=symbol,
            timeframe_str=timeframe,
            close_buf_deque=close_buf,
            high_buf_deque=high_buf,
            low_buf_deque=low_buf,
            vol_buf_deque=vol_buf,
            tech_metrics=tech_metrics
        )
        
        logger.info("📊 RESULTADO DEL ANÁLISIS VISUAL:")
        logger.info(f"- Assessment: {result.overall_visual_assessment}")
        logger.info(f"- Reasoning: {result.reasoning}")
        logger.info(f"- Suggested Action: {result.suggested_action_based_on_visuals}")
        
        if hasattr(result, 'key_candlestick_patterns') and result.key_candlestick_patterns:
            logger.info(f"- Candlestick Patterns: {result.key_candlestick_patterns}")
            
        if hasattr(result, 'chart_patterns') and result.chart_patterns:
            logger.info(f"- Chart Patterns: {result.chart_patterns}")
            
        logger.info("✅ TEST COMPLETADO EXITOSAMENTE")
        return True
        
    except Exception as e:
        logger.error(f"❌ ERROR EN TEST: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_visual_agent_with_real_capture()
    if success:
        print("\n🎉 TODOS LOS TESTS PASARON")
    else:
        print("\n💥 TESTS FALLARON")
        sys.exit(1)
