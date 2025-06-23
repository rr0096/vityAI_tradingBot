#!/usr/bin/env python3

"""
Test rápido del agente visual con captura real
"""

import sys
import logging
from pathlib import Path
from collections import deque

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_visual_agent_real():
    """Test que el agente visual use captura real."""
    
    try:
        print("🎯 PROBANDO AGENTE VISUAL CON CAPTURA REAL")
        print("=" * 50)
        
        # Crear agente visual
        visual_agent = EnhancedVisualAnalystAgent(
            llm_model_name="llava:7b",  # Modelo con visión
            model_timeout=60,
            supports_vision=True,
            save_charts_to_disk=True
        )
        
        print("✅ Agente visual creado")
        
        # Crear buffers dummy (no se usan con captura real)
        close_buf = deque([134.5, 134.8, 135.2, 134.9, 135.1], maxlen=100)
        high_buf = deque([135.0, 135.2, 135.5, 135.1, 135.3], maxlen=100)
        low_buf = deque([134.2, 134.5, 134.8, 134.6, 134.8], maxlen=100)
        vol_buf = deque([1000, 1200, 1100, 1300, 1150], maxlen=100)
        
        # Ejecutar análisis visual
        print("📊 Ejecutando análisis visual con captura real...")
        
        result = visual_agent.run(
            symbol="SOLUSDT",
            timeframe_str="1m",
            close_buf_deque=close_buf,
            high_buf_deque=high_buf,
            low_buf_deque=low_buf,
            vol_buf_deque=vol_buf,
            tech_metrics={"last_price": 135.0, "rsi": 55, "atr": 2.5}
        )
        
        print(f"✅ Análisis completado!")
        print(f"📈 Resultado: {result.overall_visual_assessment}")
        print(f"🎯 Claridad: {result.pattern_clarity_score}")
        print(f"💭 Razonamiento: {result.reasoning[:100]}...")
        
        if result.pattern_clarity_score and result.pattern_clarity_score > 0:
            print("\n🎉 ¡ÉXITO! El agente visual está procesando gráficos reales")
            return True
        else:
            print("\n⚠️ El agente no está viendo gráficos claros")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_visual_agent_real()
    if success:
        print("\n🚀 ¡LISTO! El bot puede usar gráficos reales de TradingView")
        print("💡 Ahora ejecuta el live trading para ver la magia en acción")
    else:
        print("\n🔧 Hay que ajustar algo, pero estamos cerca")
