#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
from collections import deque

# Configurar el logging para ver todos los mensajes
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configurar el path
sys.path.append(str(Path(__file__).parent))

def test_visual_analysis_with_clarity():
    """Test que el agente visual devuelve clarity score válido"""
    try:
        # Import necesarios
        from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent
        
        print("Creando agente visual...")
        agent = EnhancedVisualAnalystAgent(
            model_name="llama3.2-vision:11b",
            temperature=0.1
        )
        
        # Crear datos mock
        close_prices = deque([148.0, 149.0, 150.0, 151.0, 150.5], maxlen=100)
        high_prices = deque([149.0, 150.0, 152.0, 152.5, 151.0], maxlen=100)
        low_prices = deque([147.0, 148.0, 149.5, 150.0, 149.0], maxlen=100)
        volumes = deque([1000000.0, 1100000.0, 1200000.0, 900000.0, 1050000.0], maxlen=100)
        
        print("Ejecutando análisis visual...")
        result = agent.run(
            symbol="SOL/USDT",
            timeframe_str="1m",
            close_buf_deque=close_prices,
            high_buf_deque=high_prices,
            low_buf_deque=low_prices,
            vol_buf_deque=volumes
        )
        
        print(f"Resultado: {result.overall_visual_assessment}")
        print(f"Clarity Score: {result.pattern_clarity_score}")
        print(f"Reasoning: {result.reasoning[:100]}...")
        
        # Verificar que clarity score no sea None y esté en rango válido
        assert result.pattern_clarity_score is not None, "Clarity score no debería ser None"
        assert 0.0 <= result.pattern_clarity_score <= 1.0, f"Clarity score {result.pattern_clarity_score} fuera de rango"
        
        print("✅ Test exitoso! El clarity score funciona correctamente.")
        return True
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Probando el fix del clarity score...")
    success = test_visual_analysis_with_clarity()
    
    if success:
        print("🎉 Todos los tests pasaron! El agente visual debería funcionar correctamente en live trading.")
    else:
        print("🚨 Hay problemas que resolver antes de usar en live trading.")
