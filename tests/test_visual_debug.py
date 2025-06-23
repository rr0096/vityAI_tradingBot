#!/usr/bin/env python3
"""
Debug script para diagnosticar problemas del Visual Analyst
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from collections import deque
from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent
from config.modern_models import ModelManager

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_visual_analyst():
    """Test básico del visual analyst"""
    print("🔍 Testing Visual Analyst...")
    
    # Datos de prueba simulados
    close_prices = deque([140.1, 140.2, 140.0, 140.3, 140.4, 140.1, 140.5, 140.2, 140.6, 140.3] * 10, maxlen=100)
    high_prices = deque([140.3, 140.4, 140.2, 140.5, 140.6, 140.3, 140.7, 140.4, 140.8, 140.5] * 10, maxlen=100)
    low_prices = deque([139.9, 140.0, 139.8, 140.1, 140.2, 139.9, 140.3, 140.0, 140.4, 140.1] * 10, maxlen=100)
    volumes = deque([1000.0, 1200.0, 800.0, 1500.0, 1100.0, 900.0, 1300.0, 1000.0, 1400.0, 1200.0] * 10, maxlen=100)
    
    tech_metrics = {
        'sma_20': 140.25,
        'sma_50': 140.15,
        'rsi': 55.0,
        'bb_upper': 140.8,
        'bb_lower': 139.8
    }
    
    try:
        # Inicializar ModelManager
        model_manager = ModelManager()
        print(f"✅ ModelManager inicializado con modelos: {model_manager.available_ollama_models}")
        
        # Crear agente visual
        visual_agent = EnhancedVisualAnalystAgent()
        print(f"✅ Visual agent creado: {visual_agent.name}")
        print(f"   Modelo configurado: {visual_agent._llm_model_name}")
        print(f"   Soporta visión: {visual_agent._supports_vision}")
        
        # Ejecutar análisis
        print("🚀 Ejecutando análisis visual...")
        result = visual_agent.run(
            symbol="SOLUSDT",
            timeframe_str="1m",
            close_buf_deque=close_prices,
            high_buf_deque=high_prices,
            low_buf_deque=low_prices,
            vol_buf_deque=volumes,
            tech_metrics=tech_metrics
        )
        
        print("✅ Resultado obtenido:")
        print(f"   Assessment: {result.overall_visual_assessment}")
        print(f"   Clarity Score: {result.pattern_clarity_score}")
        print(f"   Timeframe: {result.chart_timeframe_analyzed}")
        print(f"   Reasoning: {result.reasoning[:200]}...")
        
        if result.pattern_clarity_score == 0.0:
            print("⚠️  PROBLEMA: Clarity score es 0.0, indica error en el análisis")
        else:
            print("✅ Visual analyst funcionando correctamente")
            
    except Exception as e:
        print(f"❌ Error en test visual: {e}")
        logger.error(f"Error completo: {e}", exc_info=True)

if __name__ == "__main__":
    test_visual_analyst()
