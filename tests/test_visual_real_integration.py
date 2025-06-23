#!/usr/bin/env python3

"""
TEST: Verificar que el agente visual funciona con capturas reales de TradingView
"""

import sys
import logging
from pathlib import Path
from collections import deque

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_visual_agent_with_real_capture():
    """Test del agente visual con captura real de TradingView."""
    
    try:
        # Importar después de configurar el path
        from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent
        
        print("🧪 PROBANDO AGENTE VISUAL CON CAPTURA REAL")
        print("=" * 50)
        
        # Crear agente visual
        print("1️⃣ Inicializando agente visual...")
        visual_agent = EnhancedVisualAnalystAgent(
            llm_model_name="qwen2.5:7b-instruct-q5_k_m",  # Cualquier modelo
            supports_vision=True,  # Importante: habilitar visión
            save_charts_to_disk=True
        )
        
        print("✅ Agente visual creado")
        
        # Crear datos ficticios (no se usan en captura real)
        print("2️⃣ Preparando datos de prueba...")
        close_buf = deque([130.5, 131.0, 132.0, 132.5, 133.0], maxlen=100)
        high_buf = deque([131.0, 131.5, 132.5, 133.0, 133.5], maxlen=100)
        low_buf = deque([130.0, 130.5, 131.5, 132.0, 132.5], maxlen=100)
        vol_buf = deque([1000, 1100, 1200, 1300, 1400], maxlen=100)
        
        tech_metrics = {
            "last_price": 133.0,
            "rsi": 65.0,
            "macd": 0.5
        }
        
        print("✅ Datos preparados")
        
        # PROBAR CAPTURA REAL
        print("3️⃣ 🎯 EJECUTANDO CAPTURA REAL DE TRADINGVIEW...")
        print("   (Esto va a abrir Safari y capturar el gráfico real)")
        
        try:
            result = visual_agent.run(
                symbol="SOLUSDT",
                timeframe_str="1",
                close_buf_deque=close_buf,
                high_buf_deque=high_buf,
                low_buf_deque=low_buf,
                vol_buf_deque=vol_buf,
                tech_metrics=tech_metrics
            )
            
            print("✅ ¡CAPTURA COMPLETADA!")
            print(f"📊 Resultado del análisis visual:")
            print(f"   • Evaluación: {result.overall_visual_assessment}")
            print(f"   • Patrones: {result.key_candlestick_patterns}")
            print(f"   • Claridad: {result.pattern_clarity_score}")
            print(f"   • Sugerencia: {result.suggested_action_based_on_visuals}")
            print(f"   • Razonamiento: {result.reasoning[:100]}...")
            
            if result.overall_visual_assessment != "NEUTRAL" or result.pattern_clarity_score is not None:
                print("\n🎉 ¡ÉXITO! El agente visual está analizando gráficos reales")
                return True
            else:
                print("\n⚠️ El agente respondió pero con valores por defecto")
                return False
                
        except Exception as e:
            print(f"❌ Error durante la captura: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return False
    except Exception as e:
        print(f"❌ Error general: {e}")
        return False

if __name__ == "__main__":
    print("""
🎯 TEST DE INTEGRACIÓN: AGENTE VISUAL + CAPTURA REAL
===================================================

Este test va a:
✅ Crear el agente visual
✅ Configurar captura real de TradingView  
✅ Capturar gráfico de SOLUSDT en tiempo real
✅ Analizar el gráfico con IA
✅ Mostrar resultados del análisis

¡Prepárate para ver magia! ✨
""")
    
    input("Presiona ENTER para comenzar el test...")
    
    success = test_visual_agent_with_real_capture()
    
    if success:
        print("\n🚀 ¡INTEGRACIÓN EXITOSA!")
        print("Tu bot ahora puede:")
        print("📈 Ver gráficos reales de TradingView")
        print("🕯️ Analizar velas japonesas reales")
        print("📊 Detectar patrones en tiempo real")
        print("🎯 Tomar decisiones basadas en gráficos reales")
        print("\n🎉 ¡EL FUTURO DEL TRADING AUTOMATIZADO!")
    else:
        print("\n❌ Hay algunos problemas, pero vamos a resolverlos...")
        print("💡 Revisar logs para más detalles")
