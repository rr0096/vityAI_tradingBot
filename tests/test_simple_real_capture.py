#!/usr/bin/env python3

"""
TEST SIMPLE: Probar solo la función de captura real
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    try:
        from tools.chart_generator_real import generate_chart_for_visual_agent_real
        
        print("🧪 PROBANDO CAPTURA REAL SIMPLE...")
        
        chart_b64, filepath = generate_chart_for_visual_agent_real(
            symbol="SOLUSDT",
            timeframe="1"
        )
        
        if chart_b64:
            print(f"✅ ¡ÉXITO! Captura funcionando")
            print(f"📏 Tamaño: {len(chart_b64)} caracteres")
            print(f"💾 Archivo: {filepath}")
            print("\n🎯 LISTO PARA INTEGRAR CON EL BOT")
        else:
            print("❌ Error en la captura")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
