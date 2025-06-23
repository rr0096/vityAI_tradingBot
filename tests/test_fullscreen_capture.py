#!/usr/bin/env python3

import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.chart_generator_real import generate_chart_for_visual_agent_real

def test_fullscreen_capture():
    """Test de captura con pantalla completa"""
    
    print("🧪 Probando captura con pantalla completa...")
    
    try:
        # Probar con SOL/USDT - pasar None para los buffers que no necesitamos
        result = generate_chart_for_visual_agent_real(
            "SOLUSDT", 
            "5m", 
            None,  # close_buf
            None,  # high_buf
            None,  # low_buf
            None,  # vol_buf
            {}     # tech_metrics vacío
        )
        
        if result and len(result) > 100:
            print(f"✅ Captura exitosa - {len(result)} caracteres base64")
            print("📊 Captura con pantalla completa completada")
            return True
        else:
            print("❌ Captura falló")
            return False
            
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False

if __name__ == "__main__":
    test_fullscreen_capture()
