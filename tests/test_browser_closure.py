#!/usr/bin/env python3
"""
Test script para verificar que el navegador se cierra correctamente
"""

import logging
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_browser_closure():
    """Test que el navegador se cierre correctamente"""
    
    from tools.chart_generator_real import capture_bitget_quick
    
    print("🧪 Test: Verificando que el navegador se cierre correctamente")
    print("=" * 60)
    
    # Test 1: Captura normal
    print("\n1️⃣ Test captura normal...")
    
    result = capture_bitget_quick("SOL", "1")
    
    if result:
        print(f"✅ Captura exitosa - {len(result)} caracteres")
    else:
        print("❌ Captura falló")
    
    print("\n⏳ Esperando 5 segundos para verificar cierre...")
    time.sleep(5)
    
    # Test 2: Verificar que no hay procesos Safari colgados
    print("\n2️⃣ Verificando procesos Safari...")
    
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        safari_processes = [line for line in result.stdout.split('\n') if 'SafariDriver' in line or 'com.apple.WebKit' in line]
        
        print(f"Procesos Safari encontrados: {len(safari_processes)}")
        for proc in safari_processes[:3]:  # Solo mostrar los primeros 3
            print(f"  - {proc}")
            
    except Exception as e:
        print(f"❌ Error verificando procesos: {e}")
    
    print("\n✅ Test completado!")

if __name__ == "__main__":
    test_browser_closure()
