#!/usr/bin/env python3
"""
Test script para verificar la captura de Bitget con indicadores técnicos
"""

import logging
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bitget_capture_with_indicators():
    """Test la captura de Bitget con indicadores técnicos"""
    try:
        from tools.chart_generator_real import capture_bitget_quick
        
        logger.info("🧪 Iniciando test de captura de Bitget con indicadores...")
        
        # Test con SOLUSDT
        symbol = "SOLUSDT"
        logger.info(f"📊 Probando captura para {symbol}...")
        
        # Llamar a la función de captura
        result = capture_bitget_quick(symbol, "1")
        
        if result:
            logger.info(f"✅ Captura exitosa! Longitud de imagen: {len(result)} caracteres")
            logger.info("🎯 La imagen debería incluir:")
            logger.info("   - Gráfico en pantalla completa") 
            logger.info("   - Indicador RSI")
            logger.info("   - Indicador MACD")
            return True
        else:
            logger.error("❌ La captura falló")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en el test: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Iniciando test de Bitget con indicadores técnicos...")
    
    success = test_bitget_capture_with_indicators()
    
    if success:
        logger.info("🎉 Test completado exitosamente!")
        sys.exit(0)
    else:
        logger.error("💥 Test falló")
        sys.exit(1)
