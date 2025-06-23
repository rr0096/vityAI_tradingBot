#!/usr/bin/env python3

"""
CONFIGURACIÓN: Alternar entre chart generator tradicional y captura real de TradingView
"""

import os
from typing import Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# CONFIGURACIÓN GLOBAL
USE_REAL_TRADINGVIEW_CAPTURE = os.getenv("USE_REAL_TRADINGVIEW_CAPTURE", "true").lower() == "true"

def get_chart_for_visual_agent(
    symbol: str,
    timeframe: str,
    close_buf=None,
    high_buf=None,
    low_buf=None,
    vol_buf=None,
    tech_metrics=None,
    lookback_periods=None,
    save_chart: bool = True
) -> Tuple[Optional[str], str]:
    """
    Función unificada que decide qué método usar para generar gráficos.
    
    Se puede controlar con la variable de entorno:
    USE_REAL_TRADINGVIEW_CAPTURE=true  -> Usa captura real de TradingView
    USE_REAL_TRADINGVIEW_CAPTURE=false -> Usa chart generator tradicional
    """
    
    if USE_REAL_TRADINGVIEW_CAPTURE:
        logger.info(f"🎯 USANDO CAPTURA REAL DE TRADINGVIEW para {symbol}")
        try:
            from tools.chart_generator_real import generate_chart_for_visual_agent_real
            return generate_chart_for_visual_agent_real(
                symbol=symbol,
                timeframe=timeframe,
                save_chart=save_chart
            )
        except Exception as e:
            logger.error(f"❌ Error en captura real, fallback a tradicional: {e}")
            # Fallback al método tradicional
            USE_REAL_TRADINGVIEW_CAPTURE = False
    
    # Método tradicional (fallback)
    logger.info(f"📊 Usando chart generator tradicional para {symbol}")
    try:
        from tools.chart_generator import generate_chart_for_visual_agent
        return generate_chart_for_visual_agent(
            symbol=symbol,
            timeframe=timeframe,
            close_buf=close_buf,
            high_buf=high_buf,
            low_buf=low_buf,
            vol_buf=vol_buf,
            tech_metrics=tech_metrics or {},
            lookback_periods=lookback_periods or 100,
            save_chart=save_chart
        )
    except Exception as e:
        logger.error(f"❌ Error en chart generator tradicional: {e}")
        return None, ""

def enable_real_capture():
    """Habilitar captura real de TradingView."""
    global USE_REAL_TRADINGVIEW_CAPTURE
    USE_REAL_TRADINGVIEW_CAPTURE = True
    os.environ["USE_REAL_TRADINGVIEW_CAPTURE"] = "true"
    logger.info("✅ Captura real de TradingView HABILITADA")

def disable_real_capture():
    """Deshabilitar captura real, usar chart generator tradicional."""
    global USE_REAL_TRADINGVIEW_CAPTURE
    USE_REAL_TRADINGVIEW_CAPTURE = False
    os.environ["USE_REAL_TRADINGVIEW_CAPTURE"] = "false"
    logger.info("📊 Chart generator tradicional HABILITADO")

def get_current_mode():
    """Obtener el modo actual de generación de gráficos."""
    if USE_REAL_TRADINGVIEW_CAPTURE:
        return "🎯 CAPTURA REAL DE TRADINGVIEW"
    else:
        return "📊 CHART GENERATOR TRADICIONAL"

if __name__ == "__main__":
    print("🔧 CONFIGURACIÓN DE GRÁFICOS")
    print("=" * 40)
    print(f"Modo actual: {get_current_mode()}")
    print(f"Variable de entorno: {os.getenv('USE_REAL_TRADINGVIEW_CAPTURE', 'not set')}")
    
    print("\n🛠️ PARA CAMBIAR EL MODO:")
    print("export USE_REAL_TRADINGVIEW_CAPTURE=true   # Captura real")
    print("export USE_REAL_TRADINGVIEW_CAPTURE=false  # Tradicional")
    
    print("\n🧪 PROBANDO FUNCIÓN UNIFICADA...")
    
    chart_b64, filepath = get_chart_for_visual_agent("SOLUSDT", "1")
    
    if chart_b64:
        print(f"✅ ¡Funcionando! Archivo: {filepath}")
    else:
        print("❌ Error en la generación")
