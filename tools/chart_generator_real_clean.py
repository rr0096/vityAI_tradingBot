#!/usr/bin/env python3

"""
CHART GENERATOR REAL - Reemplazo directo del chart generator tradicional
usando captura real de TradingView para el agente visual.
"""

import time
import base64
import logging
from typing import Dict, Tuple, Optional, Any, Deque as TypingDeque
from selenium import webdriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.common.exceptions import WebDriverException

logger = logging.getLogger(__name__)

class TradingViewRealCapture:
    """Singleton para mantener una sola instancia del navegador activa."""
    
    _instance = None
    _driver = None
    _last_used = 0
    _timeout_seconds = 300  # 5 minutos
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_driver(self):
        """Obtener driver, creando uno nuevo si es necesario."""
        current_time = time.time()
        
        # Si el driver es muy viejo o no existe, crear uno nuevo
        if (self._driver is None or 
            current_time - self._last_used > self._timeout_seconds):
            
            if self._driver:
                try:
                    self._driver.quit()
                except Exception:
                    pass
            
            try:
                logger.info("🚀 Iniciando nuevo navegador Safari para TradingView...")
                safari_options = SafariOptions()
                self._driver = webdriver.Safari(options=safari_options)
                self._driver.set_window_size(1400, 900)
                logger.info("✅ Navegador Safari inicializado")
                
            except Exception as e:
                logger.error(f"❌ Error inicializando Safari: {e}")
                self._driver = None
                return None
        
        self._last_used = current_time
        return self._driver
    
    def close(self):
        """Cerrar navegador."""
        if self._driver:
            try:
                self._driver.quit()
                logger.info("🔒 Navegador cerrado")
            except Exception:
                pass
            self._driver = None

def capture_tradingview_quick(symbol: str, timeframe: str = "1") -> str:
    """
    Captura rápida de TradingView para el agente visual.
    
    Args:
        symbol: Símbolo (ej: "SOLUSDT")
        timeframe: Timeframe (ej: "1", "5", "15")
        
    Returns:
        Base64 string de la imagen
    """
    capture_instance = TradingViewRealCapture()
    driver = capture_instance.get_driver()
    
    if not driver:
        logger.error("❌ No se pudo obtener driver")
        return ""
    
    try:
        # URL directa con configuración específica
        url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={timeframe}"
        
        logger.info(f"📊 Capturando {symbol} ({timeframe}m) desde TradingView...")
        
        # Navegar (reutilizando sesión si es posible)
        current_url = driver.current_url
        if symbol not in current_url or timeframe not in current_url:
            driver.get(url)
            time.sleep(6)  # Tiempo de carga inicial
        else:
            logger.info("🔄 Reutilizando sesión existente...")
            time.sleep(2)  # Tiempo mínimo para actualización
        
        # Hacer screenshot rápido
        screenshot = driver.get_screenshot_as_png()
        
        # Convertir a base64
        img_b64 = base64.b64encode(screenshot).decode('utf-8')
        
        logger.info(f"✅ Captura exitosa - {len(img_b64)} caracteres")
        
        return img_b64
        
    except Exception as e:
        logger.error(f"❌ Error en captura: {e}")
        return ""

def generate_chart_for_visual_agent_real(
    symbol: str,
    timeframe: str,
    close_buf: TypingDeque[float],
    high_buf: TypingDeque[float],
    low_buf: TypingDeque[float],
    vol_buf: TypingDeque[float],
    tech_metrics: Dict[str, Any],
    lookback_periods: int = 100,
    save_chart: bool = False
) -> Tuple[Optional[str], str]:
    """
    REEMPLAZO del generate_chart_for_visual_agent tradicional.
    
    En lugar de generar un gráfico sintético, captura uno real de TradingView.
    """
    
    logger.info(f"🎯 USANDO CAPTURA REAL DE TRADINGVIEW para {symbol} ({timeframe})")
    
    # Capturar gráfico real
    chart_b64 = capture_tradingview_quick(symbol, timeframe)
    
    chart_filepath = ""
    
    # Guardar localmente si se solicita
    if save_chart and chart_b64:
        try:
            from pathlib import Path
            
            charts_dir = Path("logs/charts_real")
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            chart_filepath = str(charts_dir / f"real_{symbol}_{timeframe}_{timestamp}.png")
            
            with open(chart_filepath, "wb") as f:
                f.write(base64.b64decode(chart_b64))
                
            logger.info(f"💾 Gráfico real guardado: {chart_filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando gráfico: {e}")
    
    if not chart_b64:
        logger.error("❌ No se pudo capturar gráfico real")
        return None, ""
    
    logger.info(f"✅ Gráfico real capturado exitosamente - Tamaño: {len(chart_b64)}")
    
    return chart_b64, chart_filepath

# Función de limpieza para cuando termine la sesión
def cleanup_real_capture():
    """Cerrar navegador al finalizar."""
    try:
        capture_instance = TradingViewRealCapture()
        capture_instance.close()
    except Exception:
        pass

# Auto-cleanup al importar
import atexit
atexit.register(cleanup_real_capture)

if __name__ == "__main__":
    # Test rápido
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Probando captura real para agente visual...")
    
    chart_b64, filepath = generate_chart_for_visual_agent_real(
        symbol="SOLUSDT",
        timeframe="1",
        close_buf=None,  # No usado
        high_buf=None,   # No usado  
        low_buf=None,    # No usado
        vol_buf=None,    # No usado
        tech_metrics={}, # No usado
        save_chart=True
    )
    
    if chart_b64:
        print(f"✅ ¡Éxito! Gráfico capturado: {len(chart_b64)} caracteres")
        if filepath:
            print(f"💾 Guardado en: {filepath}")
    else:
        print("❌ Error en la captura")
