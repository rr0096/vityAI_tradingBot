#!/usr/bin/env python3

"""
CHART GENERATOR REAL - Reemplazo directo del chart generator tradicional
usando captura real de Bitget para el agente visual.
"""

import time
import base64
import logging
import atexit
from typing import Dict, Tuple, Optional, Any, Deque as TypingDeque
from selenium import webdriver
from selenium.webdriver.safari.options import Options as SafariOptions

logger = logging.getLogger(__name__)

class BitgetRealCapture:
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
                logger.info("🚀 Iniciando nuevo navegador Safari para Bitget...")
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

def capture_bitget_quick(symbol: str, timeframe: str = "1") -> str:
    """
    Captura simple de Bitget - va a la página, espera 10 segundos y toma captura.
    
    Args:
        symbol: Símbolo (ej: "SOLUSDT")
        timeframe: Timeframe (ej: "1", "5", "15")
        
    Returns:
        Base64 string de la imagen
    """
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By
    
    logger.info(f"🚀 INICIANDO capture_bitget_quick para {symbol}")
    
    capture_instance = BitgetRealCapture()
    driver = capture_instance.get_driver()
    
    if not driver:
        logger.error("❌ No se pudo obtener driver")
        return ""
    
    try:
        # URL directa de Bitget spot trading
        url = f"https://www.bitget.com/spot/{symbol}"
        
        logger.info(f"📊 Navegando a: {url}")
        
        # Navegar a la URL
        driver.get(url)
        logger.info("🌐 Página cargada, iniciando espera...")
        
        # Esperar un poco para que cargue la página
        logger.info("⏳ Esperando 5 segundos para carga inicial...")
        time.sleep(5)
        
        # Intentar poner el gráfico en pantalla completa con Shift+F
        logger.info("📊 Intentando poner gráfico en pantalla completa...")
        try:
            # Hacer click en la página para asegurar focus
            ActionChains(driver).click().perform()
            time.sleep(1)
            
            # Presionar Shift+F para pantalla completa
            ActionChains(driver).key_down(Keys.SHIFT).send_keys('f').key_up(Keys.SHIFT).perform()
            logger.info("🔍 Shift+F presionado para pantalla completa")
            
            # Esperar un momento para que se aplique el cambio
            time.sleep(3)
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo activar pantalla completa: {e}")
        
        # Intentar agregar indicadores técnicos
        logger.info("📈 Intentando agregar indicadores técnicos...")
        try:
            # Lista de selectores posibles para botón de indicadores
            indicator_selectors = [
                "button[data-testid='indicators']",
                "button[title*='指标']", 
                "button[title*='Indicator']",
                "button[title*='indicator']",
                ".indicators-button",
                "[data-role='button']:contains('指标')",
                "button:contains('Indicators')",
                ".chart-indicators-btn",
                "button[aria-label*='indicator']"
            ]
            
            indicator_button = None
            for selector in indicator_selectors:
                try:
                    if ":contains(" in selector:
                        # Para selectores que usan :contains, usar XPath
                        xpath_selector = selector.replace("button:contains('", "//button[contains(text(),'").replace("')", "')]")
                        indicator_button = driver.find_element(By.XPATH, xpath_selector)
                    else:
                        indicator_button = driver.find_element(By.CSS_SELECTOR, selector)
                    if indicator_button.is_displayed():
                        break
                except Exception:
                    continue
            
            if indicator_button:
                logger.info("🎯 Botón de indicadores encontrado, haciendo click...")
                ActionChains(driver).click(indicator_button).perform()
                time.sleep(2)
                
                # Intentar agregar RSI
                try:
                    rsi_selectors = [
                        "//div[contains(text(),'RSI')]",
                        "//span[contains(text(),'RSI')]",
                        "[data-testid='RSI']",
                        ".indicator-item:contains('RSI')"
                    ]
                    
                    for selector in rsi_selectors:
                        try:
                            if selector.startswith("//"):
                                rsi_element = driver.find_element(By.XPATH, selector)
                            else:
                                rsi_element = driver.find_element(By.CSS_SELECTOR, selector)
                            ActionChains(driver).click(rsi_element).perform()
                            logger.info("📊 RSI agregado")
                            time.sleep(1)
                            break
                        except Exception:
                            continue
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo agregar RSI: {e}")
                
                # Intentar agregar MACD
                try:
                    macd_selectors = [
                        "//div[contains(text(),'MACD')]",
                        "//span[contains(text(),'MACD')]",
                        "[data-testid='MACD']",
                        ".indicator-item:contains('MACD')"
                    ]
                    
                    for selector in macd_selectors:
                        try:
                            if selector.startswith("//"):
                                macd_element = driver.find_element(By.XPATH, selector)
                            else:
                                macd_element = driver.find_element(By.CSS_SELECTOR, selector)
                            ActionChains(driver).click(macd_element).perform()
                            logger.info("📈 MACD agregado")
                            time.sleep(1)
                            break
                        except Exception:
                            continue
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo agregar MACD: {e}")
                
                # Cerrar panel de indicadores si está abierto
                try:
                    # Hacer click fuera del panel o presionar ESC
                    ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                    time.sleep(1)
                except Exception:
                    pass
                    
            else:
                logger.warning("⚠️ No se encontró botón de indicadores")
                
        except Exception as e:
            logger.warning(f"⚠️ Error al agregar indicadores: {e}")
        
        # Esperar tiempo adicional para estabilizar
        logger.info("⏳ Esperando 3 segundos más para estabilizar...")
        time.sleep(3)
        
        logger.info("📸 Tomando screenshot...")
        # Hacer screenshot
        screenshot = driver.get_screenshot_as_png()
        
        if not screenshot:
            logger.error("❌ Screenshot vacío!")
            return ""
        
        logger.info(f"📸 Screenshot tomado - {len(screenshot)} bytes")
        
        # Convertir a base64
        img_b64 = base64.b64encode(screenshot).decode('utf-8')
        
        logger.info(f"✅ Captura exitosa - {len(img_b64)} caracteres base64")
        
        return img_b64
        
    except Exception as e:
        logger.error(f"❌ Error en captura: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return ""
    
    finally:
        # IMPORTANTE: SIEMPRE cerrar el navegador
        try:
            logger.info("🔒 Cerrando navegador...")
            capture_instance.close()
            logger.info("✅ Navegador cerrado exitosamente")
        except Exception as close_error:
            logger.error(f"❌ Error cerrando navegador: {close_error}")

def generate_chart_for_visual_agent_real(
    symbol: str,
    timeframe: str,
    close_buf: Optional[TypingDeque[float]],
    high_buf: Optional[TypingDeque[float]],
    low_buf: Optional[TypingDeque[float]],
    vol_buf: Optional[TypingDeque[float]],
    tech_metrics: Dict[str, Any],
    lookback_periods: int = 100,
    save_chart: bool = False
) -> Tuple[Optional[str], str]:
    """
    REEMPLAZO del generate_chart_for_visual_agent tradicional.
    
    En lugar de generar un gráfico sintético, captura uno real de Bitget.
    """
    
    logger.info(f"🎯 USANDO CAPTURA REAL DE BITGET para {symbol} ({timeframe})")
    logger.info(f"🔧 Parámetros: lookback_periods={lookback_periods}, save_chart={save_chart}")
    
    # Capturar gráfico real
    logger.info("📞 Llamando a capture_bitget_quick...")
    chart_b64 = capture_bitget_quick(symbol, timeframe)
    
    if not chart_b64:
        logger.error("❌ capture_bitget_quick retornó string vacío!")
        return None, ""
    
    logger.info(f"✅ capture_bitget_quick exitoso - Recibido: {len(chart_b64)} caracteres")
    
    chart_filepath = ""
    
    # Guardar localmente si se solicita
    if save_chart and chart_b64:
        try:
            from pathlib import Path
            
            charts_dir = Path("logs/charts_real")
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            chart_filepath = str(charts_dir / f"real_{symbol}_{timeframe}_{timestamp}.png")
            
            logger.info(f"💾 Guardando gráfico en: {chart_filepath}")
            
            with open(chart_filepath, "wb") as f:
                f.write(base64.b64decode(chart_b64))
                
            logger.info(f"💾 Gráfico real guardado exitosamente: {chart_filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando gráfico: {e}")
            import traceback
            logger.error(f"❌ Traceback guardado: {traceback.format_exc()}")
    
    logger.info(f"🎉 RETORNANDO AL AGENTE VISUAL - Base64: {len(chart_b64)} chars, Path: {chart_filepath}")
    
    return chart_b64, chart_filepath

# Función de limpieza para cuando termine la sesión
def cleanup_real_capture():
    """Cerrar navegador al finalizar."""
    try:
        capture_instance = BitgetRealCapture()
        capture_instance.close()
    except Exception:
        pass

# Auto-cleanup al importar
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
