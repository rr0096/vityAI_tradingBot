#!/usr/bin/env python3

"""
CAPTURA ROBUSTA DE TRADINGVIEW - Con manejo de errores y recargas
"""

import time
import base64
import logging
from selenium import webdriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

logger = logging.getLogger(__name__)

class RobustTradingViewCapture:
    """Capturador robusto con manejo de errores y reintentos."""
    
    def __init__(self):
        self.driver = None
        self.max_retries = 3
        self._setup_driver()
    
    def _setup_driver(self):
        """Configurar Safari con configuraciones anti-detección."""
        try:
            safari_options = SafariOptions()
            
            self.driver = webdriver.Safari(options=safari_options)
            
            # Configurar ventana y comportamiento más natural
            self.driver.set_window_size(1920, 1080)
            self.driver.implicitly_wait(10)
            
            # Configurar user agent más natural
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            print("✅ Safari WebDriver configurado con protecciones anti-detección")
            
        except Exception as e:
            print(f"❌ Error configurando driver: {e}")
            self.driver = None
    
    def _reload_page_if_needed(self):
        """Detectar si la página necesita recarga y hacerlo."""
        try:
            # Verificar si hay errores comunes que requieren recarga
            error_indicators = [
                "blocked",
                "error",
                "retry",
                "refresh",
                "reload",
                "try again"
            ]
            
            page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            
            for indicator in error_indicators:
                if indicator in page_text:
                    print(f"⚠️ Detectado problema en página: {indicator}")
                    print("🔄 Recargando página...")
                    self.driver.refresh()
                    time.sleep(5)
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error verificando página: {e}")
            return False
    
    def _wait_for_page_load(self):
        """Esperar a que la página cargue completamente."""
        try:
            # Esperar a que el estado sea 'complete'
            wait = WebDriverWait(self.driver, 30)
            wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
            
            # Esperar un poco más para JavaScript
            time.sleep(3)
            
            # Verificar si necesita recarga
            if self._reload_page_if_needed():
                # Si recargamos, esperar de nuevo
                wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
                time.sleep(5)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error esperando carga: {e}")
            return False
    
    def _handle_popups_and_overlays(self):
        """Cerrar popups y overlays que pueden aparecer."""
        try:
            # Lista de selectores para cerrar popups
            close_selectors = [
                "[data-name='close']",
                ".tv-dialog__close",
                ".js-dialog__close",
                "[aria-label='Close']",
                ".close-button",
                "[data-dialog-name] .close",
                ".tv-button--ghost",
                ".tv-screener-popup__close"
            ]
            
            closed_count = 0
            for selector in close_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            self.driver.execute_script("arguments[0].click();", element)
                            closed_count += 1
                            time.sleep(1)
                except Exception:
                    continue
            
            if closed_count > 0:
                print(f"🚪 Cerrados {closed_count} popups/overlays")
            
            # También intentar presionar ESC para cerrar modales
            try:
                from selenium.webdriver.common.keys import Keys
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                time.sleep(1)
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ Error manejando popups: {e}")
    
    def capture_with_retries(self, symbol: str = "SOLUSDT", timeframe: str = "1") -> str:
        """Capturar con reintentos automáticos."""
        
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Intento {attempt + 1}/{self.max_retries} para {symbol}")
                
                result = self._single_capture_attempt(symbol, timeframe)
                
                if result:
                    print(f"✅ Captura exitosa en intento {attempt + 1}")
                    return result
                else:
                    print(f"❌ Falló intento {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        print("⏳ Esperando antes del siguiente intento...")
                        time.sleep(5)
                        
            except Exception as e:
                print(f"❌ Error en intento {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(5)
        
        print("❌ Todos los intentos fallaron")
        return ""
    
    def _single_capture_attempt(self, symbol: str, timeframe: str) -> str:
        """Un solo intento de captura."""
        
        if not self.driver:
            return ""
        
        try:
            # URL con parámetros específicos para mejor carga
            url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={timeframe}&hide_side_toolbar=1&hide_top_toolbar=1"
            
            print(f"📊 Navegando a TradingView: {symbol} ({timeframe}m)")
            
            self.driver.get(url)
            
            # Esperar carga completa
            if not self._wait_for_page_load():
                return ""
            
            print("⏳ Esperando que el gráfico se renderice...")
            time.sleep(8)
            
            # Manejar popups
            self._handle_popups_and_overlays()
            
            # Verificar que el gráfico esté presente
            chart_present = self._verify_chart_presence()
            if not chart_present:
                print("❌ No se detectó gráfico válido")
                return ""
            
            # Limpiar interfaz para mejor captura
            self._clean_interface()
            
            # Esperar un momento final
            time.sleep(3)
            
            # Capturar screenshot
            print("📸 Capturando screenshot...")
            screenshot = self.driver.get_screenshot_as_png()
            
            # Guardar archivo local
            filename = f"tradingview_robust_{symbol}_{timeframe}m_{int(time.time())}.png"
            with open(filename, "wb") as f:
                f.write(screenshot)
            
            print(f"💾 Guardado como: {filename}")
            
            # Convertir a base64
            img_b64 = base64.b64encode(screenshot).decode('utf-8')
            
            # Verificar que la imagen no esté vacía o corrupta
            if len(img_b64) < 1000:  # Imagen muy pequeña = problema
                print("❌ Imagen capturada demasiado pequeña")
                return ""
            
            print(f"✅ Captura exitosa: {len(img_b64)} caracteres")
            return img_b64
            
        except Exception as e:
            print(f"❌ Error en captura: {e}")
            return ""
    
    def _verify_chart_presence(self) -> bool:
        """Verificar que hay un gráfico válido en la página."""
        try:
            # Buscar múltiples indicadores de que el gráfico está presente
            chart_indicators = [
                ".tv-lightweight-charts",
                "[data-name='legend-source-item']",
                ".chart-container",
                ".chart-widget",
                "canvas[data-name='candle-series']",
                "canvas[width]"  # Canvas con ancho definido
            ]
            
            for selector in chart_indicators:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements and any(el.is_displayed() for el in elements):
                        print(f"✅ Gráfico detectado con: {selector}")
                        return True
                except:
                    continue
            
            print("❌ No se detectó gráfico válido")
            return False
            
        except Exception as e:
            print(f"⚠️ Error verificando gráfico: {e}")
            return False
    
    def _clean_interface(self):
        """Limpiar la interfaz para mejor captura."""
        try:
            self.driver.execute_script("""
                // Ocultar elementos innecesarios
                const hideSelectors = [
                    '.tv-header',
                    '.tv-floating-toolbar',
                    '.tv-toast-logger',
                    '.tv-screener-popup',
                    '.tv-dialog',
                    '[data-name="header"]',
                    '.header-chart-panel',
                    '.tv-header__area--left',
                    '.tv-header__area--right',
                    '.js-rootresizer__contents > div:first-child',
                    '.layout__area--top'
                ];
                
                hideSelectors.forEach(selector => {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {
                        if (el) el.style.display = 'none';
                    });
                });
                
                // Hacer que el área del gráfico use toda la pantalla
                const chartArea = document.querySelector('.layout__area--center, .chart-container');
                if (chartArea) {
                    chartArea.style.position = 'fixed';
                    chartArea.style.top = '0';
                    chartArea.style.left = '0';
                    chartArea.style.width = '100vw';
                    chartArea.style.height = '100vh';
                    chartArea.style.zIndex = '9999';
                }
            """)
            
            print("🧹 Interfaz limpiada para captura")
            
        except Exception as e:
            print(f"⚠️ Error limpiando interfaz: {e}")
    
    def close(self):
        """Cerrar el navegador."""
        if self.driver:
            try:
                self.driver.quit()
                print("🔒 Navegador cerrado")
            except:
                pass

def capture_tradingview_robust(symbol: str = "SOLUSDT", timeframe: str = "1") -> str:
    """Función principal para captura robusta."""
    
    capturer = None
    try:
        capturer = RobustTradingViewCapture()
        
        if not capturer.driver:
            print("❌ No se pudo inicializar el driver")
            return ""
        
        return capturer.capture_with_retries(symbol, timeframe)
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        return ""
        
    finally:
        if capturer:
            capturer.close()

if __name__ == "__main__":
    print("""
🛡️ CAPTURA ROBUSTA DE TRADINGVIEW
=================================

Esta versión incluye:
✅ Reintentos automáticos
✅ Detección de errores de página  
✅ Recarga automática si es necesario
✅ Mejor manejo de popups
✅ Verificación de gráfico válido
✅ Limpieza de interfaz
✅ Protecciones anti-detección

¡Probemos!
""")
    
    result = capture_tradingview_robust("SOLUSDT", "1")
    
    if result:
        print("\n🎉 ¡CAPTURA ROBUSTA EXITOSA!")
        print("📊 Gráfico capturado con todas las protecciones")
        print("🚀 Listo para integrar con el bot")
    else:
        print("\n❌ La captura robusta también falló")
        print("💡 Puede ser que TradingView esté bloqueando el acceso")
