#!/usr/bin/env python3

"""
CAPTURA AUTOMÁTICA DE TRADINGVIEW CON DUCKDUCKGO BROWSER
Versión simplificada para prueba con Safari WebDriver (DuckDuckGo usa Safari en macOS)
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

class TradingViewCaptureDDG:
    """Captura automática de gráficos de TradingView usando Safari/DuckDuckGo."""
    
    def __init__(self):
        self.driver = None
        self._setup_safari_driver()
    
    def _setup_safari_driver(self):
        """Configurar Safari WebDriver (usado por DuckDuckGo en macOS)."""
        try:
            print("🦆 Configurando DuckDuckGo/Safari WebDriver...")
            
            # Safari options (DuckDuckGo usa Safari WebDriver en macOS)
            safari_options = SafariOptions()
            
            # Crear driver de Safari
            self.driver = webdriver.Safari(options=safari_options)
            
            # Configurar tamaño de ventana
            self.driver.set_window_size(1920, 1080)
            
            print("✅ Safari WebDriver configurado correctamente para DuckDuckGo")
            
        except WebDriverException as e:
            print(f"❌ Error configurando Safari WebDriver: {e}")
            print("💡 Tip: Asegúrate de que 'Desarrollo' > 'Permitir automatización remota' esté habilitado en Safari")
            self.driver = None
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            self.driver = None
    
    def capture_tradingview_simple(self, symbol: str = "SOLUSDT", timeframe: str = "1") -> str:
        """
        Captura un gráfico completo de TradingView con velas japonesas.
        
        Args:
            symbol: Símbolo a analizar (ej: "SOLUSDT")
            timeframe: Timeframe en minutos (ej: "1", "5", "15")
            
        Returns:
            Base64 string de la imagen capturada
        """
        if not self.driver:
            print("❌ Driver no disponible")
            return ""
        
        try:
            # URL CORREGIDA: Vamos directamente al gráfico completo con parámetros específicos
            tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={timeframe}"
            
            print(f"📊 Navegando al gráfico completo de {symbol} ({timeframe}m)...")
            print(f"🔗 URL: {tv_url}")
            
            # Navegar a TradingView
            self.driver.get(tv_url)
            
            # Esperar a que la página cargue
            print("⏳ Esperando que TradingView cargue completamente...")
            wait = WebDriverWait(self.driver, 30)
            
            # Esperar a que aparezca el gráfico principal (selectors más específicos)
            try:
                # Intentar múltiples selectores para encontrar el gráfico
                chart_selectors = [
                    "[data-name='legend-source-item']",
                    ".chart-container",
                    "#chart-container", 
                    ".tv-lightweight-charts",
                    ".chart-widget",
                    "[class*='chart']"
                ]
                
                chart_found = False
                for selector in chart_selectors:
                    try:
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        print(f"✅ Gráfico encontrado con selector: {selector}")
                        chart_found = True
                        break
                    except TimeoutException:
                        continue
                
                if not chart_found:
                    print("⚠️ No se encontró el gráfico específico, continuando...")
                    
            except Exception as e:
                print(f"⚠️ Error buscando gráfico: {e}")
            
            # Esperar más tiempo para que se carguen las velas
            print("⏳ Esperando carga completa de datos y velas...")
            time.sleep(8)
            
            # REMOVER ELEMENTOS QUE PUEDEN ESTAR ENCIMA DEL GRÁFICO
            try:
                # Remover popups, banners, anuncios
                popup_selectors = [
                    ".tv-dialog__modal-wrap",
                    ".js-popup-widget",
                    "[data-name='popup']",
                    ".tv-header",
                    ".tv-screener-popup",
                    ".tv-toast-logger",
                    ".tv-floating-toolbar"
                ]
                
                for selector in popup_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            self.driver.execute_script("arguments[0].style.display = 'none';", element)
                    except:
                        pass
                        
                print("🧹 Elementos superpuestos removidos")
                
            except Exception as e:
                print(f"⚠️ Error removiendo popups: {e}")
            
            # Hacer scroll específico para centrar el gráfico
            print("📍 Centrando gráfico en pantalla...")
            self.driver.execute_script("""
                // Buscar el contenedor del gráfico y hacer scroll hacia él
                const chartSelectors = [
                    '[data-name="legend-source-item"]',
                    '.chart-container',
                    '.tv-lightweight-charts',
                    '.chart-widget'
                ];
                
                for (let selector of chartSelectors) {
                    const chartElement = document.querySelector(selector);
                    if (chartElement) {
                        chartElement.scrollIntoView({behavior: 'smooth', block: 'center'});
                        break;
                    }
                }
                
                // Scroll adicional para centrar mejor
                window.scrollBy(0, -100);
            """)
            
            time.sleep(3)
            
            # Capturar screenshot
            print("📸 Capturando screenshot...")
            screenshot = self.driver.get_screenshot_as_png()
            
            # Convertir a base64
            img_b64 = base64.b64encode(screenshot).decode('utf-8')
            
            print(f"✅ ¡Captura exitosa! Tamaño: {len(img_b64)} caracteres")
            
            # Guardar copia local para verificar
            with open(f"tradingview_capture_{symbol}_{timeframe}m.png", "wb") as f:
                f.write(screenshot)
            print(f"💾 Imagen guardada como: tradingview_capture_{symbol}_{timeframe}m.png")
            
            return img_b64
            
        except Exception as e:
            print(f"❌ Error durante la captura: {e}")
            
            # Intentar capturar screenshot de emergencia
            try:
                emergency_screenshot = self.driver.get_screenshot_as_png()
                with open("emergency_screenshot.png", "wb") as f:
                    f.write(emergency_screenshot)
                print("🚨 Screenshot de emergencia guardado como: emergency_screenshot.png")
            except:
                pass
                
            return ""
    
    def test_basic_navigation(self):
        """Prueba básica para verificar que el navegador funciona."""
        if not self.driver:
            return False
            
        try:
            print("🧪 Prueba básica de navegación...")
            
            # Ir a una página simple primero
            self.driver.get("https://www.google.com")
            time.sleep(3)
            
            title = self.driver.title
            print(f"✅ Navegación exitosa. Título: {title}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en prueba básica: {e}")
            return False
    
    def close(self):
        """Cerrar el navegador."""
        if self.driver:
            try:
                self.driver.quit()
                print("🔒 Navegador cerrado correctamente")
            except:
                print("⚠️ Error cerrando navegador (probablemente ya estaba cerrado)")

def test_duckduckgo_capture():
    """Función de prueba principal."""
    print("🦆 INICIANDO PRUEBA DE CAPTURA CON DUCKDUCKGO/SAFARI")
    print("=" * 60)
    
    capturer = None
    try:
        # Crear capturador
        capturer = TradingViewCaptureDDG()
        
        if not capturer.driver:
            print("❌ No se pudo inicializar el driver")
            return
        
        # Prueba básica
        if not capturer.test_basic_navigation():
            print("❌ Falló la prueba básica de navegación")
            return
        
        # Capturar gráfico de TradingView
        print("\n" + "=" * 60)
        print("📊 CAPTURANDO GRÁFICO DE TRADINGVIEW")
        print("=" * 60)
        
        chart_b64 = capturer.capture_tradingview_simple("SOLUSDT", "1")
        
        if chart_b64:
            print(f"\n🎉 ¡ÉXITO TOTAL! Gráfico capturado correctamente")
            print(f"📏 Tamaño de imagen: {len(chart_b64)} caracteres")
            print(f"💾 Revisa el archivo: tradingview_capture_SOLUSDT_1m.png")
        else:
            print("\n❌ No se pudo capturar el gráfico")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        
    finally:
        if capturer:
            capturer.close()

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    print("""
🦆 CAPTURA AUTOMÁTICA DE TRADINGVIEW CON DUCKDUCKGO
==================================================

Este script va a:
1. Abrir Safari (que usa DuckDuckGo)
2. Navegar a TradingView 
3. Capturar un gráfico de SOLUSDT
4. Guardar la imagen localmente
5. Devolver base64 para el bot

¡Prepárate para la magia! ✨
""")
    
    input("Presiona ENTER para comenzar...")
    
    test_duckduckgo_capture()
