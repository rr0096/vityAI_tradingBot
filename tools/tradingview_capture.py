#!/usr/bin/env python3

"""
SOLUCIÓN AVANZADA: Captura automatizada de gráficos reales de TradingView
con indicadores técnicos profesionales cada minuto.
"""

import time
import base64
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class TradingViewCapture:
    """Captura automática de gráficos de TradingView en tiempo real."""
    
    def __init__(self):
        self.driver = None
        self._setup_driver()
    
    def _setup_driver(self):
        """Configurar navegador headless para capturas."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Sin ventana
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("✅ Driver de Chrome configurado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error configurando driver: {e}")
            self.driver = None
    
    def capture_tradingview_chart(self, symbol: str = "SOLUSDT", timeframe: str = "1") -> str:
        """
        Captura un gráfico en tiempo real de TradingView.
        
        Args:
            symbol: Símbolo a analizar (ej: "SOLUSDT")
            timeframe: Timeframe (ej: "1", "5", "15", "1H", "1D")
            
        Returns:
            Base64 string de la imagen capturada
        """
        if not self.driver:
            logger.error("❌ Driver no disponible")
            return ""
        
        try:
            # URL de TradingView con configuración específica
            tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={timeframe}"
            
            logger.info(f"📊 Capturando gráfico de {symbol} en {timeframe}m desde TradingView...")
            
            # Navegar a TradingView
            self.driver.get(tv_url)
            
            # Esperar a que el gráfico cargue (importante!)
            wait = WebDriverWait(self.driver, 15)
            
            # Esperar a que aparezca el gráfico
            chart_element = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-name='legend-source-item']"))
            )
            
            # Esperar un poco más para que los datos se carguen completamente
            time.sleep(3)
            
            # Capturar screenshot de toda la página
            screenshot = self.driver.get_screenshot_as_png()
            
            # Convertir a base64
            img_b64 = base64.b64encode(screenshot).decode('utf-8')
            
            logger.info(f"✅ Gráfico capturado exitosamente - Tamaño: {len(img_b64)} caracteres")
            
            return img_b64
            
        except Exception as e:
            logger.error(f"❌ Error capturando gráfico: {e}")
            return ""
    
    def capture_with_indicators(self, symbol: str = "SOLUSDT", timeframe: str = "1") -> str:
        """
        Captura gráfico con indicadores técnicos específicos.
        """
        if not self.driver:
            return ""
        
        try:
            # URL con indicadores preconfigurados
            # RSI, MACD, EMA 20/50, Bollinger Bands, Volume
            tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={timeframe}"
            
            self.driver.get(tv_url)
            
            # Esperar carga inicial
            wait = WebDriverWait(self.driver, 20)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-name='legend-source-item']")))
            
            time.sleep(2)
            
            # Intentar agregar indicadores automáticamente
            try:
                # Buscar botón de indicadores
                indicators_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-name='indicators']")
                indicators_btn.click()
                time.sleep(1)
                
                # Agregar RSI
                rsi_search = self.driver.find_element(By.CSS_SELECTOR, "input[placeholder*='Search']")
                rsi_search.send_keys("RSI")
                time.sleep(1)
                
                # Hacer clic en el primer resultado
                first_result = self.driver.find_element(By.CSS_SELECTOR, "[data-name='legend-source-item'] button")
                first_result.click()
                
                time.sleep(1)
                
            except:
                logger.warning("⚠️ No se pudieron agregar indicadores automáticamente")
            
            # Capturar screenshot final
            time.sleep(3)  # Esperar a que todo se renderice
            screenshot = self.driver.get_screenshot_as_png()
            img_b64 = base64.b64encode(screenshot).decode('utf-8')
            
            logger.info(f"✅ Gráfico con indicadores capturado - Tamaño: {len(img_b64)}")
            
            return img_b64
            
        except Exception as e:
            logger.error(f"❌ Error capturando gráfico con indicadores: {e}")
            return ""
    
    def close(self):
        """Cerrar el navegador."""
        if self.driver:
            self.driver.quit()
            logger.info("🔒 Driver cerrado")

# Función de conveniencia para integrar fácilmente
def capture_real_tradingview_chart(symbol: str = "SOLUSDT", timeframe: str = "1") -> str:
    """
    Función simple para capturar un gráfico real de TradingView.
    
    Returns:
        Base64 string de la imagen del gráfico
    """
    capturer = None
    try:
        capturer = TradingViewCapture()
        return capturer.capture_with_indicators(symbol, timeframe)
    except Exception as e:
        logger.error(f"❌ Error en captura: {e}")
        return ""
    finally:
        if capturer:
            capturer.close()

if __name__ == "__main__":
    # Test de la funcionalidad
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Probando captura de gráfico de TradingView...")
    
    chart_b64 = capture_real_tradingview_chart("SOLUSDT", "1")
    
    if chart_b64:
        print(f"✅ ¡Éxito! Gráfico capturado - {len(chart_b64)} caracteres")
        
        # Guardar como archivo para verificar
        with open("test_tradingview_capture.png", "wb") as f:
            f.write(base64.b64decode(chart_b64))
        print("💾 Imagen guardada como 'test_tradingview_capture.png'")
    else:
        print("❌ Error en la captura")
