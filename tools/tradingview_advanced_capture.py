#!/usr/bin/env python3

"""
VERSIÓN MEJORADA: Captura específica de gráfico de velas de TradingView
"""

import time
import base64
from selenium import webdriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

def capture_tradingview_chart_advanced(symbol: str = "SOLUSDT", timeframe: str = "1"):
    """Captura avanzada con configuración automática del gráfico."""
    
    driver = None
    try:
        print(f"🚀 INICIANDO CAPTURA AVANZADA DE {symbol} ({timeframe}m)")
        print("=" * 60)
        
        # Configurar Safari
        safari_options = SafariOptions()
        driver = webdriver.Safari(options=safari_options)
        driver.set_window_size(1920, 1080)
        
        # URL directa al gráfico con configuración específica
        url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={timeframe}"
        print(f"📊 Navegando a: {url}")
        
        driver.get(url)
        
        # Esperar carga inicial
        print("⏳ Esperando carga inicial...")
        time.sleep(10)
        
        # Intentar cerrar cualquier popup inicial
        print("🚪 Cerrando popups iniciales...")
        try:
            close_buttons = [
                "[data-name='close']",
                ".tv-dialog__close",
                ".js-dialog__close",
                "[aria-label='Close']",
                ".close-button"
            ]
            
            for selector in close_buttons:
                try:
                    close_btn = driver.find_element(By.CSS_SELECTOR, selector)
                    if close_btn.is_displayed():
                        close_btn.click()
                        time.sleep(1)
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠️ No se encontraron popups para cerrar: {e}")
        
        # Asegurar que estamos viendo el gráfico de velas (candlestick)
        print("🕯️ Configurando vista de velas japonesas...")
        try:
            # Buscar botón de tipo de gráfico
            chart_type_selectors = [
                "[data-name='chart-style-button']",
                "[data-tooltip='Chart Type']",
                ".chart-style-button",
                "[title*='Chart']"
            ]
            
            for selector in chart_type_selectors:
                try:
                    chart_type_btn = driver.find_element(By.CSS_SELECTOR, selector)
                    if chart_type_btn.is_displayed():
                        chart_type_btn.click()
                        time.sleep(2)
                        
                        # Buscar opción de velas
                        candlestick_selectors = [
                            "[data-name='candlestick']",
                            "[title*='Candlestick']",
                            "[data-value='candlestick']"
                        ]
                        
                        for candle_selector in candlestick_selectors:
                            try:
                                candle_option = driver.find_element(By.CSS_SELECTOR, candle_selector)
                                if candle_option.is_displayed():
                                    candle_option.click()
                                    print("✅ Vista de velas configurada")
                                    time.sleep(3)
                                    break
                            except:
                                continue
                        break
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️ No se pudo configurar vista de velas: {e}")
        
        # Hacer el gráfico más grande/fullscreen si es posible
        print("🔍 Intentando maximizar gráfico...")
        try:
            fullscreen_selectors = [
                "[data-name='fullscreen']",
                "[title*='Fullscreen']",
                ".fullscreen-button"
            ]
            
            for selector in fullscreen_selectors:
                try:
                    fullscreen_btn = driver.find_element(By.CSS_SELECTOR, selector)
                    if fullscreen_btn.is_displayed():
                        fullscreen_btn.click()
                        time.sleep(2)
                        break
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️ No se pudo maximizar: {e}")
        
        # Ocultar elementos que no necesitamos
        print("🧹 Limpiando interfaz...")
        try:
            driver.execute_script("""
                // Ocultar elementos innecesarios
                const hideSelectors = [
                    '.tv-header',
                    '.tv-floating-toolbar',
                    '.tv-toast-logger',
                    '.tv-screener-popup',
                    '.tv-dialog',
                    '[data-name="header"]',
                    '.header-chart-panel'
                ];
                
                hideSelectors.forEach(selector => {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {
                        if (el) el.style.display = 'none';
                    });
                });
                
                // Hacer zoom al gráfico principal
                const chartContainer = document.querySelector('.chart-container, .tv-lightweight-charts, [data-name="chart"]');
                if (chartContainer) {
                    chartContainer.scrollIntoView({behavior: 'smooth', block: 'center'});
                }
            """)
            time.sleep(3)
        except Exception as e:
            print(f"⚠️ Error en limpieza: {e}")
        
        # Capturar screenshot final
        print("📸 Capturando screenshot final...")
        screenshot = driver.get_screenshot_as_png()
        
        # Guardar archivo
        filename = f"tradingview_advanced_{symbol}_{timeframe}m.png"
        with open(filename, "wb") as f:
            f.write(screenshot)
        
        # Convertir a base64
        img_b64 = base64.b64encode(screenshot).decode('utf-8')
        
        print(f"✅ ¡CAPTURA EXITOSA!")
        print(f"📏 Tamaño: {len(img_b64)} caracteres")
        print(f"💾 Archivo: {filename}")
        
        return img_b64
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""
        
    finally:
        if driver:
            driver.quit()
            print("🔒 Navegador cerrado")

if __name__ == "__main__":
    print("""
🎯 CAPTURA AVANZADA DE TRADINGVIEW
=================================

Este script va a:
✅ Ir directamente al gráfico de SOLUSDT
✅ Configurar vista de velas japonesas
✅ Cerrar popups automáticamente  
✅ Limpiar la interfaz
✅ Capturar screenshot limpio
✅ Guardar como imagen local

¡Prepárate para ver un gráfico profesional! 📈
""")
    
    result = capture_tradingview_chart_advanced("SOLUSDT", "1")
    
    if result:
        print("\n🎉 ¡ÉXITO TOTAL! Ahora tienes:")
        print("📸 Screenshot del gráfico real de TradingView")
        print("🔗 Imagen en base64 lista para el bot")
        print("🚀 ¡Tu bot puede ver gráficos reales ahora!")
    else:
        print("\n❌ Algo falló, pero vamos a intentar de nuevo...")
