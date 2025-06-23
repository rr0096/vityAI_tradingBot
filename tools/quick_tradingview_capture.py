#!/usr/bin/env python3

"""
VERSIÓN RÁPIDA Y SIMPLE - Sin esperas largas
"""

import time
import base64
from selenium import webdriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.common.by import By

def quick_tradingview_capture(symbol: str = "SOLUSDT", timeframe: str = "1"):
    """Captura rápida sin esperas largas."""
    
    driver = None
    try:
        print(f"⚡ CAPTURA RÁPIDA DE {symbol} ({timeframe}m)")
        
        # Configurar Safari rápido
        safari_options = SafariOptions()
        driver = webdriver.Safari(options=safari_options)
        driver.set_window_size(1600, 900)
        
        # Timeout más corto para no esperar tanto
        driver.set_page_load_timeout(15)  # Solo 15 segundos máximo
        
        # URL directa
        url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={timeframe}"
        print(f"📊 Navegando (timeout 15s): {url}")
        
        try:
            driver.get(url)
        except Exception as e:
            print(f"⚠️ Timeout de carga, pero continuamos: {e}")
        
        # Espera corta y fija (no elementos específicos)
        print("⏳ Espera fija de 8 segundos...")
        time.sleep(8)
        
        # Script simple para limpiar
        try:
            driver.execute_script("""
                // Ocultar elementos molestos rápidamente
                const hide = ['.tv-header', '.tv-dialog', '.tv-screener-popup'];
                hide.forEach(s => {
                    const els = document.querySelectorAll(s);
                    els.forEach(el => el.style.display = 'none');
                });
            """)
        except:
            print("⚠️ No se pudo ejecutar script de limpieza")
        
        # Scroll simple
        try:
            driver.execute_script("window.scrollTo(0, 300);")
        except:
            pass
            
        time.sleep(2)
        
        # Capturar inmediatamente
        print("📸 Capturando...")
        screenshot = driver.get_screenshot_as_png()
        
        # Guardar
        filename = f"quick_capture_{symbol}_{timeframe}m.png"
        with open(filename, "wb") as f:
            f.write(screenshot)
        
        img_b64 = base64.b64encode(screenshot).decode('utf-8')
        
        print(f"✅ Captura rápida completada!")
        print(f"📏 Tamaño: {len(img_b64)} caracteres")
        print(f"💾 Archivo: {filename}")
        
        return img_b64
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""
        
    finally:
        if driver:
            try:
                driver.quit()
                print("🔒 Navegador cerrado")
            except:
                pass

if __name__ == "__main__":
    print("⚡ CAPTURA RÁPIDA SIN ESPERAS LARGAS")
    print("=" * 40)
    
    result = quick_tradingview_capture("SOLUSDT", "1")
    
    if result:
        print("\n🎉 ¡ÉXITO! Captura completada en menos de 30 segundos")
    else:
        print("\n❌ Error en captura rápida")
