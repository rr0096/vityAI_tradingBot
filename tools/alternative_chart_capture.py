#!/usr/bin/env python3

"""
PLAN B: Captura de gráfico usando Yahoo Finance o alternativas
En caso de que TradingView tenga demasiadas protecciones
"""

import time
import base64
from selenium import webdriver
from selenium.webdriver.safari.options import Options as SafariOptions

def capture_yahoo_finance_chart(symbol: str = "SOL-USD"):
    """Captura gráfico de Yahoo Finance como alternativa."""
    
    driver = None
    try:
        print(f"📊 CAPTURA DE YAHOO FINANCE: {symbol}")
        
        safari_options = SafariOptions()
        driver = webdriver.Safari(options=safari_options)
        driver.set_window_size(1600, 900)
        driver.set_page_load_timeout(10)
        
        # Yahoo Finance es más simple y rápido
        url = f"https://finance.yahoo.com/chart/{symbol}"
        print(f"🔗 URL: {url}")
        
        driver.get(url)
        time.sleep(5)  # Espera corta
        
        # Capturar inmediatamente
        screenshot = driver.get_screenshot_as_png()
        
        filename = f"yahoo_chart_{symbol.replace('-', '')}.png"
        with open(filename, "wb") as f:
            f.write(screenshot)
        
        img_b64 = base64.b64encode(screenshot).decode('utf-8')
        
        print(f"✅ Yahoo Finance capturado: {filename}")
        return img_b64
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""
        
    finally:
        if driver:
            driver.quit()

def capture_investing_com_chart(symbol: str = "solana"):
    """Captura de Investing.com como otra alternativa."""
    
    driver = None
    try:
        print(f"📊 CAPTURA DE INVESTING.COM: {symbol}")
        
        safari_options = SafariOptions()
        driver = webdriver.Safari(options=safari_options)
        driver.set_window_size(1600, 900)
        driver.set_page_load_timeout(10)
        
        # Investing.com suele ser más accesible
        url = f"https://www.investing.com/crypto/{symbol}/chart"
        print(f"🔗 URL: {url}")
        
        driver.get(url)
        time.sleep(6)
        
        # Scroll para centrar gráfico
        driver.execute_script("window.scrollTo(0, 500);")
        time.sleep(2)
        
        screenshot = driver.get_screenshot_as_png()
        
        filename = f"investing_chart_{symbol}.png"
        with open(filename, "wb") as f:
            f.write(screenshot)
        
        img_b64 = base64.b64encode(screenshot).decode('utf-8')
        
        print(f"✅ Investing.com capturado: {filename}")
        return img_b64
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""
        
    finally:
        if driver:
            driver.quit()

def test_multiple_sources():
    """Probar múltiples fuentes para ver cuál funciona mejor."""
    
    print("🧪 PROBANDO MÚLTIPLES FUENTES DE GRÁFICOS")
    print("=" * 50)
    
    sources = [
        ("Yahoo Finance", lambda: capture_yahoo_finance_chart("SOL-USD")),
        ("Investing.com", lambda: capture_investing_com_chart("solana")),
    ]
    
    for name, func in sources:
        print(f"\n🔄 Probando {name}...")
        try:
            result = func()
            if result:
                print(f"✅ {name} FUNCIONA!")
                return result
            else:
                print(f"❌ {name} falló")
        except Exception as e:
            print(f"❌ {name} error: {e}")
    
    print("\n⚠️ Ninguna fuente funcionó")
    return ""

if __name__ == "__main__":
    print("""
📊 FUENTES ALTERNATIVAS DE GRÁFICOS
===================================

Si TradingView tiene problemas, podemos usar:
1. Yahoo Finance (muy confiable)
2. Investing.com (buena alternativa)

¡Vamos a probar cuál funciona mejor!
""")
    
    result = test_multiple_sources()
    
    if result:
        print(f"\n🎉 ¡ÉXITO! Encontramos una fuente que funciona")
        print(f"📏 Tamaño del gráfico: {len(result)} caracteres")
    else:
        print("\n💡 SUGERENCIA: Tal vez necesitamos un enfoque diferente")
        print("   - Usar el chart generator interno mejorado")
        print("   - O configurar TradingView de manera diferente")
