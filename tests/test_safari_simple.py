#!/usr/bin/env python3

"""
PRUEBA SIMPLE - VERIFICAR QUE SAFARI WEBDRIVER FUNCIONA
"""

try:
    from selenium import webdriver
    from selenium.webdriver.safari.options import Options as SafariOptions
    import time
    
    print("🧪 PRUEBA SIMPLE DE SAFARI WEBDRIVER")
    print("=" * 40)
    
    print("1️⃣ Creando driver de Safari...")
    
    # Configurar Safari
    safari_options = SafariOptions()
    driver = webdriver.Safari(options=safari_options)
    
    print("✅ Driver creado exitosamente")
    
    print("2️⃣ Navegando a Google...")
    driver.get("https://www.google.com")
    
    time.sleep(3)
    
    title = driver.title
    print(f"✅ Página cargada: {title}")
    
    print("3️⃣ Tomando screenshot...")
    screenshot = driver.get_screenshot_as_png()
    
    with open("test_screenshot.png", "wb") as f:
        f.write(screenshot)
    
    print("✅ Screenshot guardado como: test_screenshot.png")
    
    print("4️⃣ Cerrando navegador...")
    driver.quit()
    
    print("\n🎉 ¡ÉXITO! Safari WebDriver funciona correctamente")
    print("✅ Ahora puedes usar el script de TradingView")
    
except ImportError:
    print("❌ Selenium no está instalado")
    print("💡 Ejecuta: pip install selenium")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 POSIBLES SOLUCIONES:")
    print("1. Asegúrate de haber habilitado 'Desarrollo > Permitir automatización remota' en Safari")
    print("2. Ve a Safari > Preferencias > Avanzadas > ✅ Mostrar menú Desarrollo")
    print("3. Reinicia Terminal después de cambiar configuraciones de Safari")

if __name__ == "__main__":
    pass
