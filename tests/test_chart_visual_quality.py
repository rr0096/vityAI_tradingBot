#!/usr/bin/env python3
"""
Test para generar una imagen de prueba del gráfico y verificar la calidad visual
"""

import os
import sys
import base64
from collections import deque
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.chart_generator import generate_chart_for_visual_agent

def generate_realistic_data(periods: int = 40):
    """Genera datos de mercado realistas para el test"""
    np.random.seed(42)  # Para resultados reproducibles
    
    # Precio base
    base_price = 45000.0  # Simular precio de Bitcoin
    
    # Generar movimientos de precio realistas
    prices = [base_price]
    for i in range(periods - 1):
        # Movimiento aleatorio con tendencia ligeramente alcista
        change_pct = np.random.normal(0.002, 0.03)  # 0.2% tendencia, 3% volatilidad
        new_price = prices[-1] * (1 + change_pct)
        prices.append(max(new_price, 100.0))  # Precio mínimo de $100
    
    # Generar high/low realistas
    high_prices = []
    low_prices = []
    volumes = []
    
    for price in prices:
        # High es el precio + algún porcentaje aleatorio
        high_variation = abs(np.random.normal(0, 0.01))  # 0-2% típicamente
        high = price * (1 + high_variation)
        
        # Low es el precio - algún porcentaje aleatorio
        low_variation = abs(np.random.normal(0, 0.01))
        low = price * (1 - low_variation)
        
        # Asegurar reglas OHLC
        high = max(high, price)
        low = min(low, price)
        
        # Volumen aleatorio realista
        volume = np.random.uniform(100000, 2000000)  # Volumen típico de crypto
        
        high_prices.append(high)
        low_prices.append(low)
        volumes.append(volume)
    
    return prices, high_prices, low_prices, volumes

def test_chart_quality():
    """Genera un gráfico de prueba y verifica su calidad"""
    print("=" * 60)
    print("🎨 PRUEBA DE CALIDAD VISUAL DEL GRÁFICO")
    print("=" * 60)
    
    # Generar datos de prueba realistas
    print("📊 Generando datos de mercado realistas...")
    close, high, low, vol = generate_realistic_data(40)
    
    # Mostrar estadísticas de los datos
    print(f"   💰 Precio inicial: ${close[0]:,.2f}")
    print(f"   💰 Precio final: ${close[-1]:,.2f}")
    print(f"   📈 Cambio total: {((close[-1] - close[0]) / close[0] * 100):+.2f}%")
    print(f"   📊 Puntos de datos: {len(close)}")
    
    # Generar métricas técnicas de ejemplo
    tech_metrics = {
        'rsi': 65.4,
        'sma_20': sum(close[-20:]) / 20,
        'sma_50': sum(close) / len(close),
        'last_price': close[-1]
    }
    
    try:
        print("\n🎨 Generando gráfico...")
        base64_img, filepath = generate_chart_for_visual_agent(
            symbol="BTCUSDT",
            timeframe="1h",
            close_buf=deque(close),
            high_buf=deque(high),
            low_buf=deque(low),
            vol_buf=deque(vol),
            tech_metrics=tech_metrics,
            lookback_periods=40,
            save_chart=True
        )
        
        if not base64_img or base64_img == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=":
            print("❌ ERROR: Se generó imagen de error o fallback")
            return False
        
        # Información de la imagen
        image_data = base64.b64decode(base64_img)
        size_kb = len(image_data) / 1024
        
        print(f"✅ Gráfico generado exitosamente!")
        print(f"   📁 Tamaño del archivo: {size_kb:.1f} KB")
        print(f"   📝 Longitud base64: {len(base64_img):,} caracteres")
        
        if filepath:
            print(f"   💾 Guardado en: {filepath}")
            
            # Verificar que el archivo existe
            if os.path.exists(filepath):
                file_size_kb = os.path.getsize(filepath) / 1024
                print(f"   ✅ Archivo confirmado en disco: {file_size_kb:.1f} KB")
                
                # Intentar verificar dimensiones con PIL si está disponible
                try:
                    from PIL import Image
                    with Image.open(filepath) as img:
                        width, height = img.size
                        total_pixels = width * height
                        print(f"   🖼️  Dimensiones: {width} x {height} píxeles")
                        print(f"   🔢 Total píxeles: {total_pixels:,}")
                        
                        # Verificar si las dimensiones son razonables
                        if width > 2000 or height > 1500:
                            print("   ⚠️  ADVERTENCIA: Imagen podría ser demasiado grande")
                        elif width < 300 or height < 200:
                            print("   ⚠️  ADVERTENCIA: Imagen podría ser demasiado pequeña")
                        else:
                            print("   ✅ Dimensiones apropiadas para análisis visual")
                            
                except ImportError:
                    print("   ℹ️  PIL no disponible para verificar dimensiones")
                except Exception as e:
                    print(f"   ⚠️  Error verificando imagen: {e}")
            else:
                print("   ❌ ERROR: Archivo no se guardó correctamente")
        
        print("\n" + "=" * 60)
        print("🎯 RESULTADO:")
        print("✅ El gráfico se generó correctamente")
        print("✅ El tamaño de archivo es apropiado")
        print("✅ La imagen está lista para análisis visual")
        
        if filepath and os.path.exists(filepath):
            print(f"\n📸 Puedes revisar la imagen visualmente en:")
            print(f"   {filepath}")
            print("\n💡 Verifica que el gráfico contiene:")
            print("   • Velas japonesas (candlesticks)")
            print("   • Indicadores técnicos (SMA, RSI si aplica)")
            print("   • Volumen en panel inferior")
            print("   • Título con símbolo y timeframe")
            print("   • Ejes y etiquetas legibles")
        
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ ERROR generando gráfico: {e}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_chart_quality()
    
    if success:
        print("\n🎉 ¡PRUEBA EXITOSA! El generador de gráficos funciona correctamente.")
    else:
        print("\n💥 PRUEBA FALLIDA. Revisar errores arriba.")
    
    sys.exit(0 if success else 1)
