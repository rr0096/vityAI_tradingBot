#!/usr/bin/env python3
"""
Script simple para probar la generación de gráficos después de las correcciones.
"""

import os
import sys
import base64
from collections import deque
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_chart_generation():
    print("🔍 Probando la generación de gráficos con las correcciones...")
    
    try:
        from tools.chart_generator import generate_chart_for_visual_agent
        
        # Generar datos de prueba
        np.random.seed(42)
        num_points = 50  # Número razonable de puntos
        
        base_price = 100.0
        prices = []
        for i in range(num_points):
            change = np.random.normal(0, 0.02)  # 2% volatilidad
            new_price = (prices[-1] if prices else base_price) * (1 + change)
            prices.append(max(new_price, 1.0))
        
        high_prices = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        low_prices = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        volumes = [float(np.random.randint(1000, 50000)) for _ in prices]
        
        print(f"📊 Generando gráfico con {num_points} puntos de datos...")
        
        base64_img, filepath = generate_chart_for_visual_agent(
            symbol="BTCUSDT",
            timeframe="1h",
            close_buf=deque(prices),
            high_buf=deque(high_prices),
            low_buf=deque(low_prices),
            vol_buf=deque(volumes),
            tech_metrics={'rsi': 65.5, 'last_price': prices[-1]},
            lookback_periods=num_points,
            save_chart=True
        )
        
        # Verificar el resultado
        if not base64_img:
            print("❌ ERROR: No se generó imagen base64")
            return False
        
        print(f"✅ Imagen generada! Longitud base64: {len(base64_img):,} caracteres")
        
        # Verificar si es la imagen de error fallback
        fallback = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        if base64_img == fallback:
            print("⚠️  ADVERTENCIA: Se obtuvo la imagen de error fallback")
            return False
        
        # Verificar tamaño del archivo
        try:
            image_data = base64.b64decode(base64_img)
            size_kb = len(image_data) / 1024
            print(f"📁 Tamaño del archivo: {size_kb:.1f} KB")
            
            # Verificar que no sea demasiado grande
            if size_kb > 1024:  # Más de 1MB
                print(f"⚠️  ADVERTENCIA: Archivo muy grande ({size_kb:.1f} KB)")
                return False
            
            # Verificar dimensiones usando PIL si está disponible
            try:
                from PIL import Image
                import io
                
                with Image.open(io.BytesIO(image_data)) as img:
                    width, height = img.size
                    total_pixels = width * height
                    
                    print(f"🖼️  Dimensiones: {width} x {height} píxeles ({total_pixels:,} píxeles totales)")
                    
                    # Verificar que las dimensiones sean razonables
                    if width > 1500 or height > 1000:
                        print(f"❌ ERROR: Imagen demasiado grande ({width}x{height})")
                        return False
                    elif total_pixels > 1000000:  # Más de 1 millón de píxeles
                        print(f"❌ ERROR: Demasiados píxeles totales ({total_pixels:,})")
                        return False
                    else:
                        print("✅ Dimensiones de imagen son razonables")
                        
            except ImportError:
                print("📝 PIL no disponible para verificar dimensiones")
                
        except Exception as e:
            print(f"❌ ERROR decodificando imagen: {e}")
            return False
        
        if filepath:
            print(f"💾 Guardado en: {filepath}")
        
        print("🎉 ¡Generación de gráfico exitosa con tamaños controlados!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DE CORRECCIONES DEL GENERADOR DE GRÁFICOS")
    print("=" * 60)
    
    success = test_chart_generation()
    
    print("=" * 60)
    if success:
        print("🎯 ÉXITO: Las correcciones funcionan correctamente!")
        print("✅ Los gráficos ahora se generan con tamaños controlados")
    else:
        print("💥 FALLO: Las correcciones necesitan más ajustes")
    print("=" * 60)
