#!/usr/bin/env python3

"""
SOLUCIÓN RÁPIDA: Mejorar el análisis visual actual con gráficos más detallados
y configurar un modelo que soporte análisis de imágenes.
"""

# 1. VERIFICAR MODELOS CON SOPORTE DE VISIÓN
MODELS_WITH_VISION = {
    # Modelos OpenAI (los mejores para visión)
    "gpt-4-vision-preview": "Excelente para análisis de gráficos",
    "gpt-4o": "Mejor modelo multimodal de OpenAI",
    "gpt-4o-mini": "Económico con buena visión",
    
    # Modelos locales con visión (Ollama)
    "llava:13b": "Modelo local con capacidades de visión",
    "llava:7b": "Versión más ligera con visión",
    "bakllava": "Especializado en análisis visual",
    "moondream": "Rápido para análisis de imágenes",
}

# 2. VERIFICAR QUÉ MODELOS TIENES DISPONIBLES
def check_available_vision_models():
    """Verifica qué modelos con visión están disponibles en tu sistema."""
    import subprocess
    import json
    
    try:
        # Verificar modelos de Ollama
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("📋 MODELOS DISPONIBLES EN OLLAMA:")
            print(result.stdout)
            
            available_vision = []
            for model_name in MODELS_WITH_VISION.keys():
                if any(model_name.split(':')[0] in line for line in result.stdout.split('\n')):
                    available_vision.append(model_name)
            
            if available_vision:
                print(f"\n✅ MODELOS CON VISIÓN DISPONIBLES: {available_vision}")
                return available_vision
            else:
                print("\n❌ NO SE ENCONTRARON MODELOS CON VISIÓN")
                print("\n🔧 PARA INSTALAR UN MODELO CON VISIÓN:")
                print("ollama pull llava:7b")
                print("# o")
                print("ollama pull moondream")
                
        else:
            print("❌ Error verificando Ollama:", result.stderr)
            
    except FileNotFoundError:
        print("❌ Ollama no está instalado o no está en PATH")
        
    return []

# 3. CONFIGURACIÓN RECOMENDADA
RECOMMENDED_CONFIG = {
    "visual_model": "llava:7b",  # Cambiar esto por un modelo que tengas
    "fallback_model": "moondream",
    "chart_improvements": {
        "size": (1920, 1080),  # Tamaño más grande
        "dpi": 150,  # Mayor resolución
        "indicators": ["RSI", "MACD", "EMA20", "EMA50", "Volume"],
        "timeframes_to_show": 100,  # Más historia
        "colors": "professional"  # Esquema de colores más claro
    }
}

if __name__ == "__main__":
    print("🔍 VERIFICANDO CONFIGURACIÓN PARA ANÁLISIS VISUAL...\n")
    
    print("=" * 60)
    print("DIAGNÓSTICO DEL PROBLEMA ACTUAL")
    print("=" * 60)
    
    print("""
❌ PROBLEMA IDENTIFICADO:
   - El modelo 'qwen2.5:7b-instruct-q5_k_m' NO soporta análisis de imágenes
   - Los gráficos se generan pero el modelo no puede "verlos"
   - Por eso siempre devuelve NEUTRAL con Clarity: N/A

✅ SOLUCIONES DISPONIBLES:
   1. Cambiar a un modelo con soporte de visión (llava, moondream)
   2. Mejorar la calidad de los gráficos generados
   3. Implementar captura de TradingView (más avanzado)
""")

    print("\n" + "=" * 60)
    print("VERIFICANDO MODELOS DISPONIBLES")
    print("=" * 60)
    
    available_models = check_available_vision_models()
    
    print("\n" + "=" * 60)
    print("RECOMENDACIONES INMEDIATAS")
    print("=" * 60)
    
    if available_models:
        print(f"✅ USAR: {available_models[0]}")
        print(f"📝 Cambiar en config: LLM_MODEL_NAME='{available_models[0]}'")
    else:
        print("🚀 INSTALAR MODELO CON VISIÓN:")
        print("   ollama pull llava:7b")
        print("   # Luego cambiar config: LLM_MODEL_NAME='llava:7b'")
    
    print("""
🔧 CONFIGURACIÓN INMEDIATA:

1. En tu archivo de configuración, cambia:
   LLM_MODEL_NAME = "llava:7b"  # O el modelo que tengas disponible

2. Reinicia el bot

3. El análisis visual empezará a funcionar correctamente

💡 NOTA: Los modelos con visión son un poco más lentos pero pueden 
   analizar gráficos realmente.
""")
