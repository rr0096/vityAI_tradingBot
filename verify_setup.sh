#!/bin/bash
# Script de verificación del sistema después de configurar .env

echo "🔧 FenixTradingBot - Verificación del Sistema"
echo "============================================"

# 1. Verificar que el archivo .env existe
if [ ! -f ".env" ]; then
    echo "❌ Archivo .env no encontrado"
    echo "   Por favor crea el archivo .env con tus API keys"
    exit 1
fi

echo "✅ Archivo .env encontrado"

# 2. Verificar dependencias de Python
echo "🐍 Verificando dependencias de Python..."
python3 -c "
try:
    import asyncio, aiohttp, websockets
    from binance import Client
    print('✅ Dependencias básicas instaladas')
except ImportError as e:
    print(f'❌ Falta dependencia: {e}')
    exit(1)
"

# 3. Verificar que Ollama está funcionando
echo "🤖 Verificando Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama CLI encontrado"
    
    # Verificar que hay modelos disponibles
    models=$(ollama list 2>/dev/null | grep -v "NAME" | wc -l)
    if [ $models -gt 0 ]; then
        echo "✅ Modelos Ollama disponibles: $models"
    else
        echo "⚠️  No hay modelos Ollama descargados"
        echo "   Recomendación: ollama pull qwen2.5:7b-instruct"
    fi
else
    echo "❌ Ollama no encontrado"
    echo "   Por favor instala Ollama desde https://ollama.ai"
fi

# 4. Test de configuración (sin exponer API keys)
echo "⚙️  Verificando configuración..."
python3 -c "
try:
    from config.config_loader import APP_CONFIG
    print('✅ Configuración cargada correctamente')
    print(f'   Símbolo: {APP_CONFIG.trading.symbol}')
    print(f'   Timeframe: {APP_CONFIG.trading.timeframe}')
    print(f'   Testnet: {APP_CONFIG.trading.use_testnet}')
except Exception as e:
    print(f'❌ Error en configuración: {e}')
    exit(1)
"

# 5. Test de conexión Binance (sin revelar keys)
echo "🔗 Verificando conexión Binance..."
python3 -c "
try:
    import os
    from binance import Client
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print('❌ API keys no configuradas en .env')
        exit(1)
    
    print('✅ API keys encontradas')
    
    # Test de conexión básica
    client = Client(api_key, api_secret, testnet=True)
    client.futures_ping()
    print('✅ Conexión Binance Testnet exitosa')
    
except Exception as e:
    print(f'❌ Error de conexión Binance: {e}')
    print('   Verifica tus API keys en .env')
    exit(1)
"

echo ""
echo "🎉 VERIFICACIÓN COMPLETA"
echo "========================"
echo "El sistema está listo para paper trading!"
echo ""
echo "Comandos disponibles:"
echo "  python run_paper_trading.py --duration 30"
echo "  python live_trading.py (con testnet)"
echo ""
