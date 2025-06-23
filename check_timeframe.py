#!/usr/bin/env python3

"""
Script para verificar la configuración del timeframe
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config_loader import APP_CONFIG

def check_timeframe_config():
    """Verificar la configuración del timeframe"""
    
    print("🔍 VERIFICANDO CONFIGURACIÓN DE TIMEFRAME")
    print("=" * 50)
    
    print(f"📊 Symbol: {APP_CONFIG.trading.symbol}")
    print(f"⏰ Timeframe: {APP_CONFIG.trading.timeframe}")
    
    # Construir la URL del WebSocket como lo hace live_trading.py
    SYMBOL = APP_CONFIG.trading.symbol
    TIMEFRAME = APP_CONFIG.trading.timeframe
    WS_URL = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@kline_{TIMEFRAME}"
    
    print(f"🌐 WebSocket URL: {WS_URL}")
    
    if TIMEFRAME == "5m":
        print("✅ CORRECTO: Timeframe configurado a 5 minutos")
    elif TIMEFRAME == "1m":
        print("❌ PROBLEMA: Timeframe sigue en 1 minuto")
        print("🔧 Verificar que el archivo config_loader.py tenga timeframe: str = '5m'")
    else:
        print(f"⚠️  TIMEFRAME INESPERADO: {TIMEFRAME}")
    
    print("\n🚀 URLs completas:")
    print(f"   - WebSocket: {WS_URL}")
    print(f"   - Para {SYMBOL} en {TIMEFRAME}")

if __name__ == "__main__":
    check_timeframe_config()
