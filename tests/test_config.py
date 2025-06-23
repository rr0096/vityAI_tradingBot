#!/usr/bin/env python3
"""
Test de importación de configuración
"""

from config.config_loader import APP_CONFIG

print('✅ Config import: OK')
print(f'✅ API Key: {len(APP_CONFIG.binance.api_key)} chars')
print(f'✅ API Secret: {len(APP_CONFIG.binance.api_secret)} chars')

if len(APP_CONFIG.binance.api_key) > 10:
    print('🎉 Configuration working correctly!')
else:
    print('❌ Configuration issue: API keys too short')

print(f'Symbol: {APP_CONFIG.trading.symbol}')
print(f'Testnet: {APP_CONFIG.trading.use_testnet}')
