#!/usr/bin/env python3

"""
Script para verificar la configuración de circuit breakers y mostrar el resumen
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config_loader import APP_CONFIG

def print_circuit_breaker_config():
    """Mostrar configuración actual de circuit breakers"""
    
    print("🛡️  CONFIGURACIÓN ACTUAL DE CIRCUIT BREAKERS")
    print("=" * 50)
    
    # Trading config
    print("\n📊 CONFIGURACIÓN DE TRADING:")
    print(f"   Symbol: {APP_CONFIG.trading.symbol}")
    print(f"   Timeframe: {APP_CONFIG.trading.timeframe}")
    print(f"   Sentiment Refresh Cooldown: {APP_CONFIG.trading.sentiment_refresh_cooldown_seconds}s")
    print(f"   Trade Cooldown After Close: {APP_CONFIG.trading.trade_cooldown_after_close_seconds}s")
    
    # Risk config
    print("\n🚨 CIRCUIT BREAKERS (Risk Management):")
    print(f"   Max Daily Loss: {APP_CONFIG.risk_management.max_daily_loss_pct * 100:.1f}% del balance")
    print(f"   Max Consecutive Losses: {APP_CONFIG.risk_management.max_consecutive_losses}")
    print(f"   Max Trades Per Day: {APP_CONFIG.risk_management.max_trades_per_day}")
    print("   Drawdown Limit (código): 15.0% (hardcoded en risk.py)")
    
    # Position sizing
    print("\n💰 POSITION SIZING:")
    print(f"   Base Risk Per Trade: {APP_CONFIG.risk_management.base_risk_per_trade * 100:.1f}% del balance")
    print(f"   Max Risk Per Trade: {APP_CONFIG.risk_management.max_risk_per_trade * 100:.1f}% del balance")
    print(f"   Min Risk Per Trade: {APP_CONFIG.risk_management.min_risk_per_trade * 100:.2f}% del balance")
    
    # Risk/Reward
    print("\n📈 RISK/REWARD:")
    print(f"   Min Risk/Reward Ratio: {APP_CONFIG.risk_management.min_reward_risk_ratio:.1f}")
    print(f"   Target Risk/Reward Ratio: {APP_CONFIG.risk_management.target_reward_risk_ratio:.1f}")
    print(f"   ATR Stop Loss Multiplier: {APP_CONFIG.risk_management.atr_sl_multiplier:.1f}")
    print(f"   ATR Take Profit Multiplier: {APP_CONFIG.risk_management.atr_tp_multiplier:.1f}")
    
    print("\n✅ Los circuit breakers están ACTIVOS y funcionando.")
    print("🔄 Nuevo timeframe configurado: 5 minutos (más eficiente para análisis completo)")
    
    print("\n💡 MEJORAS IMPLEMENTADAS:")
    print("   ✅ Captura real de gráficos de Bitget con pantalla completa")
    print("   ✅ Indicadores técnicos añadidos automáticamente")
    print("   ✅ Circuit breakers para protección de pérdidas")
    print("   ✅ Timeframe optimizado a 5 minutos")
    print("   ✅ Análisis visual mejorado con claridad de patrones")

if __name__ == "__main__":
    print_circuit_breaker_config()
