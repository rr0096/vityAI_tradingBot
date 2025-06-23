# FenixTradingBot - Paper Trading System

## 🎭 Sistema Completo de Paper Trading

El sistema de paper trading de FenixTradingBot replica completamente el comportamiento del live trading pero usando simuladores realistas en lugar de órdenes reales con Binance.

## ✅ Características Implementadas

### 🔧 **Componentes Core**
- **Simulador de Órdenes Realista** (`paper_trading/order_simulator.py`)
  - Simulación de latencia (50ms promedio)
  - Slippage realista (0.02% promedio, max 0.1%)
  - Comisiones (0.04% por operación)
  - Órdenes TP/SL automáticas
  - 98% de probabilidad de fill para market orders

- **Simulador de Market Data** (`paper_trading/market_simulator.py`)
  - Datos históricos reales de Binance
  - Streaming simulado con volatilidad realista
  - Correlaciones entre símbolos
  - Distribución normal de cambios de precio

- **Sistema Integrado** (`paper_trading_system.py`)
  - Integración completa con todos los agentes LLM
  - Risk management idéntico al live trading
  - Monitoreo de posiciones activas
  - Tracking de P&L en tiempo real

### 🤖 **Arquitectura de Agentes**
- **Sentiment Agent**: Análisis de sentimiento con LLMs locales
- **Technical Agent**: Análisis técnico con indicadores
- **Visual Agent**: Análisis de gráficos 
- **QABBA Agent**: Validación con modelo ML
- **Decision Agent**: Decisión final consensuada
- **Risk Manager**: Gestión de riesgo idéntica al live trading

### 📊 **Monitoreo y Reporting**
- Logs detallados de todas las operaciones
- Reportes de sesión en JSON
- Tracking de portfolio en tiempo real
- Métricas de rendimiento

## 🚀 Cómo Usar el Paper Trading

### **1. Ejecución Rápida**
```bash
# Ejecutar por 30 minutos con $10,000
python run_paper_trading.py

# Ejecutar por 1 hora con $50,000
python run_paper_trading.py --balance 50000 --duration 60

# Ejecutar indefinidamente (Ctrl+C para parar)
python run_paper_trading.py --duration 0
```

### **2. Configuración Personalizada**
```bash
# Símbolos específicos
python run_paper_trading.py --symbols "BTCUSDT,ETHUSDT,SOLUSDT" --duration 120

# Modo verbose para debug
python run_paper_trading.py --verbose --duration 15
```

### **3. Usando el Sistema Programáticamente**
```python
from paper_trading_system import PaperTradingSystem
import asyncio

async def my_paper_trading():
    system = PaperTradingSystem(initial_balance=10000.0)
    
    if await system.initialize():
        # Ejecutar por 30 minutos
        trading_task = asyncio.create_task(system.start_trading())
        await asyncio.sleep(30 * 60)
        
        system.stop_trading()
        summary = system.get_portfolio_summary()
        print(f"Final return: {summary['return_percentage']:.2f}%")

asyncio.run(my_paper_trading())
```

## 📋 Lo Que Ya Tienes vs. Lo Que Falta

### ✅ **YA IMPLEMENTADO**
1. ✅ **Binance Testnet** - Configurado y funcional
2. ✅ **Sistema de Agentes Completo** - Todos los agentes funcionando
3. ✅ **Risk Management** - AdvancedRiskManager integrado
4. ✅ **Monitoreo** - Sistema completo de métricas
5. ✅ **Paper Trading Demo** - Demo básico existente
6. ✅ **Simulador de Órdenes** - Simulador realista nuevo
7. ✅ **Market Data Simulator** - Datos realistas
8. ✅ **Sistema Integrado** - Todo conectado

### ❌ **LO QUE FALTA (Opcional)**

#### **4. Interfaz Web de Paper Trading**
```bash
# Crear dashboard web para paper trading
mkdir paper_trading/web_interface
```

#### **5. Backtesting Avanzado**
- Backtesting con datos históricos largos
- Optimización de parámetros
- Análisis de drawdown

#### **6. Comparación Live vs Paper**
- Métricas comparativas
- Análisis de diferencias
- Validación de estrategias

## 🔧 Configuración

### **Archivos de Configuración**
- `config/config.yaml` - Configuración principal
- `paper_trading/` - Módulos de simulación
- `logs/paper_trading_*` - Logs y reportes

### **Variables de Entorno Necesarias**
```bash
# Solo necesarias para datos reales (opcional)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

### **Parámetros de Simulación**
```python
# En order_simulator.py
commission_rate = 0.0004  # 0.04% comisión
slippage_rate = 0.0002   # 0.02% slippage promedio  
fill_probability = 0.98   # 98% probabilidad de fill

# En market_simulator.py
volatility_factor = 1.0   # Factor de volatilidad
```

## 📊 Reportes y Logs

### **Archivos Generados**
```
logs/
├── paper_trading.log              # Log principal
├── paper_trading_trades.jsonl     # Log de trades
├── paper_trading_portfolio.json   # Estado del portfolio
└── paper_trading_reports/         # Reportes de sesión
    └── session_20250620_143000.json
```

### **Formato de Reporte**
```json
{
  "session_info": {
    "timestamp": "2025-06-20T14:30:00Z",
    "initial_balance": 10000.0,
    "duration_minutes": 30,
    "symbols": "SOLUSDT"
  },
  "results": {
    "total_return": 150.25,
    "return_percentage": 1.50,
    "active_positions": [...],
    "total_portfolio_value": 10150.25
  }
}
```

## 🎯 Para Producción

### **1. Validación con Paper Trading**
```bash
# Ejecutar sesiones largas para validar
python run_paper_trading.py --duration 480  # 8 horas
```

### **2. Switch a Live Trading**
```yaml
# En config/config.yaml
trading:
  use_testnet: false  # Cambiar a false para producción
```

### **3. Monitoreo Continuo**
- Dashboard de métricas en tiempo real
- Alertas automáticas
- Circuit breakers

## 🔍 Debugging y Testing

### **Logs Detallados**
```bash
# Ejecutar con verbose logging
python run_paper_trading.py --verbose --duration 10
```

### **Testing de Componentes**
```python
# Test del simulador de órdenes
python -c "
from paper_trading.order_simulator import order_simulator
import asyncio

async def test():
    result = await order_simulator.place_order('BTCUSDT', 'BUY', 'MARKET', 0.001)
    print(result)

asyncio.run(test())
"
```

## 📈 Próximos Pasos

1. **Ejecutar sesiones de validación** con el nuevo sistema
2. **Comparar resultados** con el demo básico existente  
3. **Ajustar parámetros** de simulación si es necesario
4. **Crear interfaz web** (opcional)
5. **Implementar backtesting** avanzado (opcional)

## ❗ Notas Importantes

- El sistema usa **LLMs reales de Ollama**, no mocks
- Los **datos de mercado son reales** de Binance
- La **simulación es muy realista** con latencia, slippage y comisiones
- **Compatible con la configuración existente** sin cambios
- **Listo para usar inmediatamente**
