# Sistema de Monitoreo - FenixTradingBot

## Descripción General

El sistema de monitoreo de FenixTradingBot proporciona una solución completa para supervisar el rendimiento del bot, detectar problemas y generar alertas en tiempo real.

## Componentes Principales

### 1. MetricsCollector (`monitoring/metrics_collector.py`)

Recopila y almacena métricas del sistema:

- **Métricas de Trading**: PnL, win rate, trades completados, Sharpe ratio
- **Métricas del Sistema**: CPU, memoria, latencia de red, conexiones API
- **Métricas de Agentes**: Tiempo de ejecución, tasa de éxito, señales generadas

#### Uso Básico:

```python
from monitoring.metrics_collector import metrics_collector, TradeMetrics

# Registrar un trade
trade = TradeMetrics(
    trade_id="BTC_12345",
    symbol="BTCUSDT",
    side="BUY",
    entry_price=50000.0,
    exit_price=51000.0,
    quantity=0.01,
    pnl_usd=10.0,
    # ... otros campos
)
metrics_collector.record_trade(trade)

# Obtener resumen de rendimiento
summary = metrics_collector.get_performance_summary()
```

### 2. Sistema de Alertas (`monitoring/alerts.py`)

Detecta condiciones anómalas y envía notificaciones:

#### Canales de Alerta Disponibles:

- **EmailAlertChannel**: Envío por correo electrónico
- **DiscordWebhookChannel**: Notificaciones a Discord
- **FileLogChannel**: Registro en archivos JSON

#### Configuración de Alertas:

```python
from monitoring.alerts import alert_manager, EmailAlertChannel, AlertSeverity

# Configurar canal de email
email_channel = EmailAlertChannel(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="tu_email@gmail.com",
    password="tu_password",
    to_emails=["destino@gmail.com"]
)
alert_manager.add_channel(email_channel)

# Las alertas se generan automáticamente basadas en reglas predefinidas
```

#### Reglas de Alerta Predefinidas:

- **Salud del Sistema Crítica**: Cuando la salud general < 30%
- **Alta Tasa de Errores API**: > 20% de errores en llamadas API
- **Pérdida Diaria Significativa**: Pérdidas > $500 USD
- **Drawdown Elevado**: Drawdown > 15%

### 3. Dashboard Web (`monitoring/dashboard.py`)

Interfaz web para visualización en tiempo real:

```python
from monitoring.dashboard import dashboard

# Iniciar dashboard
dashboard.start()  # Disponible en http://127.0.0.1:5000
```

#### Características del Dashboard:

- **Métricas de Trading**: Balance, PnL, win rate, Sharpe ratio
- **Salud del Sistema**: CPU, memoria, latencia, estado de agentes
- **Alertas Activas**: Lista de alertas pendientes con opción de resolución
- **Actualización Automática**: Datos actualizados cada 30 segundos

## Integración con el Bot Principal

### Configuración Automática

El sistema se integra automáticamente en `live_trading.py`:

```python
# El sistema se inicializa automáticamente si está disponible
if MONITORING_AVAILABLE:
    # Configurar canales de alerta
    alert_manager.add_channel(FileLogChannel("logs/alerts.jsonl"))
    
    # Iniciar monitoreo del sistema
    metrics_collector.start_system_monitoring(interval_seconds=30.0)
    
    # Iniciar dashboard (opcional)
    dashboard.start()
```

### Registro Automático de Métricas

El bot registra automáticamente:

- **Trades Completados**: Al cerrar posiciones
- **Llamadas API**: A Binance y otros servicios
- **Ejecución de Agentes**: Tiempo de respuesta y éxito
- **Balance del Portfolio**: Actualizado en tiempo real

## Configuración

### Archivo de Configuración (`config/monitoring_config.py`)

```python
from config.monitoring_config import DEFAULT_MONITORING_CONFIG

config = DEFAULT_MONITORING_CONFIG
config.dashboard.enabled = True  # Habilitar dashboard
config.dashboard.port = 8080     # Cambiar puerto
```

### Variables de Entorno

Configurar canales de alerta mediante variables de entorno:

```bash
# Email
ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_USERNAME=tu_email@gmail.com
ALERT_EMAIL_PASSWORD=tu_password
ALERT_EMAIL_TO=destino@gmail.com

# Discord
ALERT_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### Umbrales de Alerta Personalizados

```python
from monitoring.alerts import alert_manager

def custom_rule(metrics):
    if metrics.get('daily_pnl', 0) < -1000:  # Pérdida > $1000
        return Alert(
            severity=AlertSeverity.CRITICAL,
            title="Pérdida Diaria Crítica",
            message=f"Pérdida diaria: ${metrics['daily_pnl']:.2f}",
            # ... otros campos
        )
    return None

alert_manager.add_rule(custom_rule)
```

## Archivos de Salida

### Estructura de Directorios

```
logs/
├── metrics/
│   ├── trades.jsonl           # Historial de trades
│   └── metrics_export_*.json  # Exportaciones completas
├── alerts.jsonl               # Log de alertas
└── fenix_bot_live.log        # Log principal del bot
```

### Formato de Datos

#### Métricas de Trade (`trades.jsonl`):

```json
{
    "trade_id": "BTCUSDT_1640995200",
    "symbol": "BTCUSDT",
    "side": "BUY",
    "entry_price": 50000.0,
    "exit_price": 51000.0,
    "quantity": 0.01,
    "pnl_usd": 10.0,
    "entry_time": "2021-12-31T23:00:00Z",
    "exit_time": "2021-12-31T23:05:00Z",
    "duration_seconds": 300.0,
    "agent_signals": {
        "sentiment": "POSITIVE",
        "technical": "BUY",
        "visual": "CONSIDER_BUY"
    },
    "risk_score": 0.7
}
```

#### Alertas (`alerts.jsonl`):

```json
{
    "id": "system_health_critical_1640995200",
    "severity": "critical",
    "title": "Salud del Sistema Crítica",
    "message": "La salud del sistema ha caído a 25%",
    "timestamp": "2021-12-31T23:00:00Z",
    "category": "system_health",
    "resolved": false
}
```

## API del Dashboard

### Endpoints Disponibles:

- `GET /`: Dashboard principal
- `GET /api/metrics`: Métricas actuales en JSON
- `GET /api/alerts`: Alertas activas
- `POST /api/alerts/{id}/resolve`: Resolver una alerta
- `GET /api/status`: Estado del bot

### Ejemplo de Respuesta de Métricas:

```json
{
    "performance_counters": {
        "total_trades": 45,
        "winning_trades": 28,
        "losing_trades": 17,
        "api_calls": 1250,
        "api_errors": 3
    },
    "real_time_metrics": {
        "current_balance": 10500.50,
        "daily_pnl": 125.75,
        "total_pnl": 2345.80,
        "win_rate": 0.622,
        "sharpe_ratio": 1.45,
        "max_drawdown": 0.08
    },
    "system_health": 0.85,
    "agent_health": {
        "sentiment": 0.92,
        "technical": 0.88,
        "visual": 0.95,
        "qabba": 0.90,
        "decision": 0.87
    }
}
```

## Mejores Prácticas

### 1. Configuración de Alertas

- Configurar múltiples canales para redundancia
- Ajustar umbrales según el capital y tolerancia al riesgo
- Revisar y resolver alertas regularmente

### 2. Análisis de Métricas

- Exportar datos regularmente para análisis histórico
- Monitorear tendencias en lugar de valores absolutos
- Correlacionar rendimiento con condiciones del mercado

### 3. Mantenimiento

- Limpiar logs antiguos periódicamente
- Actualizar umbrales basado en experiencia
- Revisar y optimizar reglas de alerta

## Troubleshooting

### Problemas Comunes

#### Dashboard no se inicia:

```bash
# Verificar que Flask esté instalado
pip install flask

# Verificar puerto disponible
netstat -an | grep :5000
```

#### Alertas no se envían:

```python
# Verificar configuración de canales
for channel in alert_manager.channels:
    print(f"Canal {channel.name}: {'✓' if channel.is_available() else '✗'}")
```

#### Métricas no se registran:

```python
# Verificar que el monitoreo esté habilitado
if MONITORING_AVAILABLE:
    print("✓ Sistema de monitoreo disponible")
else:
    print("✗ Sistema de monitoreo no disponible")
```

## Desarrollo y Extensión

### Agregar Nuevas Métricas

```python
from monitoring.metrics_collector import metrics_collector

# Agregar métrica personalizada
def record_custom_metric(value):
    # Implementar lógica personalizada
    pass
```

### Crear Nuevos Canales de Alerta

```python
from monitoring.alerts import AlertChannel

class CustomAlertChannel(AlertChannel):
    def send_alert(self, alert):
        # Implementar envío personalizado
        return True
```

### Agregar Nuevas Reglas de Alerta

```python
def custom_alert_rule(metrics):
    # Implementar lógica de detección
    if condition_met:
        return Alert(...)
    return None

alert_manager.add_rule(custom_alert_rule)
```
