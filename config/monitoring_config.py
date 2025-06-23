# config/monitoring_config.py
"""
Configuración específica para el sistema de monitoreo
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class AlertChannelConfig(BaseModel):
    """Configuración de canal de alertas"""
    enabled: bool = True
    channel_type: str  # 'email', 'discord', 'file'
    config: Dict[str, Any] = Field(default_factory=dict)

class MetricsConfig(BaseModel):
    """Configuración del sistema de métricas"""
    enabled: bool = True
    output_dir: str = "logs/metrics"
    system_monitoring_interval: float = 30.0
    max_trade_history: int = 1000
    max_system_metrics: int = 1000
    max_agent_metrics: int = 500

class DashboardConfig(BaseModel):
    """Configuración del dashboard web"""
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 5000
    auto_refresh_seconds: int = 30

class AlertConfig(BaseModel):
    """Configuración del sistema de alertas"""
    enabled: bool = True
    suppression_rules: Dict[str, Any] = Field(default_factory=lambda: {
        "duplicate_timeout": 300,  # 5 minutos
        "max_alerts_per_category_per_hour": 10,
    })
    channels: List[AlertChannelConfig] = Field(default_factory=list)

class MonitoringConfig(BaseModel):
    """Configuración completa del sistema de monitoreo"""
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    
    # Umbrales para alertas
    thresholds: Dict[str, Any] = Field(default_factory=lambda: {
        "system_health_critical": 0.3,
        "api_error_rate_high": 0.2,
        "daily_loss_limit": -500.0,
        "max_drawdown_warning": 0.15,
        "cpu_usage_high": 80.0,
        "memory_usage_high": 90.0,
        "websocket_latency_high": 1000.0,
    })

# Configuración por defecto
DEFAULT_MONITORING_CONFIG = MonitoringConfig(
    metrics=MetricsConfig(
        enabled=True,
        output_dir="logs/metrics",
        system_monitoring_interval=30.0
    ),
    alerts=AlertConfig(
        enabled=True,
        channels=[
            AlertChannelConfig(
                channel_type="file",
                config={"log_file": "logs/alerts.jsonl"}
            )
        ]
    ),
    dashboard=DashboardConfig(
        enabled=False,  # Disabled by default
        host="127.0.0.1",
        port=5000
    )
)
