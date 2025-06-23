# config/config_loader_simple.py
"""
Configuración simplificada para evitar problemas con Pydantic
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from pathlib import Path
import yaml
from pydantic import BaseModel, SecretStr, Field
from pydantic_settings import BaseSettings

# Función para cargar .env de manera robusta
def load_env_variables():
    """Carga variables de entorno de manera robusta"""
    # Intentar cargar desde múltiples ubicaciones
    possible_paths = [
        Path(__file__).parent.parent / ".env",  # Desde config/
        Path.cwd() / ".env",                    # Desde working directory
        Path(".env")                            # Relativo actual
    ]
    
    for env_path in possible_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return True
    
    return False

# Cargar .env inmediatamente
load_env_variables()

class TradingConfig(BaseModel):
    symbol: str = "SOLUSDT"
    timeframe: str = "5m"
    use_testnet: bool = True
    min_candles_for_bot_start: int = 51
    trade_cooldown_after_close_seconds: int = 60
    sentiment_refresh_cooldown_seconds: int = 600
    order_status_max_retries: int = 7
    order_status_initial_delay: float = 0.5

class BinanceConfig(BaseModel):
    api_key: str
    api_secret: str

class RiskManagementConfig(BaseModel):
    base_risk_per_trade: float = 0.02
    max_risk_per_trade: float = 0.04
    min_risk_per_trade: float = 0.005
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    min_reward_risk_ratio: float = 1.5
    target_reward_risk_ratio: float = 2.0
    max_daily_loss_pct: float = 0.05
    max_consecutive_losses: int = 6
    max_trades_per_day: int = 60

class LLMConfig(BaseModel):
    default_timeout: int = 90
    default_temperature: float = 0.15
    default_max_tokens: int = 1500

class NewsScraperConfig(BaseModel):
    cryptopanic_api_tokens: list[str] = []

class ChartGeneratorConfig(BaseModel):
    save_charts_to_disk: bool = True
    charts_dir: str = "logs/charts"

class ToolsConfig(BaseModel):
    news_scraper: NewsScraperConfig = Field(default_factory=NewsScraperConfig)
    chart_generator: ChartGeneratorConfig = Field(default_factory=ChartGeneratorConfig)

class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_file: str = "logs/fenix_live_trading.log"

class TechnicalToolsConfig(BaseModel):
    maxlen_buffer: int = 100
    min_candles_for_reliable_calc: int = 30

class AppConfig(BaseModel):
    trading: TradingConfig = Field(default_factory=TradingConfig)
    binance: BinanceConfig
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    technical_tools: TechnicalToolsConfig = Field(default_factory=TechnicalToolsConfig)

def create_app_config() -> AppConfig:
    """Crea la configuración de la aplicación"""
    try:
        # Cargar configuración YAML si existe
        config_path = Path(__file__).parent / "config.yaml"
        yaml_config = {}
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
        
        # Obtener API keys del entorno
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file"
            )
        
        # Crear configuración completa usando dict básico
        config_data = {
            "trading": yaml_config.get("trading", {}),
            "binance": {
                "api_key": api_key,
                "api_secret": api_secret
            },
            "risk_management": yaml_config.get("risk_management", {}),
            "llm": yaml_config.get("llm", {}),
            "tools": yaml_config.get("tools", {}),
            "logging": yaml_config.get("logging", {}),
            "technical_tools": yaml_config.get("technical_tools", {})
        }
        
        app_config = AppConfig(**config_data)
        
        return app_config
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

# Crear la configuración global
APP_CONFIG = create_app_config()

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Symbol: {APP_CONFIG.trading.symbol}")
    print(f"Timeframe: {APP_CONFIG.trading.timeframe}")
    print(f"Use Testnet: {APP_CONFIG.trading.use_testnet}")
    print(f"API Key: {'*' * 20}...")
