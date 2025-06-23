# config/config_loader.py
from __future__ import annotations

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, SecretStr, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

root_dotenv = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=root_dotenv)

# --- Sub-modelos para la configuración ---
class TradingConfig(BaseModel):
    symbol: str = "SOLUSDT"
    timeframe: str = "5m"
    use_testnet: bool = True
    min_candles_for_bot_start: int = 51
    trade_cooldown_after_close_seconds: int = 60
    sentiment_refresh_cooldown_seconds: int = 600
    order_status_max_retries: int = 7
    order_status_initial_delay: float = 0.5

class BinanceConfig(BaseSettings):
    api_key:    SecretStr = Field(alias="BINANCE_API_KEY")
    api_secret: SecretStr = Field(alias="BINANCE_API_SECRET")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignorar campos extra del .env
    )


class RiskManagementConfig(BaseModel):
    base_risk_per_trade: float = 0.02
    max_risk_per_trade: float = 0.04
    min_risk_per_trade: float = 0.005
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    min_reward_risk_ratio: float = 1.5
    target_reward_risk_ratio: float = 2.0
    max_daily_loss_pct: float = 0.05
    max_consecutive_losses: int = 3
    max_trades_per_day: int = 15

class LLMConfig(BaseModel):
    default_timeout: int = 90
    default_temperature: float = 0.15
    default_max_tokens: int = 1500

class NewsScraperConfig(BaseModel):
    cryptopanic_api_tokens: List[str] = Field(default_factory=list)

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

# --- Modelo Principal de Configuración ---
class AppConfig(BaseModel):
    trading: TradingConfig = Field(default_factory=TradingConfig)
    binance: BinanceConfig = Field(default_factory=lambda: BinanceConfig())
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    technical_tools: TechnicalToolsConfig = Field(default_factory=TechnicalToolsConfig)

    @validator('binance', pre=True, always=True)
    def load_binance_secrets_from_env(cls, v, values):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file")
        
        # Si se proporcionan en el YAML, se usan esos. Si no, se usan los de ENV.
        if v and isinstance(v, dict):
            return BinanceConfig(
                api_key=SecretStr(v.get('api_key', api_key)),
                api_secret=SecretStr(v.get('api_secret', api_secret))
            )
        else:
            return BinanceConfig(
                api_key=SecretStr(api_key),
                api_secret=SecretStr(api_secret)
            )

    @validator('tools', pre=True, always=True)
    def load_tool_secrets_from_env(cls, v, values):
        news_config = v.get('news_scraper', {})
        cryptopanic_tokens_env = os.getenv("CRYPTOPANIC_TOKENS")
        
        tokens_list = []
        if news_config.get('cryptopanic_api_tokens'):
            tokens_list = news_config['cryptopanic_api_tokens']
        elif cryptopanic_tokens_env:
            tokens_list = [token.strip() for token in cryptopanic_tokens_env.split(',') if token.strip()]
        
        return ToolsConfig(
            news_scraper=NewsScraperConfig(cryptopanic_api_tokens=tokens_list),
            chart_generator=v.get('chart_generator', ChartGeneratorConfig())
        )


CONFIG_FILE_PATH = Path(__file__).parent.parent / "config.yaml" # Asume que config.yaml está en el directorio raíz del proyecto

def load_app_config(config_path: Path = CONFIG_FILE_PATH) -> AppConfig:
    if not config_path.exists():
        print(f"WARNING: Configuration file not found at {config_path}. Using default Pydantic values.")
        # Log this warning as well if logger is configured at this point
        # logger.warning(f"Configuration file not found at {config_path}. Using default Pydantic values.")
        return AppConfig() # Retorna configuración con valores por defecto de Pydantic
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if config_data is None:
        print(f"WARNING: Configuration file {config_path} is empty. Using default Pydantic values.")
        return AppConfig()

    return AppConfig(**config_data)

# Cargar la configuración una vez para ser importada por otros módulos
try:
    APP_CONFIG = load_app_config()
except FileNotFoundError:
    print(f"CRITICAL: Main config file {CONFIG_FILE_PATH} not found. Bot might not run correctly. Using defaults.")
    APP_CONFIG = AppConfig() # Fallback a defaults si el archivo no existe
except Exception as e:
    print(f"CRITICAL: Error loading configuration: {e}. Bot might not run correctly. Using defaults.")
    APP_CONFIG = AppConfig()

if __name__ == "__main__":
    # Para probar la carga de configuración
    print("AppConfig loaded:")
    print(f"Trading Symbol: {APP_CONFIG.trading.symbol}")
    print(f"Use Testnet: {APP_CONFIG.trading.use_testnet}")
    if APP_CONFIG.binance.api_key:
        print(f"Binance API Key Loaded: {'Yes' if APP_CONFIG.binance.api_key.get_secret_value() else 'No'}")
    else:
        print("Binance API Key: Not configured")
    print(f"Risk per trade: {APP_CONFIG.risk_management.base_risk_per_trade}")
    print(f"CryptoPanic Tokens: {APP_CONFIG.tools.news_scraper.cryptopanic_api_tokens}")
    print(f"Logging Level: {APP_CONFIG.logging.level}")
    print(f"Technical Tools Maxlen Buffer: {APP_CONFIG.technical_tools.maxlen_buffer}")


