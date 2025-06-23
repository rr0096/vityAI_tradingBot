"""
FenixTradingBot - Market Data Simulator for Paper Trading
Simula datos de mercado realistas usando datos históricos de Binance
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from dataclasses import dataclass
import aiohttp
import random

logger = logging.getLogger(__name__)

@dataclass
class Kline:
    """Representa una vela/kline de mercado"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    timeframe: str

class MarketDataSimulator:
    """
    Simulador de datos de mercado que proporciona:
    - Datos históricos reales de Binance
    - Streaming simulado de klines
    - Volatilidad realista
    - Correlaciones entre símbolos
    """
    
    def __init__(self, base_url: str = "https://fapi.binance.com"):
        self.base_url = base_url
        self.historical_data: Dict[str, List[Kline]] = {}
        self.current_prices: Dict[str, float] = {}
        self.price_subscribers: List = []
        self.is_streaming = False
        
        # Configuración del simulador
        self.volatility_factor = 1.0  # Factor de volatilidad
        self.correlation_matrix = {}  # Correlaciones entre símbolos
        
        logger.info("📊 MarketDataSimulator initialized")
    
    async def load_historical_data(self, 
                                 symbol: str, 
                                 timeframe: str = "1m", 
                                 days_back: int = 30) -> bool:
        """Carga datos históricos reales de Binance"""
        try:
            # Calcular timestamp de inicio
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days_back)
            
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1500  # Máximo por request
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        klines = []
                        for kline_data in data:
                            kline = Kline(
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(kline_data[0] / 1000, tz=timezone.utc),
                                open_price=float(kline_data[1]),
                                high_price=float(kline_data[2]),
                                low_price=float(kline_data[3]),
                                close_price=float(kline_data[4]),
                                volume=float(kline_data[5]),
                                timeframe=timeframe
                            )
                            klines.append(kline)
                        
                        self.historical_data[symbol] = klines
                        self.current_prices[symbol] = klines[-1].close_price if klines else 0.0
                        
                        logger.info(f"✅ Loaded {len(klines)} historical klines for {symbol}")
                        return True
                    else:
                        logger.error(f"❌ Failed to load data for {symbol}: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Error loading historical data for {symbol}: {e}")
            return False
    
    async def start_streaming(self, symbols: List[str], timeframe: str = "1m"):
        """Inicia el streaming simulado de datos de mercado"""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        self.is_streaming = True
        logger.info(f"🔄 Starting market data streaming for {symbols}")
        
        # Cargar datos históricos para todos los símbolos
        for symbol in symbols:
            await self.load_historical_data(symbol, timeframe)
        
        # Iniciar simulación de streaming
        asyncio.create_task(self._simulate_streaming(symbols, timeframe))
    
    async def _simulate_streaming(self, symbols: List[str], timeframe: str):
        """Simula el streaming de datos de mercado con volatilidad realista"""
        interval_seconds = self._timeframe_to_seconds(timeframe)
        
        while self.is_streaming:
            try:
                current_time = datetime.now(timezone.utc)
                
                for symbol in symbols:
                    if symbol not in self.historical_data:
                        continue
                    
                    # Obtener última vela histórica
                    last_kline = self.historical_data[symbol][-1]
                    
                    # Generar nueva vela basada en la anterior
                    new_kline = self._generate_next_kline(last_kline, current_time, timeframe)
                    
                    # Actualizar datos
                    self.historical_data[symbol].append(new_kline)
                    self.current_prices[symbol] = new_kline.close_price
                    
                    # Mantener solo las últimas N velas para no consumir demasiada memoria
                    if len(self.historical_data[symbol]) > 1000:
                        self.historical_data[symbol] = self.historical_data[symbol][-500:]
                    
                    # Notificar suscriptores
                    await self._notify_price_update(symbol, new_kline)
                
                # Esperar hasta el siguiente intervalo
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in streaming simulation: {e}")
                await asyncio.sleep(5)
    
    def _generate_next_kline(self, last_kline: Kline, timestamp: datetime, timeframe: str) -> Kline:
        """Genera la siguiente vela con volatilidad y tendencia realistas"""
        
        # Calcular volatilidad basada en datos históricos
        if len(self.historical_data[last_kline.symbol]) >= 20:
            recent_klines = self.historical_data[last_kline.symbol][-20:]
            price_changes = [
                (k.close_price - k.open_price) / k.open_price 
                for k in recent_klines
            ]
            volatility = abs(sum(price_changes) / len(price_changes)) * 10
        else:
            volatility = 0.001  # 0.1% volatilidad base
        
        # Aplicar factor de volatilidad
        volatility *= self.volatility_factor
        
        # Generar cambio de precio con distribución más realista
        price_change_pct = random.gauss(0, volatility)  # Distribución normal
        
        # Limitar cambios extremos
        price_change_pct = max(-0.05, min(0.05, price_change_pct))  # ±5% máximo
        
        # Calcular precios de la nueva vela
        open_price = last_kline.close_price
        close_price = open_price * (1 + price_change_pct)
        
        # Generar high y low realistas
        if price_change_pct >= 0:  # Vela verde
            high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility * 0.5))
        else:  # Vela roja
            high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility * 0.5))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility))
        
        # Asegurar que high >= max(open, close) y low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generar volumen basado en volatilidad
        base_volume = last_kline.volume
        volume_change = random.uniform(-0.3, 0.3) + abs(price_change_pct) * 5
        volume = base_volume * (1 + volume_change)
        volume = max(volume, base_volume * 0.1)  # Mínimo 10% del volumen anterior
        
        return Kline(
            symbol=last_kline.symbol,
            timestamp=timestamp,
            open_price=round(open_price, 6),
            high_price=round(high_price, 6),
            low_price=round(low_price, 6),
            close_price=round(close_price, 6),
            volume=round(volume, 2),
            timeframe=timeframe
        )
    
    async def _notify_price_update(self, symbol: str, kline: Kline):
        """Notifica a los suscriptores sobre actualizaciones de precio"""
        for subscriber in self.price_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(symbol, kline)
                else:
                    subscriber(symbol, kline)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe_to_price_updates(self, callback):
        """Suscribe un callback para recibir actualizaciones de precio"""
        self.price_subscribers.append(callback)
        logger.debug(f"Added price update subscriber: {callback.__name__}")
    
    def unsubscribe_from_price_updates(self, callback):
        """Desuscribe un callback"""
        if callback in self.price_subscribers:
            self.price_subscribers.remove(callback)
            logger.debug(f"Removed price update subscriber: {callback.__name__}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtiene el precio actual de un símbolo"""
        return self.current_prices.get(symbol)
    
    def get_recent_klines(self, symbol: str, count: int = 100) -> List[Kline]:
        """Obtiene las últimas N velas de un símbolo"""
        if symbol in self.historical_data:
            return self.historical_data[symbol][-count:]
        return []
    
    def set_volatility_factor(self, factor: float):
        """Ajusta el factor de volatilidad del mercado"""
        self.volatility_factor = factor
        logger.info(f"Volatility factor set to {factor}")
    
    def stop_streaming(self):
        """Detiene el streaming de datos"""
        self.is_streaming = False
        logger.info("📴 Market data streaming stopped")
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convierte un timeframe a segundos"""
        timeframe_map = {
            "1s": 1,
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400
        }
        return timeframe_map.get(timeframe, 60)  # Default a 1 minuto
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Resumen del estado del mercado simulado"""
        return {
            "active_symbols": list(self.current_prices.keys()),
            "current_prices": self.current_prices.copy(),
            "is_streaming": self.is_streaming,
            "volatility_factor": self.volatility_factor,
            "data_points": {
                symbol: len(klines) 
                for symbol, klines in self.historical_data.items()
            }
        }

# Global instance
market_simulator = MarketDataSimulator()
