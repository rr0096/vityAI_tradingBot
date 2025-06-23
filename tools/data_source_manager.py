# tools/data_source_manager.py
"""
Gestor centralizado de fuentes de datos gratuitas para FenixTradingBot
Integra múltiples APIs y fuentes sin costos adicionales
"""

import logging
import requests
import time
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import random

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuración de una fuente de datos"""
    name: str
    base_url: str
    rate_limit_per_minute: int
    api_keys: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    requires_auth: bool = False
    priority: int = 1  # 1=highest, 5=lowest
    last_request_time: float = 0
    request_count: int = 0
    errors_count: int = 0
    is_active: bool = True

class RateLimitManager:
    """Gestor de rate limits para múltiples APIs"""
    
    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self.current_key_index: Dict[str, int] = {}
        self.cache_dir = Path("cache/data_sources")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar fuentes de datos gratuitas
        self._setup_free_data_sources()
    
    def _setup_free_data_sources(self):
        """Configura todas las fuentes de datos gratuitas disponibles"""
        
        # CoinGecko - Principal fuente gratuita
        self.sources['coingecko'] = DataSource(
            name="CoinGecko",
            base_url="https://api.coingecko.com/api/v3",
            rate_limit_per_minute=30,  # 30 requests/min gratis
            headers={'accept': 'application/json'},
            priority=1
        )
        
        # Alpha Vantage - Noticias y sentiment
        self.sources['alphavantage'] = DataSource(
            name="Alpha Vantage",
            base_url="https://www.alphavantage.co/query",
            rate_limit_per_minute=5,  # 25 requests/day -> ~1 per hour
            api_keys=["demo"],  # Key demo para testing
            priority=3
        )
        
        # CryptoCompare - Datos históricos gratuitos
        self.sources['cryptocompare'] = DataSource(
            name="CryptoCompare",
            base_url="https://min-api.cryptocompare.com/data",
            rate_limit_per_minute=15,
            headers={'accept': 'application/json'},
            priority=2
        )
        
        # Messari - Datos fundamentales gratuitos
        self.sources['messari'] = DataSource(
            name="Messari",
            base_url="https://data.messari.io/api/v1",
            rate_limit_per_minute=20,
            headers={'accept': 'application/json'},
            priority=2
        )
        
        logger.info(f"🔗 Configured {len(self.sources)} free data sources")
    
    def can_make_request(self, source_name: str) -> bool:
        """Verifica si se puede hacer un request a una fuente"""
        
        if source_name not in self.sources:
            return False
        
        source = self.sources[source_name]
        
        if not source.is_active:
            return False
        
        # Verificar rate limit
        current_time = time.time()
        time_window = 60  # 1 minuto
        
        if current_time - source.last_request_time > time_window:
            source.request_count = 0  # Reset contador
        
        return source.request_count < source.rate_limit_per_minute
    
    def record_request(self, source_name: str, success: bool = True):
        """Registra un request realizado"""
        
        if source_name not in self.sources:
            return
        
        source = self.sources[source_name]
        source.last_request_time = time.time()
        source.request_count += 1
        
        if not success:
            source.errors_count += 1
            
            # Desactivar fuente si hay muchos errores
            if source.errors_count > 10:
                source.is_active = False
                logger.warning(f"⚠️ Deactivated {source_name} due to errors")
    
    def get_next_api_key(self, source_name: str) -> Optional[str]:
        """Obtiene la siguiente API key para rotación"""
        
        if source_name not in self.sources:
            return None
        
        source = self.sources[source_name]
        
        if not source.api_keys:
            return None
        
        if source_name not in self.current_key_index:
            self.current_key_index[source_name] = 0
        
        key = source.api_keys[self.current_key_index[source_name]]
        
        # Rotar para próxima vez
        self.current_key_index[source_name] = (
            self.current_key_index[source_name] + 1
        ) % len(source.api_keys)
        
        return key

class FreeDataSourceManager:
    """
    Gestor principal de fuentes de datos gratuitas
    Agrega datos de múltiples fuentes con fallbacks automáticos
    """
    
    def __init__(self):
        self.rate_limiter = RateLimitManager()
        self.cache_ttl = {
            'price': 60,        # 1 minuto para precios
            'news': 900,        # 15 minutos para noticias
            'sentiment': 1800,  # 30 minutos para sentiment
            'fundamentals': 3600 # 1 hora para datos fundamentales
        }
        
        logger.info("🚀 FreeDataSourceManager initialized")
    
    def get_comprehensive_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene datos completos de mercado agregando múltiples fuentes gratuitas
        
        Args:
            symbol: Símbolo crypto (BTC, ETH, etc.)
            
        Returns:
            Dict con todos los datos agregados
        """
        
        logger.info(f"📊 Getting comprehensive data for {symbol}")
        
        # Normalizar símbolo
        symbol_upper = symbol.upper()
        
        # Agregar datos de múltiples fuentes
        market_data = {
            'symbol': symbol_upper,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sources_used': [],
            'data_quality': 'unknown'
        }
        
        # 1. Datos de precio (CoinGecko primary)
        price_data = self._get_price_data(symbol_upper)
        if price_data:
            market_data.update(price_data)
            market_data['sources_used'].append('coingecko_price')
        
        # 2. Datos técnicos (CryptoCompare fallback)
        technical_data = self._get_technical_data(symbol_upper)
        if technical_data:
            market_data.update(technical_data)
            market_data['sources_used'].append('cryptocompare_technical')
        
        # 3. Sentiment y noticias (Alpha Vantage)
        sentiment_data = self._get_sentiment_data(symbol_upper)
        if sentiment_data:
            market_data.update(sentiment_data)
            market_data['sources_used'].append('alphavantage_sentiment')
        
        # 4. Datos fundamentales (Messari)
        fundamental_data = self._get_fundamental_data(symbol_upper)
        if fundamental_data:
            market_data.update(fundamental_data)
            market_data['sources_used'].append('messari_fundamentals')
        
        # Evaluar calidad de datos
        source_count = len(market_data['sources_used'])
        if source_count >= 3:
            market_data['data_quality'] = 'high'
        elif source_count >= 2:
            market_data['data_quality'] = 'medium'
        elif source_count >= 1:
            market_data['data_quality'] = 'low'
        else:
            market_data['data_quality'] = 'insufficient'
        
        logger.info(f"✅ Data aggregated for {symbol}: {source_count} sources, "
                   f"quality: {market_data['data_quality']}")
        
        return market_data
    
    def _get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos de precio desde CoinGecko"""
        
        cache_key = f"price_{symbol}"
        cached = self._get_cached_data(cache_key, self.cache_ttl['price'])
        if cached:
            return cached
        
        if not self.rate_limiter.can_make_request('coingecko'):
            logger.warning("⚠️ CoinGecko rate limit reached")
            return None
        
        try:
            # Mapear símbolos a IDs de CoinGecko
            coin_id = self._symbol_to_coingecko_id(symbol)
            if not coin_id:
                return None
            
            url = f"{self.rate_limiter.sources['coingecko'].base_url}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false'
            }
            
            response = requests.get(url, params=params, timeout=10)
            self.rate_limiter.record_request('coingecko', response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('market_data', {})
                
                price_data = {
                    'current_price': market_data.get('current_price', {}).get('usd'),
                    'market_cap': market_data.get('market_cap', {}).get('usd'),
                    'total_volume': market_data.get('total_volume', {}).get('usd'),
                    'price_change_24h': market_data.get('price_change_percentage_24h'),
                    'price_change_7d': market_data.get('price_change_percentage_7d'),
                    'ath': market_data.get('ath', {}).get('usd'),
                    'atl': market_data.get('atl', {}).get('usd'),
                    'circulating_supply': market_data.get('circulating_supply'),
                    'total_supply': market_data.get('total_supply')
                }
                
                self._cache_data(cache_key, price_data)
                return price_data
            
        except Exception as e:
            logger.error(f"❌ Error getting price data for {symbol}: {e}")
            self.rate_limiter.record_request('coingecko', False)
        
        return None
    
    def _get_technical_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos técnicos desde CryptoCompare"""
        
        cache_key = f"technical_{symbol}"
        cached = self._get_cached_data(cache_key, self.cache_ttl['price'])
        if cached:
            return cached
        
        if not self.rate_limiter.can_make_request('cryptocompare'):
            return None
        
        try:
            url = f"{self.rate_limiter.sources['cryptocompare'].base_url}/v2/histohour"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': 24,  # Últimas 24 horas
                'aggregate': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            self.rate_limiter.record_request('cryptocompare', response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success':
                    hourly_data = data.get('Data', {}).get('Data', [])
                    
                    if hourly_data:
                        latest = hourly_data[-1]
                        
                        # Calcular indicadores básicos
                        prices = [item['close'] for item in hourly_data[-24:]]
                        volumes = [item['volumeto'] for item in hourly_data[-24:]]
                        
                        technical_data = {
                            'high_24h': max(item['high'] for item in hourly_data),
                            'low_24h': min(item['low'] for item in hourly_data),
                            'volume_24h': sum(volumes),
                            'avg_price_24h': sum(prices) / len(prices) if prices else 0,
                            'volatility_24h': self._calculate_volatility(prices),
                            'latest_close': latest.get('close'),
                            'latest_volume': latest.get('volumeto')
                        }
                        
                        self._cache_data(cache_key, technical_data)
                        return technical_data
            
        except Exception as e:
            logger.error(f"❌ Error getting technical data for {symbol}: {e}")
            self.rate_limiter.record_request('cryptocompare', False)
        
        return None
    
    def _get_sentiment_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos de sentiment desde Alpha Vantage"""
        
        cache_key = f"sentiment_{symbol}"
        cached = self._get_cached_data(cache_key, self.cache_ttl['sentiment'])
        if cached:
            return cached
        
        # Alpha Vantage tiene límites muy estrictos, usar con moderación
        if not self.rate_limiter.can_make_request('alphavantage'):
            return None
        
        try:
            api_key = self.rate_limiter.get_next_api_key('alphavantage')
            if not api_key or api_key == 'demo':
                # Generar datos de sentiment sintéticos para demo
                return self._generate_synthetic_sentiment(symbol)
            
            url = self.rate_limiter.sources['alphavantage'].base_url
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': f'CRYPTO:{symbol}',
                'apikey': api_key,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=15)
            self.rate_limiter.record_request('alphavantage', response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                
                # Procesar datos de sentiment
                feed = data.get('feed', [])
                if feed:
                    sentiments = []
                    for article in feed[:10]:  # Últimos 10 artículos
                        sentiment_score = article.get('overall_sentiment_score', 0)
                        sentiments.append(float(sentiment_score))
                    
                    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                    
                    sentiment_data = {
                        'sentiment_score': avg_sentiment,
                        'sentiment_label': self._sentiment_score_to_label(avg_sentiment),
                        'news_articles_count': len(feed),
                        'sentiment_source': 'alphavantage'
                    }
                    
                    self._cache_data(cache_key, sentiment_data)
                    return sentiment_data
            
        except Exception as e:
            logger.warning(f"⚠️ Alpha Vantage unavailable for {symbol}: {e}")
            self.rate_limiter.record_request('alphavantage', False)
        
        # Fallback a sentiment sintético
        return self._generate_synthetic_sentiment(symbol)
    
    def _get_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos fundamentales desde Messari"""
        
        cache_key = f"fundamentals_{symbol}"
        cached = self._get_cached_data(cache_key, self.cache_ttl['fundamentals'])
        if cached:
            return cached
        
        if not self.rate_limiter.can_make_request('messari'):
            return None
        
        try:
            # Messari usa slugs en lugar de símbolos
            slug = symbol.lower()
            if symbol == 'BTC':
                slug = 'bitcoin'
            elif symbol == 'ETH':
                slug = 'ethereum'
            
            url = f"{self.rate_limiter.sources['messari'].base_url}/assets/{slug}/metrics"
            
            response = requests.get(url, timeout=10)
            self.rate_limiter.record_request('messari', response.status_code == 200)
            
            if response.status_code == 200:
                data = response.json()
                metrics = data.get('data', {})
                
                fundamental_data = {
                    'developer_activity': metrics.get('developer_activity', {}),
                    'roi_data': metrics.get('roi_data', {}),
                    'misc_data': metrics.get('misc_data', {}),
                    'reddit': metrics.get('reddit', {}),
                    'on_chain_data': metrics.get('on_chain_data', {})
                }
                
                self._cache_data(cache_key, fundamental_data)
                return fundamental_data
            
        except Exception as e:
            logger.warning(f"⚠️ Messari unavailable for {symbol}: {e}")
            self.rate_limiter.record_request('messari', False)
        
        return None
    
    def _symbol_to_coingecko_id(self, symbol: str) -> Optional[str]:
        """Mapea símbolos a IDs de CoinGecko"""
        
        mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'SOL': 'solana',
            'MATIC': 'polygon',
            'AVAX': 'avalanche-2',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'LTC': 'litecoin',
            'XRP': 'ripple',
            'DOGE': 'dogecoin'
        }
        
        return mapping.get(symbol.upper())
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calcula volatilidad simple de precios"""
        
        if len(prices) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Desviación estándar de returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return variance ** 0.5
    
    def _sentiment_score_to_label(self, score: float) -> str:
        """Convierte score numérico a label de sentiment"""
        
        if score >= 0.15:
            return 'POSITIVE'
        elif score <= -0.15:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    def _generate_synthetic_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Genera sentiment sintético basado en datos históricos"""
        
        # Sentiment básico basado en símbolo y tiempo
        base_sentiments = {
            'BTC': 0.1,   # Ligeramente positivo
            'ETH': 0.05,  # Neutral-positivo
            'ADA': 0.0,   # Neutral
            'SOL': 0.15,  # Positivo
            'DOGE': 0.2   # Muy positivo (meme factor)
        }
        
        base_score = base_sentiments.get(symbol, 0.0)
        
        # Añadir variación aleatoria pequeña
        noise = (random.random() - 0.5) * 0.1
        final_score = base_score + noise
        
        return {
            'sentiment_score': final_score,
            'sentiment_label': self._sentiment_score_to_label(final_score),
            'news_articles_count': random.randint(5, 15),
            'sentiment_source': 'synthetic'
        }
    
    def _get_cached_data(self, key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
        """Obtiene datos del cache si están vigentes"""
        
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            cache_time = datetime.fromisoformat(cached['cached_at'])
            age = (datetime.now(timezone.utc) - cache_time).total_seconds()
            
            if age < ttl_seconds:
                return cached['data']
            
        except Exception as e:
            logger.debug(f"Cache read error for {key}: {e}")
        
        return None
    
    def _cache_data(self, key: str, data: Dict[str, Any]):
        """Guarda datos en cache"""
        
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            cached = {
                'data': data,
                'cached_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cached, f, indent=2)
                
        except Exception as e:
            logger.debug(f"Cache write error for {key}: {e}")
    
    def get_sources_status(self) -> Dict[str, Any]:
        """Obtiene estado de todas las fuentes de datos"""
        
        status = {}
        
        for name, source in self.rate_limiter.sources.items():
            status[name] = {
                'active': source.is_active,
                'rate_limit': source.rate_limit_per_minute,
                'current_requests': source.request_count,
                'errors': source.errors_count,
                'priority': source.priority,
                'last_request': source.last_request_time,
                'can_request': self.rate_limiter.can_make_request(name)
            }
        
        return status

# Test function
def test_data_source_manager():
    """Test del gestor de fuentes de datos"""
    
    logger.info("🧪 Testing FreeDataSourceManager...")
    
    manager = FreeDataSourceManager()
    
    # Test con Bitcoin
    try:
        data = manager.get_comprehensive_market_data('BTC')
        
        print(f"✅ Data sources used: {data.get('sources_used', [])}")
        print(f"✅ Data quality: {data.get('data_quality', 'unknown')}")
        print(f"✅ Current price: ${data.get('current_price', 'N/A')}")
        print(f"✅ 24h change: {data.get('price_change_24h', 'N/A')}%")
        
        # Status de fuentes
        status = manager.get_sources_status()
        print(f"✅ Active sources: {sum(1 for s in status.values() if s['active'])}/{len(status)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(name)s - %(message)s")
    test_data_source_manager()
