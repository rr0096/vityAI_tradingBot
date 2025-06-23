# utils/error_handling.py
"""
Sistema robusto de manejo de errores y fallbacks para APIs externas
Incluye circuit breakers, rate limiting y recuperación automática
"""
from __future__ import annotations # Para type hints de clases aún no definidas

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, Callable, Union, List
from datetime import datetime, timedelta, timezone # Asegurar timezone para datetime.now()
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiohttp
# import requests # No usado en la versión async, se puede quitar si no hay uso síncrono.
from functools import wraps
import traceback
# from collections import defaultdict # No usado directamente en la versión final.

logger = logging.getLogger(__name__)

# --- Decoradores Utilitarios (Opcionales si no se usan externamente) ---
def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorador para reintentos síncronos con backoff exponencial."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    current_delay = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. Retrying in {current_delay:.2f}s...")
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
            
            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            if last_exception:
                raise last_exception
            # Esto solo se alcanzaría si max_retries es 0 y falla, o si no hay last_exception
            raise RuntimeError(f"{func.__name__} failed after all retries.")
        return wrapper
    return decorator

def safe_execute(default_return=None):
    """Decorador para ejecución segura síncrona, retornando un valor por defecto en caso de error."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Safe execution failed in {func.__name__}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator

# --- Clases Principales para Manejo de Errores ---

class CircuitState(Enum):
    """Estados posibles de un Circuit Breaker."""
    CLOSED = "closed"      # Funcionando normalmente, las llamadas pasan.
    OPEN = "open"          # Fallando, las llamadas se bloquean inmediatamente.
    HALF_OPEN = "half_open"  # Periodo de prueba, permite algunas llamadas para ver si el servicio se recuperó.

@dataclass
class CircuitBreakerConfig:
    """Configuración para un CircuitBreaker."""
    failure_threshold: int = 5       # Número de fallos para abrir el circuito.
    success_threshold: int = 3       # Número de éxitos en HALF_OPEN para cerrar el circuito.
    timeout_seconds: int = 60        # Tiempo en segundos que el circuito permanece OPEN antes de pasar a HALF_OPEN.
    # reset_timeout_seconds: int = 300 # No usado explícitamente en la lógica actual, timeout_seconds cubre el periodo OPEN.

@dataclass
class APICallResult:
    """Resultado estandarizado de una llamada a API."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    used_cache: bool = False
    used_fallback: bool = False

class CircuitBreaker:
    """Implementación de un Circuit Breaker para proteger llamadas a servicios externos."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.success_count: int = 0 # Usado en estado HALF_OPEN
        self.last_failure_time: Optional[datetime] = None

    def can_execute(self) -> bool:
        """Determina si se puede ejecutar la llamada basado en el estado del circuito."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() > self.config.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0 # Resetear contador de éxitos para el nuevo intento
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN.")
                return True # Permitir el primer intento en HALF_OPEN
            return False # Sigue en OPEN, bloquear llamada
        elif self.state == CircuitState.HALF_OPEN:
            # En HALF_OPEN, se permite la llamada (controlado por el que llama, que luego reporta éxito/fallo)
            return True
        return False # Default, aunque no debería llegar aquí

    def record_success(self):
        """Registra una llamada exitosa."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' RECOVERED, transitioned to CLOSED.")
        elif self.state == CircuitState.CLOSED:
            # Si estaba cerrado y hay un éxito, resetea el contador de fallos (si hubiera alguno esporádico)
            if self.failure_count > 0:
                logger.debug(f"Circuit breaker '{self.name}' had {self.failure_count} previous sporadic failures, resetting on success.")
                self.failure_count = 0
    
    def record_failure(self):
        """Registra una llamada fallida."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.state == CircuitState.HALF_OPEN:
            # Fallo en HALF_OPEN significa que el servicio no se ha recuperado. Volver a OPEN.
            self.state = CircuitState.OPEN
            self.success_count = 0 # Resetear success_count
            logger.warning(f"Circuit breaker '{self.name}' failed in HALF_OPEN state. Transitioning back to OPEN. Failures: {self.failure_count}")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.success_count = 0 # Resetear success_count
                logger.error(f"Circuit breaker '{self.name}' OPENED due to {self.failure_count} failures.")
        # Si ya está OPEN, solo actualiza last_failure_time y failure_count (ya hecho arriba)

class RateLimiter:
    """Rate limiter para controlar la frecuencia de llamadas a una API."""
    
    def __init__(self, max_calls: int, time_window_seconds: int):
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if time_window_seconds <=0:
            raise ValueError("time_window_seconds must be positive")

        self.max_calls = max_calls
        self.time_window = timedelta(seconds=time_window_seconds)
        self.calls: List[datetime] = [] # Almacena timestamps de las llamadas recientes
    
    def can_proceed(self) -> bool:
        """Verifica si se puede realizar una nueva llamada sin exceder el límite."""
        now = datetime.now(timezone.utc)
        
        # Eliminar timestamps de llamadas que ya están fuera de la ventana de tiempo.
        self.calls = [call_ts for call_ts in self.calls if now - call_ts < self.time_window]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Registra una nueva llamada."""
        if self.can_proceed(): # Solo registrar si se permite, aunque la verificación usualmente se hace antes.
            self.calls.append(datetime.now(timezone.utc))
        else:
            # Esto no debería ocurrir si can_proceed() se chequea antes de llamar a record_call().
            logger.warning("RateLimiter: record_call invoked when limit already reached.")

    def time_until_next_call(self) -> float:
        """Tiempo en segundos hasta que se pueda realizar la próxima llamada. 0 si se puede ya."""
        now = datetime.now(timezone.utc)
        self.calls = [call_ts for call_ts in self.calls if now - call_ts < self.time_window] # Limpiar primero

        if len(self.calls) < self.max_calls:
            return 0.0
        
        # Si el límite está alcanzado, calcular el tiempo hasta que la llamada más antigua en la ventana expire.
        if not self.calls: # No debería pasar si len(self.calls) >= self.max_calls
            return 0.0

        oldest_call_in_window = self.calls[0] # Las llamadas se añaden al final, la primera es la más antigua
        time_passed_for_oldest = now - oldest_call_in_window
        wait_time = self.time_window - time_passed_for_oldest
        return max(0.0, wait_time.total_seconds())


class CacheManager:
    """Gestor de cache para respuestas de API."""
    
    def __init__(self, default_ttl_seconds: int = 300):
        self.default_ttl = timedelta(seconds=default_ttl_seconds)
        self.cache: Dict[str, Dict[str, Any]] = {} # key: {data, timestamp, ttl}
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del cache si existe y no ha expirado."""
        entry = self.cache.get(key)
        if entry:
            if datetime.now(timezone.utc) - entry['timestamp'] < entry['ttl']:
                logger.debug(f"Cache HIT for key: {key}")
                return entry['data']
            else:
                # Cache entry ha expirado, eliminarla.
                del self.cache[key]
                logger.debug(f"Cache EXPIRED for key: {key}")
        logger.debug(f"Cache MISS for key: {key}")
        return None
    
    def set(self, key: str, data: Any, ttl_seconds: Optional[int] = None):
        """Guarda un valor en el cache con un TTL específico o el por defecto."""
        ttl_to_use = timedelta(seconds=ttl_seconds) if ttl_seconds is not None else self.default_ttl
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now(timezone.utc),
            'ttl': ttl_to_use
        }
        logger.debug(f"Cache SET for key: {key}, TTL: {ttl_to_use.total_seconds()}s")
    
    def clear(self):
        """Limpia todo el cache."""
        self.cache.clear()
        logger.info("CacheManager: All cache cleared.")
    
    def cleanup(self):
        """Elimina entradas de cache expiradas."""
        now = datetime.now(timezone.utc)
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry['timestamp'] >= entry['ttl']
        ]
        for key in expired_keys:
            del self.cache[key]
        if expired_keys:
            logger.debug(f"CacheManager cleanup: removed {len(expired_keys)} expired entries.")

class RobustAPIClient:
    """Cliente API robusto con circuit breaker, rate limiting y cache para llamadas asyncio."""
    
    def __init__(
        self,
        name: str, # Nombre del cliente API (para logging y stats)
        base_url: str = "",
        circuit_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_calls: int = 10, # Max llamadas
        rate_limit_window_seconds: int = 60, # Por ventana de tiempo en segundos
        cache_default_ttl_seconds: int = 300,
        timeout_seconds: int = 10, # Timeout total para la petición HTTP
        max_retries: int = 3,      # Número de reintentos para la petición HTTP
        retry_base_delay_seconds: float = 1.0 # Delay base para el backoff exponencial
    ):
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        
        self.circuit_breaker = CircuitBreaker(name, circuit_config or CircuitBreakerConfig())
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window_seconds)
        self.cache = CacheManager(cache_default_ttl_seconds)
        
        self.stats = {
            'total_calls_attempted': 0, # Intentos totales (incluye bloqueados)
            'total_calls_executed': 0,  # Llamadas que pasaron protecciones y se ejecutaron
            'successful_calls': 0,
            'failed_calls': 0, # Fallos después de reintentos
            'cache_hits': 0,
            'circuit_breaker_blocks': 0,
            'rate_limit_blocks': 0,
            'total_latency_ms': 0.0, # Suma de latencias para calcular promedio
            'last_call_timestamp': None
        }
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_key: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
        fallback_data: Any = None # Datos a retornar si todo falla
    ) -> APICallResult:
        """Realiza una petición GET robusta."""
        return await self._make_request(
            method='GET',
            endpoint=endpoint,
            params=params,
            headers=headers,
            cache_key=cache_key,
            cache_ttl_seconds=cache_ttl_seconds,
            fallback_data=fallback_data
        )
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None, # Form data
        json_data: Optional[Dict[str, Any]] = None, # JSON body
        headers: Optional[Dict[str, str]] = None,
        fallback_data: Any = None
    ) -> APICallResult:
        """Realiza una petición POST robusta (no cacheable por defecto)."""
        return await self._make_request(
            method='POST',
            endpoint=endpoint,
            data=data,
            json_data=json_data,
            headers=headers,
            fallback_data=fallback_data
            # cache_key y cache_ttl_seconds usualmente no se usan para POST,
            # pero podrían pasarse si la API POST es idempotente y cacheable.
        )
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_key: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
        fallback_data: Any = None
    ) -> APICallResult:
        """Lógica central para realizar una petición HTTP con todas las protecciones."""
        
        self.stats['total_calls_attempted'] += 1
        self.stats['last_call_timestamp'] = datetime.now(timezone.utc)
        request_start_time = time.perf_counter() # Para medir latencia

        # 1. Verificar Cache (si aplica, usualmente para GET)
        if method == 'GET' and cache_key:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.stats['cache_hits'] += 1
                latency_ms = (time.perf_counter() - request_start_time) * 1000
                return APICallResult(
                    success=True, data=cached_data, latency_ms=latency_ms, used_cache=True
                )
        
        # 2. Verificar Circuit Breaker
        if not self.circuit_breaker.can_execute():
            self.stats['circuit_breaker_blocks'] += 1
            error_msg = f"Circuit breaker for '{self.name}' is OPEN."
            logger.warning(error_msg)
            return self._prepare_fallback_result(fallback_data, error_msg, request_start_time)
        
        # 3. Verificar Rate Limiter
        if not self.rate_limiter.can_proceed():
            self.stats['rate_limit_blocks'] += 1
            wait_time_s = self.rate_limiter.time_until_next_call()
            error_msg = f"Rate limit exceeded for '{self.name}'. Need to wait {wait_time_s:.2f}s."
            logger.warning(error_msg)
            # Podría implementarse una espera aquí si wait_time_s es corto, o simplemente fallar.
            # Por ahora, se considera un bloqueo y se usa fallback.
            return self._prepare_fallback_result(fallback_data, error_msg, request_start_time)
        
        # Si se procede, registrar la llamada en el rate limiter
        self.rate_limiter.record_call()
        self.stats['total_calls_executed'] += 1
        
        # 4. Realizar la petición HTTP con reintentos y backoff exponencial
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if self.base_url else endpoint
        
        last_exception_msg = "No attempts made or unknown error."

        for attempt in range(self.max_retries + 1): # +1 para que max_retries sea el número de reintentos
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                    request_kwargs = {'headers': headers or {}}
                    if params: request_kwargs['params'] = params
                    if method == 'POST':
                        if json_data: request_kwargs['json'] = json_data
                        elif data: request_kwargs['data'] = data
                    
                    async with session.request(method, url, **request_kwargs) as response:
                        response_text = await response.text() # Leer el cuerpo una vez
                        
                        if 200 <= response.status < 300: # Éxito
                            self.circuit_breaker.record_success()
                            self.stats['successful_calls'] += 1
                            latency_ms = (time.perf_counter() - request_start_time) * 1000
                            self.stats['total_latency_ms'] += latency_ms
                            
                            try:
                                parsed_response_data = json.loads(response_text)
                            except json.JSONDecodeError:
                                parsed_response_data = response_text # Si no es JSON, devolver texto plano
                            
                            if method == 'GET' and cache_key:
                                self.cache.set(cache_key, parsed_response_data, cache_ttl_seconds)
                            
                            return APICallResult(
                                success=True, data=parsed_response_data, latency_ms=latency_ms
                            )
                        else: # Error HTTP (4xx, 5xx)
                            last_exception_msg = f"HTTP {response.status}: {response_text[:200]}"
                            logger.warning(f"API call to {url} failed (attempt {attempt+1}/{self.max_retries+1}): {last_exception_msg}")
                            # Para errores de cliente (4xx) no reintentar usualmente, salvo 429 (Too Many Requests)
                            if 400 <= response.status < 500 and response.status != 429:
                                break # No reintentar errores 4xx (excepto 429)
                            # Para otros errores (5xx, 429), continuar con reintento si quedan.
                
            except asyncio.TimeoutError:
                last_exception_msg = f"Timeout after {self.timeout_seconds}s on attempt {attempt+1}"
                logger.warning(f"API call to {url} timed out (attempt {attempt+1}/{self.max_retries+1}).")
            except aiohttp.ClientError as e: # Engloba errores de conexión, etc.
                last_exception_msg = f"ClientError on attempt {attempt+1}: {str(e)}"
                logger.warning(f"API call to {url} client error (attempt {attempt+1}/{self.max_retries+1}): {e}")
            
            # Si es el último intento o un error que no se reintenta (4xx), salir del bucle.
            if attempt == self.max_retries or (last_exception_msg.startswith("HTTP 4") and not last_exception_msg.startswith("HTTP 429")):
                break
            
            # Backoff exponencial
            delay = self.retry_base_delay_seconds * (2 ** attempt)
            await asyncio.sleep(delay)
        
        # Si todos los reintentos fallan
        self.circuit_breaker.record_failure()
        self.stats['failed_calls'] += 1
        logger.error(f"API call to {url} failed after all retries. Last error: {last_exception_msg}")
        return self._prepare_fallback_result(fallback_data, last_exception_msg, request_start_time)

    def _prepare_fallback_result(self, fallback_data: Any, error_msg: str, request_start_time: float) -> APICallResult:
        """Prepara un APICallResult cuando se usa fallback o hay un error final."""
        latency_ms = (time.perf_counter() - request_start_time) * 1000
        if fallback_data is not None:
            logger.info(f"Using fallback data for '{self.name}' due to: {error_msg}")
            return APICallResult(
                success=True, data=fallback_data, error=error_msg, latency_ms=latency_ms, used_fallback=True
            )
        else:
            return APICallResult(
                success=False, error=error_msg, latency_ms=latency_ms
            )

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de este cliente API."""
        current_stats = self.stats.copy()
        if current_stats['successful_calls'] > 0:
            avg_latency = current_stats['total_latency_ms'] / current_stats['successful_calls']
        else:
            avg_latency = 0.0
        
        current_stats['avg_latency_ms'] = round(avg_latency, 2)
        current_stats['circuit_state'] = self.circuit_breaker.state.value
        current_stats['circuit_failure_count'] = self.circuit_breaker.failure_count
        
        total_ended_calls = current_stats['successful_calls'] + current_stats['failed_calls']
        if total_ended_calls > 0:
            current_stats['success_rate_pct'] = round((current_stats['successful_calls'] / total_ended_calls) * 100, 2)
        else:
            current_stats['success_rate_pct'] = 0.0
            
        if current_stats['total_calls_executed'] > 0:
             current_stats['cache_hit_rate_pct'] = round((current_stats['cache_hits'] / current_stats['total_calls_executed']) * 100, 2) \
                if current_stats['total_calls_executed'] > current_stats['cache_hits'] else 100.0 # Evitar div por cero si solo hay cache hits
        else:
            current_stats['cache_hit_rate_pct'] = 0.0

        return current_stats
    
    def reset_stats(self):
        """Resetea las estadísticas de este cliente."""
        self.stats = {
            'total_calls_attempted': 0, 'total_calls_executed': 0,
            'successful_calls': 0, 'failed_calls': 0, 'cache_hits': 0,
            'circuit_breaker_blocks': 0, 'rate_limit_blocks': 0,
            'total_latency_ms': 0.0, 'last_call_timestamp': None
        }
        # No resetea el estado del circuit breaker ni del rate limiter, solo las estadísticas de llamadas.
        logger.info(f"Stats reset for API client '{self.name}'.")


# --- Implementaciones Específicas de Clientes API para Fénix Bot ---

class RobustNewsAPI(RobustAPIClient):
    """Cliente robusto para la API de CryptoPanic."""
    def __init__(self, api_token: str):
        super().__init__(
            name="CryptoPanicNews",
            base_url="https://cryptopanic.com/api/v1",
            rate_limit_calls=25, # Un poco menos del límite oficial (30/min) por seguridad
            rate_limit_window_seconds=60,
            cache_default_ttl_seconds=300,  # Noticias cacheadas por 5 minutos
            max_retries=2 # Menos reintentos para APIs de noticias
        )
        self.api_token = api_token
        self.fallback_news_data = { # Estructura similar a la respuesta esperada
            "results": [
                {"title": "Bitcoin stable amidst market calm. No new FUD today.", "kind": "news", "created_at": datetime.now(timezone.utc).isoformat()},
                {"title": "Ethereum developers announce progress on next upgrade.", "kind": "news", "created_at": datetime.now(timezone.utc).isoformat()}
            ]
        }

    async def fetch_news(self, limit: int = 20) -> APICallResult:
        if not self.api_token or self.api_token in ["", "YOUR_TOKEN_HERE", "test_token"]: # Añadido "test_token"
            logger.warning(f"API token for {self.name} is missing or a placeholder. Using fallback data.")
            return APICallResult(success=True, data=self.fallback_news_data, used_fallback=True)
            
        return await self.get(
            endpoint="posts/",
            params={"auth_token": self.api_token, "kind": "news", "public": "true", "limit": str(limit)},
            cache_key=f"news_public_{limit}",
            fallback_data=self.fallback_news_data
        )

class RobustFearGreedAPI(RobustAPIClient):
    """Cliente robusto para la API del Fear & Greed Index de alternative.me."""
    def __init__(self):
        super().__init__(
            name="FearGreedIndex",
            base_url="https://api.alternative.me",
            rate_limit_calls=50, # Límite generoso
            rate_limit_window_seconds=60,
            cache_default_ttl_seconds=3600,  # F&G index no cambia tan frecuentemente (1 hora)
        )
        self.fallback_fg_data = { # Estructura similar a la respuesta esperada
            "name": "Fear and Greed Index",
            "data": [{"value": "50", "value_classification": "Neutral", "timestamp": str(int(time.time()))}]
        }

    async def get_fear_greed_index(self, limit: int = 1) -> APICallResult:
        return await self.get(
            endpoint="fng/",
            params={"limit": str(limit), "format": "json"},
            cache_key=f"fear_greed_limit_{limit}",
            fallback_data=self.fallback_fg_data
        )

class RobustRedditAPI(RobustAPIClient):
    """Cliente robusto para obtener datos de subreddits de Reddit (vía JSON público)."""
    def __init__(self):
        super().__init__(
            name="RedditPublicJSON",
            # No hay base_url fijo ya que la URL completa se construye por subreddit
            rate_limit_calls=25,  # Reddit es sensible al rate limiting (30/min es un buen objetivo conservador)
            rate_limit_window_seconds=60,
            cache_default_ttl_seconds=600, # Posts de Reddit cacheados por 10 minutos
        )
        self.default_subreddits_for_fallback = ["CryptoCurrency", "Bitcoin"]
        self.fallback_reddit_posts = { # Estructura similar a la respuesta esperada
            "data": { "children": [ {"data": {"title": "Reddit API fallback: General crypto discussion ongoing."}} ]}
        }
    
    async def scrape_subreddit(self, subreddit: str, limit: int = 10, sort: str = "new") -> APICallResult:
        # La URL completa se pasa como endpoint
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        # Headers son importantes para Reddit para evitar ser bloqueado como un bot malicioso.
        headers = {"User-Agent": "FenixBot/1.0 (compatible; +http://mybotinfo.com/fenix)"} # Ejemplo de User-Agent
        
        return await self.get(
            endpoint=url, # Endpoint es la URL completa aquí
            params={"limit": str(limit), "raw_json": "1"}, # raw_json=1 para evitar HTML en respuestas
            headers=headers,
            cache_key=f"reddit_{subreddit}_{sort}_{limit}",
            fallback_data=self.fallback_reddit_posts
        )

    async def scrape_multiple_subreddits(
        self,
        subreddits: Optional[List[str]] = None,
        limit_per_sub: int = 5
    ) -> Dict[str, List[str]]: # Retorna un dict de subreddit -> lista de títulos
        """Obtiene posts de múltiples subreddits."""
        target_subreddits = subreddits or self.default_subreddits_for_fallback
        all_titles: Dict[str, List[str]] = {}

        for sub in target_subreddits:
            result = await self.scrape_subreddit(sub, limit_per_sub)
            titles: List[str] = []
            if result.success and isinstance(result.data, dict):
                try:
                    children = result.data.get("data", {}).get("children", [])
                    titles = [post.get("data", {}).get("title", "No title") for post in children if isinstance(post.get("data"), dict)]
                except Exception as e:
                    logger.error(f"Error parsing Reddit data for r/{sub}: {e}. Data: {str(result.data)[:200]}")
                    titles = ["Error parsing Reddit response."] if not result.used_fallback else \
                             [fb_post['data']['title'] for fb_post in self.fallback_reddit_posts['data']['children']]

            elif result.used_fallback and isinstance(result.data, dict): # Si usó fallback y el fallback es el esperado
                 children = result.data.get("data", {}).get("children", [])
                 titles = [post.get("data", {}).get("title", "No title") for post in children if isinstance(post.get("data"), dict)]
            else:
                logger.warning(f"Failed to fetch or parse posts for r/{sub}. Error: {result.error}")
                titles = ["Could not fetch posts for this subreddit."]
            all_titles[sub] = titles
        return all_titles

# --- Factory para Clientes API ---

class APIClientFactory:
    """Factory para crear y gestionar instancias de clientes API robustos (Singleton)."""
    
    _instances: Dict[str, RobustAPIClient] = {} # Almacén de instancias singleton

    @classmethod
    def get_news_client(cls, api_token: str) -> RobustNewsAPI:
        if "news" not in cls._instances or not isinstance(cls._instances["news"], RobustNewsAPI):
            cls._instances["news"] = RobustNewsAPI(api_token)
        # Podría ser necesario actualizar el token si cambia, pero para singleton simple esto es común.
        # elif isinstance(cls._instances["news"], RobustNewsAPI) and cls._instances["news"].api_token != api_token:
        #    cls._instances["news"].api_token = api_token # Actualizar token si es diferente
        return cls._instances["news"] # type: ignore
    
    @classmethod
    def get_fear_greed_client(cls) -> RobustFearGreedAPI:
        if "fear_greed" not in cls._instances or not isinstance(cls._instances["fear_greed"], RobustFearGreedAPI):
            cls._instances["fear_greed"] = RobustFearGreedAPI()
        return cls._instances["fear_greed"] # type: ignore

    @classmethod
    def get_reddit_client(cls) -> RobustRedditAPI:
        if "reddit" not in cls._instances or not isinstance(cls._instances["reddit"], RobustRedditAPI):
            cls._instances["reddit"] = RobustRedditAPI()
        return cls._instances["reddit"] # type: ignore
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Retorna estadísticas agregadas de todos los clientes instanciados."""
        return {name: client.get_stats() for name, client in cls._instances.items()}
    
    @classmethod
    def reset_all_stats(cls):
        """Resetea las estadísticas de todos los clientes instanciados."""
        for client in cls._instances.values():
            client.reset_stats()
        logger.info("APIClientFactory: All client stats reset.")
            
    @classmethod
    def cleanup_all_caches(cls):
        """Ejecuta la limpieza de cache para todos los clientes instanciados."""
        for client in cls._instances.values():
            client.cache.cleanup()
        logger.info("APIClientFactory: All client caches cleaned up.")

# --- Función de Testing (Ejemplo de Uso) ---
async def test_robust_apis():
    """Función de ejemplo para probar los clientes API robustos."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("\n🧪 Testing Robust API Clients...")
    
    # Test News API (requiere un token válido para no usar siempre fallback)
    # Sustituye "YOUR_CRYPTO_PANIC_TOKEN" por tu token real si tienes uno.
    news_token = "YOUR_CRYPTO_PANIC_TOKEN" # o "test_token" para forzar fallback
    news_client = APIClientFactory.get_news_client(api_token=news_token)
    news_result = await news_client.fetch_news(limit=3)
    print(f"\n📰 News API Result (Success: {news_result.success}, Used Fallback: {news_result.used_fallback}):")
    if news_result.success and isinstance(news_result.data, dict):
        for i, post in enumerate(news_result.data.get("results", [])[:3]):
            print(f"  - {post.get('title', 'No Title')}")
    elif news_result.error:
        print(f"  Error: {news_result.error}")

    # Test Fear & Greed API
    fg_client = APIClientFactory.get_fear_greed_client()
    fg_result = await fg_client.get_fear_greed_index()
    print(f"\n😨 Fear & Greed API Result (Success: {fg_result.success}, Used Fallback: {fg_result.used_fallback}):")
    if fg_result.success and isinstance(fg_result.data, dict):
        fg_value = fg_result.data.get("data", [{}])[0].get("value", "N/A")
        fg_class = fg_result.data.get("data", [{}])[0].get("value_classification", "N/A")
        print(f"  - Current Value: {fg_value} ({fg_class})")
    elif fg_result.error:
        print(f"  Error: {fg_result.error}")

    # Test Reddit API
    reddit_client = APIClientFactory.get_reddit_client()
    reddit_results = await reddit_client.scrape_multiple_subreddits(subreddits=["Bitcoin", "nonexistentsub"], limit_per_sub=2)
    print(f"\n🤖 Reddit API Results:")
    for sub, posts in reddit_results.items():
        print(f"  r/{sub}:")
        for i, title in enumerate(posts[:2]):
            print(f"    - {title}")
            
    # Mostrar Estadísticas Agregadas
    print("\n📊📈 API Client Statistics:")
    all_stats = APIClientFactory.get_all_stats()
    for client_name, stats_dict in all_stats.items():
        print(f"  Client: {client_name}")
        for stat_key, stat_val in stats_dict.items():
            print(f"    {stat_key}: {stat_val}")
    
    APIClientFactory.cleanup_all_caches()

if __name__ == "__main__":
    # Para ejecutar el test: python -m utils.error_handling (si está en esa ruta)
    asyncio.run(test_robust_apis())
