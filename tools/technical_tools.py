# tools/technical_tools.py
from __future__ import annotations
from collections import deque
from typing import Dict, Deque, List, Optional, Union, Any
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

try:
    import talib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    talib = None
    logger.warning(
        "TA-Lib not available. Falling back to simplified numpy implementations"
    )

MAXLEN: int = 100 # Max length for primary data buffers
MIN_CANDLES_FOR_CALC = 20 # Minimum candles needed before attempting TA-Lib calculations (some need more, e.g., MA50)
# TA-Lib functions often require a certain minimum number of data points.
# For example, SMA(timeperiod=N) needs at least N points.
# BBANDS(timeperiod=20) needs at least 20.
# RSI(timeperiod=14) needs at least 14+1 = 15.
# ADX(timeperiod=14) needs 2*14-1 = 27.
# ATR(timeperiod=14) needs 14.
# Set MIN_CANDLES_FOR_CALC to a value that covers most common periods, e.g., 30-50.
# Let's use 30 as a general minimum for this set of indicators.
# MA50 will only calculate if len >= 50.
MIN_CANDLES_FOR_RELIABLE_CALC = 30 # Stricter minimum for reliable indicator calculation output

_buffer_lock = threading.RLock()

# Primary data buffers
close_buf: Deque[float] = deque(maxlen=MAXLEN)
high_buf: Deque[float] = deque(maxlen=MAXLEN)
low_buf: Deque[float] = deque(maxlen=MAXLEN)
vol_buf: Deque[float] = deque(maxlen=MAXLEN)

# Buffers for calculated indicator values (stores only the latest value)
# These are updated by _calculate_and_store_all_indicators
# For sequences, we'll slice from the primary buffers or re-calculate on demand.
_latest_indicators_cache: Dict[str, Optional[float]] = {}


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Simple exponential moving average implementation."""
    if len(values) < period:
        return np.array([])
    weights = 2 / (period + 1)
    ema = np.zeros_like(values, dtype=float)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = (values[i] - ema[i - 1]) * weights + ema[i - 1]
    return ema


def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    if len(values) < period + 1:
        return np.array([])
    diff = np.diff(values)
    gain = np.maximum(diff, 0)
    loss = np.abs(np.minimum(diff, 0))
    avg_gain = np.convolve(gain, np.ones(period), 'valid') / period
    avg_loss = np.convolve(loss, np.ones(period), 'valid') / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    # pad result to match input length
    return np.concatenate([np.full(period, np.nan), rsi])


def _macd(values: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    if len(values) < slow + signal:
        return np.array([]), np.array([])
    ema_fast = _ema(values, fast)
    ema_slow = _ema(values, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line[slow - 1 :], signal)
    macd_line = macd_line[(slow - 1) :]
    signal_line = signal_line[-len(macd_line) :]
    return macd_line, signal_line


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) < period + 1:
        return np.array([])
    high_low = high[1:] - low[1:]
    high_close = np.abs(high[1:] - close[:-1])
    low_close = np.abs(low[1:] - close[:-1])
    tr = np.maximum.reduce([high_low, high_close, low_close])
    atr = np.zeros_like(high, dtype=float)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr

# Buffers for indicator sequences (if you want to store sequences of indicators)
# This can consume more memory. Alternatively, calculate sequences on-demand.
# For now, let's keep them if LLM4FTS or other consumers need sequences.
rsi_seq_buf: Deque[float] = deque(maxlen=MAXLEN)
macd_line_seq_buf: Deque[float] = deque(maxlen=MAXLEN)
macd_signal_seq_buf: Deque[float] = deque(maxlen=MAXLEN)
adx_seq_buf: Deque[float] = deque(maxlen=MAXLEN)
atr_seq_buf: Deque[float] = deque(maxlen=MAXLEN)
# obv_seq_buf: Deque[float] = deque(maxlen=MAXLEN) # OBV is cumulative, storing sequence might be less useful than latest
# ma50_seq_buf: Deque[float] = deque(maxlen=MAXLEN)
# upper_band_seq_buf: Deque[float] = deque(maxlen=MAXLEN)
# middle_band_seq_buf: Deque[float] = deque(maxlen=MAXLEN)
# lower_band_seq_buf: Deque[float] = deque(maxlen=MAXLEN)


def _validate_float(value: Any, name: str, positive: bool = False, non_negative: bool = False) -> Optional[float]:
    """Helper to validate and convert a value to float."""
    if not isinstance(value, (int, float)):
        logger.warning(f"Invalid type for {name}: {type(value)}. Expected float or int.")
        return None
    f_value = float(value)
    if np.isnan(f_value) or np.isinf(f_value):
        logger.warning(f"Invalid numeric value for {name}: {f_value}.")
        return None
    if positive and f_value <= 0:
        logger.warning(f"{name} must be positive: {f_value}.")
        return None
    if non_negative and f_value < 0:
        logger.warning(f"{name} must be non-negative: {f_value}.")
        return None
    return f_value

def validate_kline_data(close: float, high: float, low: float, volume: float) -> bool:
    """Validates k-line data components."""
    c = _validate_float(close, "close", positive=True)
    h = _validate_float(high, "high", positive=True)
    l = _validate_float(low, "low", positive=True)
    v = _validate_float(volume, "volume", non_negative=True)

    if None in [c, h, l, v]: # Type or basic value error
        return False
    
    # Ensure type hinting knows these are floats now
    c, h, l = float(c), float(h), float(l)

    if not (l <= c <= h and l <= h): # Logical consistency
        logger.warning(f"OHLC inconsistency: L({l}) <= C({c}) <= H({h}) failed.")
        return False
    return True

def add_kline(close: float, high: float, low: float, volume: float) -> bool:
    """
    Adds a new k-line data point to the buffers and recalculates indicators.
    Thread-safe. Returns True if successful, False otherwise.
    """
    if not validate_kline_data(close, high, low, volume):
        logger.error(f"Invalid k-line data provided to add_kline: C={close}, H={high}, L={low}, V={volume}")
        return False
    
    # Ensure values are float after validation
    close_f, high_f, low_f, vol_f = float(close), float(high), float(low), float(volume)

    with _buffer_lock:
        close_buf.append(close_f)
        high_buf.append(high_f)
        low_buf.append(low_f)
        vol_buf.append(vol_f)

        # Recalculate all indicators and update cache if enough data
        if len(close_buf) >= MIN_CANDLES_FOR_CALC: # Use a less strict minimum for attempting calculation
            _calculate_and_store_all_indicators()
        else:
            # Not enough data yet, clear stale cache
            _latest_indicators_cache.clear()
            logger.debug(f"Not enough candles ({len(close_buf)}/{MIN_CANDLES_FOR_CALC}) to calculate indicators yet.")
        return True

def _calculate_and_store_all_indicators() -> None:
    """
    Internal function to calculate all TA indicators and store their latest values.
    Assumes _buffer_lock is already acquired.
    """
    global _latest_indicators_cache # Modifying global cache

    current_len = len(close_buf)
    if current_len < MIN_CANDLES_FOR_CALC: # Double check, though called after initial check
        return

    # Prepare numpy arrays from deques
    # Slicing deques: list(d)[-N:] is inefficient. Convert to numpy array once.
    close_arr = np.array(close_buf, dtype=np.float64)
    high_arr = np.array(high_buf, dtype=np.float64)
    low_arr = np.array(low_buf, dtype=np.float64)
    vol_arr = np.array(vol_buf, dtype=np.float64)

    temp_cache: Dict[str, Optional[float]] = {}

    # Helper to safely get last valid value from a TA-Lib output array
    def get_last_valid(arr: np.ndarray, name: str) -> Optional[float]:
        if arr is not None and len(arr) > 0:
            val = arr[-1]
            if np.isfinite(val): # Checks for NaN and Inf
                return float(val)
            else:
                logger.debug(f"Indicator '{name}' last value is invalid ({val}).")
        return None

    # RSI
    if current_len >= 14 + 1:
        try:
            if talib:
                temp_cache["rsi"] = get_last_valid(talib.RSI(close_arr, timeperiod=14), "RSI")
            else:
                temp_cache["rsi"] = get_last_valid(_rsi(close_arr, 14), "RSI")
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
    
    # MACD
    if current_len >= 26 + 9 -1:
        try:
            if talib:
                macd, macdsignal, _ = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
            else:
                macd, macdsignal = _macd(close_arr, 12, 26, 9)
            temp_cache["macd_line"] = get_last_valid(macd, "MACD Line")
            temp_cache["signal_line"] = get_last_valid(macdsignal, "MACD Signal")
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")

    # Bollinger Bands
    if current_len >= 20 and talib:
        try:
            upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            temp_cache["upper_band"] = get_last_valid(upper, "BB Upper")
            temp_cache["middle_band"] = get_last_valid(middle, "BB Middle")
            temp_cache["lower_band"] = get_last_valid(lower, "BB Lower")
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")

    # MA50
    if current_len >= 50 and talib:
        try:
            temp_cache["ma50"] = get_last_valid(talib.SMA(close_arr, timeperiod=50), "MA50")
        except Exception as e:
            logger.warning(f"Error calculating MA50: {e}")
    
    # ADX
    if current_len >= 14 * 2 -1 and talib:
        try:
            temp_cache["adx"] = get_last_valid(talib.ADX(high_arr, low_arr, close_arr, timeperiod=14), "ADX")
        except Exception as e:
            logger.warning(f"Error calculating ADX: {e}")

    # OBV
    if talib:
        try:
            temp_cache["obv"] = get_last_valid(talib.OBV(close_arr, vol_arr), "OBV")
        except Exception as e:
            logger.warning(f"Error calculating OBV: {e}")
    
    # ATR
    if current_len >= 14 + 1:
        try:
            if talib:
                temp_cache["atr"] = get_last_valid(talib.ATR(high_arr, low_arr, close_arr, timeperiod=14), "ATR")
            else:
                temp_cache["atr"] = get_last_valid(_atr(high_arr, low_arr, close_arr, 14), "ATR")
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")

    # Update indicator sequence buffers (optional, if needed elsewhere)
    if temp_cache.get("rsi") is not None: rsi_seq_buf.append(temp_cache["rsi"])
    if temp_cache.get("macd_line") is not None: macd_line_seq_buf.append(temp_cache["macd_line"])
    if temp_cache.get("signal_line") is not None: macd_signal_seq_buf.append(temp_cache["signal_line"])
    if temp_cache.get("adx") is not None: adx_seq_buf.append(temp_cache["adx"])
    if temp_cache.get("atr") is not None: atr_seq_buf.append(temp_cache["atr"])
    
    # Update the main cache
    _latest_indicators_cache = temp_cache
    # logger.debug(f"Indicators cache updated: {_latest_indicators_cache}")


def get_current_indicators() -> Dict[str, Any]:
    """
    Returns a dictionary of the latest calculated technical indicators.
    Returns empty if not enough data. Thread-safe.
    """
    with _buffer_lock:
        if len(close_buf) < MIN_CANDLES_FOR_RELIABLE_CALC: # Use stricter minimum for returning values
            logger.debug(f"Not enough data for reliable indicators: {len(close_buf)}/{MIN_CANDLES_FOR_RELIABLE_CALC}")
            return {}
        
        # If cache is empty but we have enough data, try to calculate
        if not _latest_indicators_cache and len(close_buf) >= MIN_CANDLES_FOR_CALC:
            _calculate_and_store_all_indicators()

        # Copy the cache to avoid returning internal mutable object
        indicators = _latest_indicators_cache.copy()
        
        # Add essential non-TA-Lib data
        try:
            indicators["last_price"] = float(close_buf[-1])
            indicators["curr_vol"] = float(vol_buf[-1])
            
            vol_period = min(20, len(vol_buf))
            if vol_period > 0:
                vol_slice = list(vol_buf)[-vol_period:]
                indicators["avg_vol_20"] = float(np.mean(vol_slice))
            else:
                indicators["avg_vol_20"] = 0.0
        except IndexError:
            logger.error("Error accessing latest close/volume, buffers might be empty despite length check.")
            return {} # Critical data missing
        except ValueError as e:
            logger.error(f"ValueError for basic data: {e}")
            return {}


        # Filter out any None values before returning, consumers should handle missing keys
        return {k: v for k, v in indicators.items() if v is not None}

def get_indicator_sequences(sequence_length: int = 10) -> Dict[str, List[float]]:
    """
    Returns recent sequences of specified indicators. Thread-safe.
    """
    with _buffer_lock:
        if len(close_buf) < max(MIN_CANDLES_FOR_RELIABLE_CALC, sequence_length):
            logger.debug(f"Not enough data for indicator sequences: {len(close_buf)} needed {max(MIN_CANDLES_FOR_RELIABLE_CALC, sequence_length)}")
            return {}

        # Ensure latest indicators (and thus sequences) are up-to-date
        if not _latest_indicators_cache and len(close_buf) >= MIN_CANDLES_FOR_CALC:
             _calculate_and_store_all_indicators()

        sequences: Dict[str, List[float]] = {}
        
        # Helper to safely get sequence
        def _get_seq(buf: Deque[float], name: str) -> List[float]:
            if len(buf) >= sequence_length:
                seq = list(buf)[-sequence_length:]
                if all(isinstance(x, (int, float)) and np.isfinite(x) for x in seq):
                    return seq
                else:
                    logger.warning(f"Sequence for '{name}' contains invalid values, returning empty.")
                    return []
            return []

        sequences["close_seq"] = _get_seq(close_buf, "close")
        sequences["high_seq"] = _get_seq(high_buf, "high")
        sequences["low_seq"] = _get_seq(low_buf, "low")
        sequences["volume_seq"] = _get_seq(vol_buf, "volume")
        sequences["rsi_seq"] = _get_seq(rsi_seq_buf, "rsi")
        sequences["macd_line_seq"] = _get_seq(macd_line_seq_buf, "macd_line")
        sequences["macd_signal_seq"] = _get_seq(macd_signal_seq_buf, "macd_signal")
        sequences["adx_seq"] = _get_seq(adx_seq_buf, "adx")
        sequences["atr_seq"] = _get_seq(atr_seq_buf, "atr")
        
        return {k: v for k, v in sequences.items() if v} # Return only non-empty sequences

def clear_all_buffers() -> None:
    """Clears all data and indicator buffers. Thread-safe."""
    with _buffer_lock:
        close_buf.clear()
        high_buf.clear()
        low_buf.clear()
        vol_buf.clear()
        
        rsi_seq_buf.clear()
        macd_line_seq_buf.clear()
        macd_signal_seq_buf.clear()
        adx_seq_buf.clear()
        atr_seq_buf.clear()
        
        _latest_indicators_cache.clear()
        logger.info("All technical tool buffers and cache cleared.")

def get_buffer_status() -> Dict[str, int]:
    """Returns the current length of primary data buffers."""
    with _buffer_lock:
        return {
            "close_buffer_len": len(close_buf),
            "high_buffer_len": len(high_buf),
            "low_buffer_len": len(low_buf),
            "volume_buffer_len": len(vol_buf),
            "rsi_sequence_len": len(rsi_seq_buf),
            "max_buffer_len": MAXLEN,
            "min_candles_for_calc": MIN_CANDLES_FOR_CALC,
            "min_candles_for_reliable_output": MIN_CANDLES_FOR_RELIABLE_CALC
        }

# Alias for backward compatibility if calc_indicators was used elsewhere
calc_indicators = get_current_indicators
