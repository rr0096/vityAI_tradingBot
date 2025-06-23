# agents/llm4fts_implementation.py
"""
Implementation of LLM4FTS: Converts numerical time series data to natural language descriptions.
Based on the paper "LLM4FTS: Enhancing LLMs with Financial Time Series"
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Deque, Optional # Explicit Deque import
from collections import deque # Keep for direct usage if any, though TypingDeque is preferred for type hints
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class LLM4FTSConverter:
    """Converts numerical trading data (OHLCV, indicators) to natural language descriptions."""
    
    def __init__(self):
        self.thresholds = {
            'price_change_pct': {'small': 0.2, 'medium': 0.5, 'large': 1.0},
            'volume_ratio': {'low': 0.6, 'normal': 1.5, 'high': 2.5},
            'volatility_atr_pct': {'low': 0.4, 'medium': 1.2, 'high': 2.5},
            'rsi': {'extreme_oversold': 20, 'oversold': 30, 'neutral_low': 45, 'neutral_high': 55, 'overbought': 70, 'extreme_overbought': 80},
            'trend_slope_pct_per_candle': {'strong_down': -0.05, 'down': -0.015, 'up': 0.015, 'strong_up': 0.05},
            'ma_proximity_pct': 0.5
        }
        self.min_periods_for_analysis = 15

    def convert_ohlcv_to_text(self, candles: Deque[Tuple[float, float, float, float]], current_metrics: Dict[str, Any]) -> str:
        if not candles or len(candles) < self.min_periods_for_analysis:
            return "Insufficient candle data for detailed temporal analysis."
        
        try:
            if not all(isinstance(c, (tuple, list)) and len(c) == 4 for c in candles):
                logger.error("Invalid candle structure in input deque for LLM4FTSConverter.")
                return "Invalid candle data structure."

            close_prices = np.array([c[0] for c in candles], dtype=float)
            high_prices = np.array([c[1] for c in candles], dtype=float)
            low_prices = np.array([c[2] for c in candles], dtype=float)
            volumes = np.array([c[3] for c in candles], dtype=float)
        except (TypeError, IndexError) as e:
            logger.error(f"Error extracting data from candles for LLM4FTS: {e}")
            return "Error processing candle data."

        if not (len(close_prices) == len(high_prices) == len(low_prices) == len(volumes)):
            logger.error("Mismatched lengths in OHLCV data for LLM4FTSConverter.")
            return "Inconsistent candle data lengths."

        trend_text = self._analyze_trend(close_prices)
        price_text = self._analyze_current_price_context(close_prices[-1], close_prices, high_prices, low_prices, current_metrics)
        volatility_text = self._analyze_volatility(high_prices, low_prices, close_prices, current_metrics.get('atr'))
        indicators_text = self._summarize_technical_indicators(current_metrics)
        patterns_text = self._detect_simple_price_patterns(close_prices, high_prices, low_prices, volumes) # Pass volumes
        volume_text = self._analyze_recent_volume(volumes, close_prices)
        
        last_price = close_prices[-1]
        first_price = close_prices[0]
        period_change_pct = ((last_price - first_price) / first_price * 100) if first_price > 1e-9 else 0.0

        full_description = (
            f"TEMPORAL MARKET OVERVIEW ({len(candles)} periods):\n"
            f"{trend_text}\n\n"
            f"{price_text}\n\n"
            f"{volatility_text}\n\n"
            f"{indicators_text}\n\n"
            f"{patterns_text}\n\n"
            f"{volume_text}\n\n"
            f"SUMMARY: Current Price: ${last_price:.4f}. Change over period: {period_change_pct:+.2f}%."
        )
        return full_description.strip()

    def _analyze_trend(self, prices: np.ndarray) -> str:
        n = len(prices)
        if n < 5: return "TREND: Insufficient data for trend analysis."

        x = np.arange(n)
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        except ValueError as e:
            logger.warning(f"Linregress failed for trend analysis: {e}")
            return "TREND: Calculation error."
            
        r_squared = r_value**2
        
        mean_price = prices.mean()
        # Ensure mean_price is not zero before division
        slope_pct_per_period = (slope / mean_price * 100) if mean_price > 1e-9 else 0.0

        direction = "lateral"
        if slope_pct_per_period > self.thresholds['trend_slope_pct_per_candle']['strong_up']: direction = "fuertemente alcista"
        elif slope_pct_per_period > self.thresholds['trend_slope_pct_per_candle']['up']: direction = "alcista"
        elif slope_pct_per_period < self.thresholds['trend_slope_pct_per_candle']['strong_down']: direction = "fuertemente bajista"
        elif slope_pct_per_period < self.thresholds['trend_slope_pct_per_candle']['down']: direction = "bajista"
        
        strength = "muy clara (R² > 0.7)" if r_squared > 0.7 else \
                   "moderadamente clara (R² > 0.4)" if r_squared > 0.4 else \
                   "débil (R² <= 0.4)"
        
        short_term_n = n // 3
        short_trend_desc = "Pocos datos para tendencia reciente."
        if short_term_n >= 3:
            short_term_prices = prices[-short_term_n:]
            short_term_mean_price = short_term_prices.mean()
            if short_term_mean_price > 1e-9: # Check for zero division
                slope_short, _, _, _, _ = stats.linregress(x[-short_term_n:], short_term_prices)
                slope_short_pct = (slope_short / short_term_mean_price * 100)
                short_trend_desc = f"Tendencia más reciente ({short_term_n} per.): {slope_short_pct:+.2f}%/per."
        
        return f"TREND: General: {direction}, {strength} (pendiente {slope_pct_per_period:+.2f}%/per.). {short_trend_desc}"

    def _analyze_current_price_context(self, current_price: float, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, current_metrics: Dict[str, Any]) -> str:
        period_high = highs.max()
        period_low = lows.min()
        period_range = period_high - period_low

        position_in_range_pct = ((current_price - period_low) / period_range * 100) if period_range > 1e-9 else 50.0
        
        pos_desc = "cerca del máximo del período" if position_in_range_pct > 80 else \
                   "cerca del mínimo del período" if position_in_range_pct < 20 else \
                   "en la zona media del período"
        
        ma_proximity_text = ""
        ma20 = current_metrics.get('middle_band') # BB Middle is often an SMA20
        ma50 = current_metrics.get('ma50')

        if isinstance(ma20, (int, float)) and ma20 > 1e-9 and current_price > 1e-9:
            if abs(current_price - ma20) / current_price * 100 < self.thresholds['ma_proximity_pct']:
               ma_proximity_text += f" Cerca de MA20 (${ma20:.4f})."
        if isinstance(ma50, (int, float)) and ma50 > 1e-9 and current_price > 1e-9:
            if abs(current_price - ma50) / current_price * 100 < self.thresholds['ma_proximity_pct']:
               ma_proximity_text += f" Cerca de MA50 (${ma50:.4f})."

        return f"PRICE CONTEXT: Actual ${current_price:.4f}, {pos_desc} ({position_in_range_pct:.0f}% del rango). Rango período: ${period_low:.4f}-${period_high:.4f}.{ma_proximity_text}"

    def _analyze_volatility(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, current_atr: Optional[float]) -> str:
        if len(closes) < 2: return "VOLATILITY: Insufficient data."

        atr_pct_text = "N/A"
        vol_level = "indeterminada (ATR no disponible)"
        if current_atr is not None and isinstance(current_atr, (int, float)) and current_atr > 0 and closes[-1] > 1e-9:
            atr_pct = (current_atr / closes[-1]) * 100
            atr_pct_text = f"{atr_pct:.2f}%"
            if atr_pct < self.thresholds['volatility_atr_pct']['low']: vol_level = "baja"
            elif atr_pct < self.thresholds['volatility_atr_pct']['medium']: vol_level = "moderada"
            else: vol_level = "alta"
            
        candle_ranges = highs - lows
        avg_candle_range = candle_ranges.mean() if len(candle_ranges) > 0 else 0.0
        recent_candle_range_avg = candle_ranges[-5:].mean() if len(candle_ranges) >= 5 else avg_candle_range
        
        range_comp_text = ""
        if avg_candle_range > 1e-9: # Avoid division by zero or meaningless ratio
            range_ratio = recent_candle_range_avg / avg_candle_range
            if range_ratio > 1.5: range_comp_text = "Rangos de vela recientes notablemente mayores al promedio."
            elif range_ratio < 0.6: range_comp_text = "Rangos de vela recientes notablemente menores (compresión)."
        
        return f"VOLATILITY: Nivel {vol_level} (ATR actual: {atr_pct_text}). {range_comp_text}"

    def _summarize_technical_indicators(self, metrics: Dict[str, Any]) -> str:
        parts = []
        rsi = metrics.get('rsi')
        if rsi is not None and isinstance(rsi, (int, float)):
            if rsi < self.thresholds['rsi']['extreme_oversold']: rsi_desc = f"extremadamente sobreventa ({rsi:.1f})"
            elif rsi < self.thresholds['rsi']['oversold']: rsi_desc = f"sobreventa ({rsi:.1f})"
            elif rsi > self.thresholds['rsi']['extreme_overbought']: rsi_desc = f"extremadamente sobrecompra ({rsi:.1f})"
            elif rsi > self.thresholds['rsi']['overbought']: rsi_desc = f"sobrecompra ({rsi:.1f})"
            elif rsi < self.thresholds['rsi']['neutral_low']: rsi_desc = f"lado bajo de neutral ({rsi:.1f})"
            elif rsi > self.thresholds['rsi']['neutral_high']: rsi_desc = f"lado alto de neutral ({rsi:.1f})"
            else: rsi_desc = f"neutral ({rsi:.1f})"
            parts.append(f"RSI: {rsi_desc}")

        macd_line = metrics.get('macd_line')
        signal_line = metrics.get('signal_line')
        if macd_line is not None and signal_line is not None and isinstance(macd_line, (int,float)) and isinstance(signal_line, (int,float)):
            macd_hist = macd_line - signal_line
            macd_desc = "cruce alcista" if macd_hist > 0.0001 else "cruce bajista" if macd_hist < -0.0001 else "plano" # Added small threshold
            parts.append(f"MACD: {macd_desc} (Hist: {macd_hist:.4f})")

        adx = metrics.get('adx')
        if adx is not None and isinstance(adx, (int,float)):
            adx_strength = "fuerte" if adx > 35 else "moderada" if adx > 20 else "débil/ausente" # Adjusted ADX thresholds slightly
            parts.append(f"ADX: {adx_strength} ({adx:.1f})")
            
        bb_upper = metrics.get('upper_band')
        bb_middle = metrics.get('middle_band') # Often SMA20
        bb_lower = metrics.get('lower_band')
        price = metrics.get('last_price')
        if all(isinstance(x, (int,float)) and x is not None for x in [bb_upper, bb_middle, bb_lower, price]):
            # Ensure all values are valid floats before proceeding
            bb_upper, bb_middle, bb_lower, price = float(bb_upper), float(bb_middle), float(bb_lower), float(price)
            if bb_middle > 1e-9 and (bb_upper - bb_lower) > 1e-9 : # Check for valid band width and middle band
                bb_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100
                # Squeeze definition based on volatility thresholds
                bb_squeeze_threshold = self.thresholds['volatility_atr_pct']['low'] * 1.5 # Example: squeeze if BB width < 0.6% (0.4*1.5)
                bb_squeeze_text = f" (Squeeze! Ancho: {bb_width_pct:.2f}%)" if bb_width_pct < bb_squeeze_threshold else f" (Ancho: {bb_width_pct:.2f}%)"
                
                if price > bb_upper: parts.append(f"Precio SOBRE BB Superior{bb_squeeze_text}")
                elif price < bb_lower: parts.append(f"Precio BAJO BB Inferior{bb_squeeze_text}")
                else: parts.append(f"Precio DENTRO de BBs{bb_squeeze_text}")

        if not parts: return "INDICATORS: No hay datos de indicadores clave disponibles."
        return "INDICATORS:\n- " + "\n- ".join(parts)

    def _detect_simple_price_patterns(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> str:
        n = len(closes)
        if n < 3: return "PATTERNS: Insuficientes datos."
        
        patterns = []
        # Last candle analysis
        last_c, last_h, last_l, last_v = closes[-1], highs[-1], lows[-1], volumes[-1]
        prev_c = closes[-2]
        
        body_abs = abs(last_c - prev_c)
        wick_range = last_h - last_l
        
        if wick_range > 1e-9: # Avoid division by zero
            # Doji/Spinning Top
            if body_abs / wick_range < 0.2 and (abs(last_c - last_l) / wick_range > 0.3 or abs(last_h - last_c) / wick_range > 0.3):
                patterns.append("Última vela tipo Doji/Spinning Top (indecisión).")
            # Marubozu (simplified)
            elif body_abs / wick_range > 0.85:
                 pattern_name = "Marubozu Alcista" if last_c > prev_c else "Marubozu Bajista"
                 patterns.append(f"Última vela tipo {pattern_name} (fuerte convicción).")
        
        # Engulfing (simplified, checks last two full candles)
        if n >= 3:
            # Current candle (index -1), Previous candle (index -2)
            o_curr, c_curr, h_curr, l_curr = closes[-2], closes[-1], highs[-1], lows[-1] # Open approx by prev close
            o_prev, c_prev, h_prev, l_prev = closes[-3], closes[-2], highs[-2], lows[-2] # Open approx by prev close
            
            body_curr_abs = abs(c_curr - o_curr)
            body_prev_abs = abs(c_prev - o_prev)

            is_curr_bullish = c_curr > o_curr
            is_prev_bearish = c_prev < o_prev
            if is_curr_bullish and is_prev_bearish: # Potential Bullish Engulfing
                if c_curr > o_prev and o_curr < c_prev and body_curr_abs > body_prev_abs:
                    patterns.append("Posible Engulfing Alcista (vela -1 envuelve a vela -2).")
            
            is_curr_bearish = c_curr < o_curr
            is_prev_bullish = c_prev > o_prev
            if is_curr_bearish and is_prev_bullish: # Potential Bearish Engulfing
                if c_curr < o_prev and o_curr > c_prev and body_curr_abs > body_prev_abs:
                    patterns.append("Posible Engulfing Bajista (vela -1 envuelve a vela -2).")

        # Consecutive up/down closes (simple momentum)
        if n >= 4:
            if all(closes[-i] > closes[-(i+1)] for i in range(1,4)): # 3 consecutive up closes
                patterns.append("Momentum alcista (3 cierres verdes consecutivos).")
            elif all(closes[-i] < closes[-(i+1)] for i in range(1,4)): # 3 consecutive down closes
                patterns.append("Momentum bajista (3 cierres rojos consecutivos).")

        if not patterns: return "PATTERNS: No se detectaron patrones simples obvios en las últimas velas."
        return "PATTERNS:\n- " + "\n- ".join(patterns)

    def _analyze_recent_volume(self, volumes: np.ndarray, closes: np.ndarray) -> str:
        n = len(volumes)
        if n < 5: return "VOLUME: Insufficient data."

        avg_vol_short = volumes[-5:].mean()
        avg_vol_period = volumes.mean()
        
        vol_ratio_to_avg = avg_vol_short / avg_vol_period if avg_vol_period > 1e-9 else 1.0
        
        vol_level_desc = "normal"
        if vol_ratio_to_avg > self.thresholds['volume_ratio']['high']: vol_level_desc = "muy alto"
        elif vol_ratio_to_avg > self.thresholds['volume_ratio']['normal']: vol_level_desc = "alto"
        elif vol_ratio_to_avg < self.thresholds['volume_ratio']['low']: vol_level_desc = "bajo"
        
        vol_price_corr_text = ""
        if n >= 5:
            # Check if volume on last candle confirms price move direction
            if (closes[-1] > closes[-2] and volumes[-1] > avg_vol_short * 1.1): # Price up, volume up
                vol_price_corr_text = "Volumen reciente confirma movimiento alcista."
            elif (closes[-1] < closes[-2] and volumes[-1] > avg_vol_short * 1.1): # Price down, volume up
                vol_price_corr_text = "Volumen reciente alto en caída (posible clímax o continuación)."
            elif (closes[-1] > closes[-2] and volumes[-1] < avg_vol_short * 0.9): # Price up, volume down
                vol_price_corr_text = "Volumen reciente bajo en subida (divergencia, posible debilidad)."
            elif (closes[-1] < closes[-2] and volumes[-1] < avg_vol_short * 0.9): # Price down, volume down
                 vol_price_corr_text = "Volumen reciente bajo en caída (posible agotamiento bajista)."

        return f"VOLUME: Nivel reciente {vol_level_desc} (ratio {vol_ratio_to_avg:.2f} vs promedio del período). {vol_price_corr_text}"
