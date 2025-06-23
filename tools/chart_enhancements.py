# tools/chart_enhancements.py
"""
Chart enhancement module that extends the existing chart_generator.py
Provides additional pattern detection and analysis capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Deque as TypingDeque
import logging
from dataclasses import dataclass
from enum import Enum

# Import the original chart generator
from .chart_generator import ChartGenerator, generate_chart_for_visual_agent

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns that can be detected"""
    CANDLESTICK = "candlestick"
    TECHNICAL_SIGNAL = "technical_signal"
    TREND_LINE = "trend_line"

@dataclass
class ChartPattern:
    """Represents a detected chart pattern"""
    pattern_type: PatternType
    name: str
    confidence: float
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    price_level: Optional[float] = None
    description: str = ""
    bullish_signal: Optional[bool] = None

@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level"""
    price: float
    strength: float
    touches: int
    is_support: bool

class EnhancedPatternDetector:
    """Enhanced pattern detection for trading charts"""
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect basic candlestick patterns with improved logic"""
        if len(df) < 3:
            return []
        
        patterns = []
        
        try:
            # Convert to numpy arrays for faster processing
            opens = df['Open'].astype(float).values
            highs = df['High'].astype(float).values
            lows = df['Low'].astype(float).values
            closes = df['Close'].astype(float).values
            
            for i in range(2, len(df)):
                # Doji pattern
                patterns.extend(self._detect_doji(i, opens, highs, lows, closes))
                
                # Hammer pattern
                patterns.extend(self._detect_hammer(i, opens, highs, lows, closes))
                
                # Engulfing patterns (need previous candle)
                if i > 0:
                    patterns.extend(self._detect_engulfing(i, opens, highs, lows, closes))
                
                # Shooting star
                patterns.extend(self._detect_shooting_star(i, opens, highs, lows, closes))
        
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
        
        return [p for p in patterns if p.confidence >= self.min_confidence]
    
    def _detect_doji(self, i: int, opens, highs, lows, closes) -> List[ChartPattern]:
        """Detect Doji pattern"""
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        body = abs(c - o)
        total_range = h - l
        
        if total_range > 0 and body / total_range < 0.1:
            confidence = 0.8 if body / total_range < 0.05 else 0.6
            return [ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Doji",
                confidence=confidence,
                start_index=i,
                end_index=i,
                price_level=c,
                description="Indecision pattern - potential reversal",
                bullish_signal=None
            )]
        return []
    
    def _detect_hammer(self, i: int, opens, highs, lows, closes) -> List[ChartPattern]:
        """Detect Hammer pattern"""
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        
        # Hammer criteria: long lower shadow, small upper shadow, small body
        if (body > 0 and lower_shadow > body * 2 and upper_shadow < body * 0.5):
            # Check if it's in a downtrend (simplified check)
            in_downtrend = i > 5 and c < np.mean(closes[i-5:i])
            confidence = 0.8 if in_downtrend else 0.6
            
            return [ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Hammer",
                confidence=confidence,
                start_index=i,
                end_index=i,
                price_level=c,
                description="Potential bullish reversal after downtrend",
                bullish_signal=True
            )]
        return []
    
    def _detect_shooting_star(self, i: int, opens, highs, lows, closes) -> List[ChartPattern]:
        """Detect Shooting Star pattern"""
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        
        # Shooting star criteria: long upper shadow, small lower shadow, small body
        if (body > 0 and upper_shadow > body * 2 and lower_shadow < body * 0.5):
            # Check if it's in an uptrend
            in_uptrend = i > 5 and c > np.mean(closes[i-5:i])
            confidence = 0.8 if in_uptrend else 0.6
            
            return [ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Shooting Star",
                confidence=confidence,
                start_index=i,
                end_index=i,
                price_level=c,
                description="Potential bearish reversal after uptrend",
                bullish_signal=False
            )]
        return []
    
    def _detect_engulfing(self, i: int, opens, highs, lows, closes) -> List[ChartPattern]:
        """Detect Engulfing patterns"""
        patterns = []
        
        # Current and previous candle data
        curr_o, curr_c = opens[i], closes[i]
        prev_o, prev_c = opens[i-1], closes[i-1]
        
        curr_body = abs(curr_c - curr_o)
        prev_body = abs(prev_c - prev_o)
        
        # Bullish Engulfing
        if (curr_c > curr_o and prev_c < prev_o and  # Current bullish, prev bearish
            curr_c > prev_o and curr_o < prev_c and curr_body > prev_body * 1.1):
            patterns.append(ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Bullish Engulfing",
                confidence=0.85,
                start_index=i-1,
                end_index=i,
                price_level=curr_c,
                description="Strong bullish reversal signal",
                bullish_signal=True
            ))
        
        # Bearish Engulfing
        elif (curr_c < curr_o and prev_c > prev_o and  # Current bearish, prev bullish
              curr_c < prev_o and curr_o > prev_c and curr_body > prev_body * 1.1):
            patterns.append(ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Bearish Engulfing",
                confidence=0.85,
                start_index=i-1,
                end_index=i,
                price_level=curr_c,
                description="Strong bearish reversal signal",
                bullish_signal=False
            ))
        
        return patterns
    
    def detect_support_resistance_levels(self, df: pd.DataFrame, 
                                       lookback: int = 20, min_touches: int = 2,
                                       tolerance_pct: float = 0.5) -> List[SupportResistanceLevel]:
        """Detect support and resistance levels using improved algorithm"""
        if len(df) < lookback * 2:
            return []
        
        levels = []
        highs = df['High'].astype(float).values
        lows = df['Low'].astype(float).values
        
        try:
            # Find pivot points
            for i in range(lookback, len(df) - lookback):
                # Check for pivot high (resistance)
                if self._is_pivot_high(i, highs, lookback):
                    level = self._validate_sr_level(highs[i], df, True, min_touches, tolerance_pct)
                    if level:
                        levels.append(level)
                
                # Check for pivot low (support)
                if self._is_pivot_low(i, lows, lookback):
                    level = self._validate_sr_level(lows[i], df, False, min_touches, tolerance_pct)
                    if level:
                        levels.append(level)
            
            # Merge similar levels and sort by strength
            levels = self._merge_similar_levels(levels, tolerance_pct)
            levels.sort(key=lambda x: x.strength, reverse=True)
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance levels: {e}")
        
        return levels[:10]  # Return top 10 levels
    
    def _is_pivot_high(self, i: int, highs, lookback: int) -> bool:
        """Check if index i is a pivot high"""
        try:
            for j in range(i - lookback, i + lookback + 1):
                if j != i and 0 <= j < len(highs) and highs[j] > highs[i]:
                    return False
            return True
        except:
            return False
    
    def _is_pivot_low(self, i: int, lows, lookback: int) -> bool:
        """Check if index i is a pivot low"""
        try:
            for j in range(i - lookback, i + lookback + 1):
                if j != i and 0 <= j < len(lows) and lows[j] < lows[i]:
                    return False
            return True
        except:
            return False
    
    def _validate_sr_level(self, price: float, df: pd.DataFrame, is_resistance: bool,
                          min_touches: int, tolerance_pct: float) -> Optional[SupportResistanceLevel]:
        """Validate support/resistance level"""
        try:
            tolerance = price * (tolerance_pct / 100)
            touches = 0
            
            # Check how many times price touched this level
            prices_to_check = df['High'].astype(float).values if is_resistance else df['Low'].astype(float).values
            
            for test_price in prices_to_check:
                if abs(test_price - price) <= tolerance:
                    touches += 1
            
            if touches >= min_touches:
                strength = min(touches / 10.0, 1.0)  # Normalize to 0-1
                return SupportResistanceLevel(
                    price=price,
                    strength=strength,
                    touches=touches,
                    is_support=not is_resistance
                )
        except Exception as e:
            logger.error(f"Error validating S/R level: {e}")
        
        return None
    
    def _merge_similar_levels(self, levels: List[SupportResistanceLevel], 
                            tolerance_pct: float) -> List[SupportResistanceLevel]:
        """Merge similar levels"""
        if not levels:
            return []
        
        merged = []
        levels.sort(key=lambda x: x.price)
        
        for level in levels:
            merged_with_existing = False
            
            for i, existing_level in enumerate(merged):
                if (existing_level.is_support == level.is_support and 
                    abs(existing_level.price - level.price) / existing_level.price * 100 < tolerance_pct):
                    # Merge levels
                    total_touches = existing_level.touches + level.touches
                    avg_price = (existing_level.price * existing_level.touches + 
                               level.price * level.touches) / total_touches
                    
                    merged[i] = SupportResistanceLevel(
                        price=avg_price,
                        strength=max(existing_level.strength, level.strength),
                        touches=total_touches,
                        is_support=level.is_support
                    )
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                merged.append(level)
        
        return merged

class EnhancedIndicatorCalculator:
    """Enhanced technical indicator calculations"""
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                                std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calculate_rsi_simple(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using simple method"""
        try:
            delta = prices.diff()
            gains = delta.where(delta > 0, 0.0)
            losses = -delta.where(delta < 0, 0.0)
            
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            rs = avg_gains / avg_losses.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    @staticmethod
    def calculate_macd_simple(prices: pd.Series, fast: int = 12, slow: int = 26, 
                            signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD using simple EMA"""
        try:
            ema_fast = EnhancedIndicatorCalculator.calculate_ema(prices, fast)
            ema_slow = EnhancedIndicatorCalculator.calculate_ema(prices, slow)
            macd_line = ema_fast - ema_slow
            signal_line = EnhancedIndicatorCalculator.calculate_ema(macd_line, signal)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series

class EnhancedChartGenerator(ChartGenerator):
    """Enhanced version of the original ChartGenerator with additional features"""
    
    def __init__(self, save_charts_to_disk: bool = False, charts_dir_str: str = "logs/charts"):
        super().__init__(save_charts_to_disk, charts_dir_str)
        self.pattern_detector = EnhancedPatternDetector()
        self.indicator_calculator = EnhancedIndicatorCalculator()
    
    def _calculate_enhanced_indicators(self, df: pd.DataFrame, tech_metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate enhanced indicators including the original ones"""
        # Start with original indicators
        indicators = super()._calculate_chart_indicators(df, tech_metrics)
        
        try:
            close_prices = df['Close'].astype(float)
            
            # Add enhanced indicators
            if len(close_prices) >= 12:
                indicators['ema_12'] = self.indicator_calculator.calculate_ema(close_prices, 12)
            
            if len(close_prices) >= 26:
                indicators['ema_26'] = self.indicator_calculator.calculate_ema(close_prices, 26)
                
                # MACD
                macd_line, signal_line, histogram = self.indicator_calculator.calculate_macd_simple(close_prices)
                indicators['macd_line'] = macd_line
                indicators['macd_signal'] = signal_line
                indicators['macd_hist'] = histogram
            
            # Enhanced RSI if not already calculated
            if 'rsi' not in indicators and len(close_prices) >= 14:
                indicators['rsi_enhanced'] = self.indicator_calculator.calculate_rsi_simple(close_prices)
            
            # Enhanced support/resistance
            sr_levels = self.pattern_detector.detect_support_resistance_levels(df)
            indicators['support_levels'] = [level for level in sr_levels if level.is_support]
            indicators['resistance_levels'] = [level for level in sr_levels if not level.is_support]
            
        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}")
        
        return indicators
    
    def generate_enhanced_chart_image(self, symbol: str, timeframe: str,
                                    close_prices: Union[List[float], TypingDeque[float]],
                                    high_prices: Union[List[float], TypingDeque[float]],
                                    low_prices: Union[List[float], TypingDeque[float]],
                                    volumes: Union[List[float], TypingDeque[float]],
                                    tech_metrics: Optional[Dict[str, Any]] = None,
                                    lookback_periods: int = 100,
                                    detect_patterns: bool = True) -> Tuple[str, str, List[ChartPattern]]:
        """Generate enhanced chart with pattern detection"""
        
        try:
            # Prepare DataFrame using parent method
            df = self._prepare_and_validate_dataframe(
                close_prices, high_prices, low_prices, volumes, lookback_periods, timeframe
            )
            
            if df is None or len(df) < 15:
                logger.warning(f"Insufficient data for enhanced chart: {len(df) if df is not None else 0} points")
                error_b64 = self._generate_error_chart(f"Insufficient data for {symbol}")
                return error_b64, "", []
            
            # Calculate enhanced indicators
            indicators = self._calculate_enhanced_indicators(df, tech_metrics)
            
            # Detect patterns if requested
            detected_patterns = []
            if detect_patterns:
                detected_patterns = self.pattern_detector.detect_candlestick_patterns(df)
            
            # Create chart using parent method
            chart_b64 = self._create_chart(df, indicators, [], symbol, timeframe, tech_metrics or {})
            
            # Save to disk if enabled
            saved_filepath = ""
            if self._save_charts_to_disk and chart_b64:
                saved_filepath = self._save_enhanced_chart_to_disk(symbol, timeframe, chart_b64, detected_patterns)
            
            return chart_b64, saved_filepath, detected_patterns
            
        except Exception as e:
            logger.error(f"Error generating enhanced chart for {symbol}: {e}")
            error_b64 = self._generate_error_chart(f"Enhanced Chart Error: {str(e)[:70]}")
            return error_b64, "", []
    
    def _save_enhanced_chart_to_disk(self, symbol: str, timeframe: str, chart_base64: str,
                                   patterns: List[ChartPattern]) -> str:
        """Save enhanced chart with pattern metadata"""
        try:
            # Use parent method for basic saving
            filepath = super()._save_chart_to_file_internal(symbol, timeframe, chart_base64)
            
            # Save pattern data if any patterns detected
            if patterns and filepath:
                import json
                from pathlib import Path
                
                base_path = Path(filepath).with_suffix('')
                pattern_file = f"{base_path}_patterns.json"
                
                pattern_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': Path(filepath).stem.split('_')[-1],
                    'total_patterns': len(patterns),
                    'patterns': [
                        {
                            'type': pattern.pattern_type.value,
                            'name': pattern.name,
                            'confidence': pattern.confidence,
                            'description': pattern.description,
                            'bullish_signal': pattern.bullish_signal,
                            'price_level': pattern.price_level
                        }
                        for pattern in patterns
                    ]
                }
                
                with open(pattern_file, 'w') as f:
                    json.dump(pattern_data, f, indent=2)
                
                logger.info(f"Pattern data saved: {pattern_file}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving enhanced chart data: {e}")
            return ""

# Enhanced wrapper function for visual agent integration
def generate_enhanced_chart_for_visual_agent(
    symbol: str, timeframe: str,
    close_buf: Union[TypingDeque[float], List[float]],
    high_buf: Union[TypingDeque[float], List[float]],
    low_buf: Union[TypingDeque[float], List[float]],
    vol_buf: Union[TypingDeque[float], List[float]],
    tech_metrics: Optional[Dict[str, Any]] = None,
    lookback_periods: int = 100,
    save_chart: bool = True,
    detect_patterns: bool = True
) -> Tuple[str, str, List[ChartPattern]]:
    """Enhanced wrapper function with pattern detection"""
    try:
        generator = EnhancedChartGenerator(save_charts_to_disk=save_chart)
        
        return generator.generate_enhanced_chart_image(
            symbol=symbol, timeframe=timeframe,
            close_prices=close_buf, high_prices=high_buf,
            low_prices=low_buf, volumes=vol_buf,
            tech_metrics=tech_metrics or {},
            lookback_periods=lookback_periods,
            detect_patterns=detect_patterns
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced chart wrapper: {e}")
        try:
            error_b64 = EnhancedChartGenerator()._generate_error_chart(f"Enhanced Wrapper Error: {str(e)[:60]}")
            return error_b64, "", []
        except Exception:
            fallback_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            return fallback_b64, "", []

# Utility functions for integration
def analyze_chart_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze chart patterns and return summary"""
    detector = EnhancedPatternDetector()
    patterns = detector.detect_candlestick_patterns(df)
    sr_levels = detector.detect_support_resistance_levels(df)
    
    return {
        'patterns_detected': len(patterns),
        'bullish_patterns': len([p for p in patterns if p.bullish_signal is True]),
        'bearish_patterns': len([p for p in patterns if p.bullish_signal is False]),
        'neutral_patterns': len([p for p in patterns if p.bullish_signal is None]),
        'support_levels': len([l for l in sr_levels if l.is_support]),
        'resistance_levels': len([l for l in sr_levels if not l.is_support]),
        'patterns': patterns,
        'sr_levels': sr_levels
    }

def get_pattern_summary(patterns: List[ChartPattern]) -> str:
    """Generate a text summary of detected patterns"""
    if not patterns:
        return "No significant patterns detected"
    
    bullish = len([p for p in patterns if p.bullish_signal is True])
    bearish = len([p for p in patterns if p.bullish_signal is False])
    neutral = len([p for p in patterns if p.bullish_signal is None])
    
    summary = f"Detected {len(patterns)} patterns: "
    if bullish > 0:
        summary += f"{bullish} bullish, "
    if bearish > 0:
        summary += f"{bearish} bearish, "
    if neutral > 0:
        summary += f"{neutral} neutral"
    
    # Add strongest pattern
    strongest = max(patterns, key=lambda x: x.confidence)
    summary += f". Strongest: {strongest.name} ({strongest.confidence:.1%})"
    
    return summary.rstrip(", ")
