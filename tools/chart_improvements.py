# tools/chart_improvements.py
"""
Improvements and additions to the existing chart generator.
This module provides enhanced functionality without replacing the original.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns that can be detected"""
    CANDLESTICK = "candlestick"
    CHART_FORMATION = "chart_formation"
    TECHNICAL_SIGNAL = "technical_signal"

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

class PatternDetector:
    """Enhanced pattern detection for trading charts"""
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect basic candlestick patterns"""
        if len(df) < 3:
            return []
        
        patterns = []
        
        try:
            open_prices = df['Open'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            close_prices = df['Close'].values
            
            for i in range(2, len(df)):
                patterns.extend(self._check_doji_pattern(i, open_prices, high_prices, low_prices, close_prices))
                patterns.extend(self._check_hammer_pattern(i, open_prices, high_prices, low_prices, close_prices))
                
                if i > 0:
                    patterns.extend(self._check_engulfing_pattern(i, open_prices, high_prices, low_prices, close_prices))
        
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
        
        return [p for p in patterns if p.confidence >= self.min_confidence]
    
    def _check_doji_pattern(self, i: int, opens, highs, lows, closes) -> List[ChartPattern]:
        """Check for Doji pattern at index i"""
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        body = abs(c - o)
        total_range = h - l
        
        if total_range > 0 and body / total_range < 0.1:
            return [ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Doji",
                confidence=0.8,
                start_index=i,
                end_index=i,
                price_level=c,
                description="Indecision pattern - market uncertainty",
                bullish_signal=None
            )]
        return []
    
    def _check_hammer_pattern(self, i: int, opens, highs, lows, closes) -> List[ChartPattern]:
        """Check for Hammer pattern at index i"""
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        
        if (lower_shadow > body * 2 and upper_shadow < body * 0.5 and 
            i > 10 and c < np.mean(closes[i-5:i])):
            return [ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Hammer",
                confidence=0.75,
                start_index=i,
                end_index=i,
                price_level=c,
                description="Potential bullish reversal signal",
                bullish_signal=True
            )]
        return []
    
    def _check_engulfing_pattern(self, i: int, opens, highs, lows, closes) -> List[ChartPattern]:
        """Check for Engulfing patterns at index i"""
        patterns = []
        
        # Current candle
        o, c = opens[i], closes[i]
        body = abs(c - o)
        
        # Previous candle
        prev_o, prev_c = opens[i-1], closes[i-1]
        prev_body = abs(prev_c - prev_o)
        
        # Bullish Engulfing
        if (c > o and prev_c < prev_o and  # Current bullish, previous bearish
            c > prev_o and o < prev_c and body > prev_body * 1.1):
            patterns.append(ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Bullish Engulfing",
                confidence=0.85,
                start_index=i-1,
                end_index=i,
                price_level=c,
                description="Strong bullish reversal pattern",
                bullish_signal=True
            ))
        
        # Bearish Engulfing
        elif (c < o and prev_c > prev_o and  # Current bearish, previous bullish
              c < prev_o and o > prev_c and body > prev_body * 1.1):
            patterns.append(ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Bearish Engulfing",
                confidence=0.85,
                start_index=i-1,
                end_index=i,
                price_level=c,
                description="Strong bearish reversal pattern",
                bullish_signal=False
            ))
        
        return patterns
    
    def detect_support_resistance(self, df: pd.DataFrame, lookback: int = 20, 
                                 min_touches: int = 2, tolerance_pct: float = 0.5) -> List[SupportResistanceLevel]:
        """Detect support and resistance levels using pivot points"""
        if len(df) < lookback * 2:
            return []
        
        levels = []
        high_prices = df['High'].values
        low_prices = df['Low'].values
        
        try:
            # Find pivot points
            for i in range(lookback, len(df) - lookback):
                # Pivot High (Resistance)
                if self._is_pivot_high(i, high_prices, lookback):
                    level = self._validate_level(high_prices[i], df, i, False, min_touches, tolerance_pct)
                    if level:
                        levels.append(level)
                
                # Pivot Low (Support)
                if self._is_pivot_low(i, low_prices, lookback):
                    level = self._validate_level(low_prices[i], df, i, True, min_touches, tolerance_pct)
                    if level:
                        levels.append(level)
            
            # Merge similar levels
            levels = self._merge_levels(levels, tolerance_pct)
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
        
        return levels
    
    def _is_pivot_high(self, i: int, highs, lookback: int) -> bool:
        """Check if index i is a pivot high"""
        return all(highs[i] >= highs[j] for j in range(i - lookback, i + lookback + 1) if j != i)
    
    def _is_pivot_low(self, i: int, lows, lookback: int) -> bool:
        """Check if index i is a pivot low"""
        return all(lows[i] <= lows[j] for j in range(i - lookback, i + lookback + 1) if j != i)
    
    def _validate_level(self, price: float, df: pd.DataFrame, index: int, is_support: bool,
                       min_touches: int, tolerance_pct: float) -> Optional[SupportResistanceLevel]:
        """Validate and calculate strength of a level"""
        tolerance = price * (tolerance_pct / 100)
        touches = 0
        
        prices_to_check = df['Low'].values if is_support else df['High'].values
        
        for test_price in prices_to_check:
            if abs(test_price - price) <= tolerance:
                touches += 1
        
        if touches >= min_touches:
            strength = min(touches / 10.0, 1.0)
            return SupportResistanceLevel(
                price=price,
                strength=strength,
                touches=touches,
                is_support=is_support
            )
        return None
    
    def _merge_levels(self, levels: List[SupportResistanceLevel], tolerance_pct: float) -> List[SupportResistanceLevel]:
        """Merge similar support/resistance levels"""
        if not levels:
            return []
        
        merged = []
        levels.sort(key=lambda x: x.price)
        
        for level in levels:
            if not merged:
                merged.append(level)
                continue
            
            last_level = merged[-1]
            price_diff_pct = abs(level.price - last_level.price) / last_level.price * 100
            
            if (price_diff_pct < tolerance_pct and level.is_support == last_level.is_support):
                # Merge levels
                avg_price = (level.price * level.touches + last_level.price * last_level.touches) / (level.touches + last_level.touches)
                merged[-1] = SupportResistanceLevel(
                    price=avg_price,
                    strength=max(level.strength, last_level.strength),
                    touches=level.touches + last_level.touches,
                    is_support=level.is_support
                )
            else:
                merged.append(level)
        
        return merged

class IndicatorCalculator:
    """Enhanced indicator calculations"""
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using simple method"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = IndicatorCalculator.calculate_ema(prices, fast)
        ema_slow = IndicatorCalculator.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = IndicatorCalculator.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

class ChartEnhancer:
    """Enhances existing charts with additional features"""
    
    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.indicator_calculator = IndicatorCalculator()
    
    def analyze_chart_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive analysis of chart data"""
        analysis = {
            'patterns': [],
            'support_resistance': [],
            'indicators': {},
            'trend_analysis': {},
            'quality_metrics': {}
        }
        
        try:
            # Detect patterns
            analysis['patterns'] = self.pattern_detector.detect_candlestick_patterns(df)
            
            # Detect support/resistance
            analysis['support_resistance'] = self.pattern_detector.detect_support_resistance(df)
            
            # Calculate additional indicators
            if len(df) >= 20:
                close_prices = df['Close']
                
                # EMAs
                analysis['indicators']['ema_12'] = self.indicator_calculator.calculate_ema(close_prices, 12)
                analysis['indicators']['ema_26'] = self.indicator_calculator.calculate_ema(close_prices, 26)
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = self.indicator_calculator.calculate_bollinger_bands(close_prices)
                analysis['indicators']['bb_upper'] = bb_upper
                analysis['indicators']['bb_middle'] = bb_middle
                analysis['indicators']['bb_lower'] = bb_lower
                
                # RSI
                if len(df) >= 14:
                    analysis['indicators']['rsi'] = self.indicator_calculator.calculate_rsi(close_prices)
                
                # MACD
                if len(df) >= 26:
                    macd, signal, hist = self.indicator_calculator.calculate_macd(close_prices)
                    analysis['indicators']['macd_line'] = macd
                    analysis['indicators']['macd_signal'] = signal
                    analysis['indicators']['macd_hist'] = hist
            
            # Trend analysis
            analysis['trend_analysis'] = self._analyze_trend(df)
            
            # Quality metrics
            analysis['quality_metrics'] = self._calculate_quality_metrics(df, analysis)
            
        except Exception as e:
            logger.error(f"Error in chart analysis: {e}")
        
        return analysis
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall trend direction and strength"""
        if len(df) < 10:
            return {'direction': 'unknown', 'strength': 0.0}
        
        close_prices = df['Close'].values
        
        # Linear regression for trend
        x = np.arange(len(close_prices))
        slope, intercept = np.polyfit(x, close_prices, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((close_prices - y_pred) ** 2)
        ss_tot = np.sum((close_prices - np.mean(close_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine direction
        price_change_pct = (slope / np.mean(close_prices)) * 100 if np.mean(close_prices) > 0 else 0
        
        if price_change_pct > 0.1:
            direction = 'bullish'
        elif price_change_pct < -0.1:
            direction = 'bearish'
        else:
            direction = 'sideways'
        
        return {
            'direction': direction,
            'strength': r_squared,
            'slope_pct': price_change_pct,
            'confidence': 'high' if r_squared > 0.7 else 'medium' if r_squared > 0.4 else 'low'
        }
    
    def _calculate_quality_metrics(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate chart quality and reliability metrics"""
        metrics = {
            'data_points': len(df),
            'volatility': 0.0,
            'pattern_count': len(analysis.get('patterns', [])),
            'sr_level_count': len(analysis.get('support_resistance', [])),
            'reliability_score': 0.0
        }
        
        try:
            if len(df) > 1:
                # Calculate volatility using ATR-like measure
                high_low = df['High'] - df['Low']
                high_close = abs(df['High'] - df['Close'].shift(1))
                low_close = abs(df['Low'] - df['Close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                metrics['volatility'] = true_range.mean()
                
                # Calculate reliability score
                data_score = min(len(df) / 100, 1.0)  # More data = higher score
                pattern_score = min(len(analysis.get('patterns', [])) / 5, 1.0)  # More patterns = higher score
                trend_score = analysis.get('trend_analysis', {}).get('strength', 0.0)
                
                metrics['reliability_score'] = (data_score + pattern_score + trend_score) / 3
        
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
        
        return metrics
    
    def generate_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a text summary of the chart analysis"""
        try:
            patterns = analysis.get('patterns', [])
            trend = analysis.get('trend_analysis', {})
            quality = analysis.get('quality_metrics', {})
            
            summary_parts = []
            
            # Trend summary
            direction = trend.get('direction', 'unknown')
            confidence = trend.get('confidence', 'low')
            summary_parts.append(f"Trend: {direction.upper()} ({confidence} confidence)")
            
            # Pattern summary
            if patterns:
                bullish_patterns = [p for p in patterns if p.bullish_signal is True]
                bearish_patterns = [p for p in patterns if p.bullish_signal is False]
                neutral_patterns = [p for p in patterns if p.bullish_signal is None]
                
                summary_parts.append(f"Patterns detected: {len(patterns)} total")
                if bullish_patterns:
                    summary_parts.append(f"  - {len(bullish_patterns)} bullish signals")
                if bearish_patterns:
                    summary_parts.append(f"  - {len(bearish_patterns)} bearish signals")
                if neutral_patterns:
                    summary_parts.append(f"  - {len(neutral_patterns)} neutral patterns")
            else:
                summary_parts.append("No significant patterns detected")
            
            # Quality summary
            reliability = quality.get('reliability_score', 0.0)
            data_points = quality.get('data_points', 0)
            summary_parts.append(f"Analysis reliability: {reliability:.1%} ({data_points} data points)")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return "Error generating summary"


# Integration functions for the existing chart generator
def enhance_existing_chart_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Enhance existing chart with additional analysis"""
    enhancer = ChartEnhancer()
    return enhancer.analyze_chart_data(df)

def get_additional_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Get additional technical indicators for existing charts"""
    calculator = IndicatorCalculator()
    indicators = {}
    
    if len(df) >= 20:
        close_prices = df['Close']
        
        # EMAs
        indicators['ema_12'] = calculator.calculate_ema(close_prices, 12)
        indicators['ema_26'] = calculator.calculate_ema(close_prices, 26)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculator.calculate_bollinger_bands(close_prices)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # RSI
        if len(df) >= 14:
            indicators['rsi'] = calculator.calculate_rsi(close_prices)
        
        # MACD
        if len(df) >= 26:
            macd, signal, hist = calculator.calculate_macd(close_prices)
            indicators['macd_line'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_hist'] = hist
    
    return indicators

def detect_chart_patterns(df: pd.DataFrame) -> List[ChartPattern]:
    """Detect patterns in chart data"""
    detector = PatternDetector()
    return detector.detect_candlestick_patterns(df)

def find_support_resistance_levels(df: pd.DataFrame) -> List[SupportResistanceLevel]:
    """Find support and resistance levels"""
    detector = PatternDetector()
    return detector.detect_support_resistance(df)
