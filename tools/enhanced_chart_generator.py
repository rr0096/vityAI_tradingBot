# tools/enhanced_chart_generator.py
"""
Enhanced Chart Generator with advanced technical analysis and pattern detection.
Extends the base chart generator with configurable indicators, pattern recognition,
and improved support/resistance detection.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Union, Any, Deque as TypingDeque
from collections import deque
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Constants
FALLBACK_CHART_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

class PatternType(Enum):
    """Types of patterns that can be detected"""
    CANDLESTICK = "candlestick"
    CHART_FORMATION = "chart_formation"
    TECHNICAL_SIGNAL = "technical_signal"

class TrendDirection(Enum):
    """Trend direction classification"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNCERTAIN = "uncertain"

@dataclass
class ChartConfig:
    """Configuration for chart generation and indicators"""
    # SMA periods
    sma_short_period: int = 20
    sma_long_period: int = 50
    ema_short_period: int = 12
    ema_long_period: int = 26
    
    # Technical indicator periods
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    
    # Support/Resistance detection
    sr_lookback: int = 20
    sr_min_touches: int = 2
    sr_tolerance_pct: float = 0.5
    
    # Pattern detection
    enable_candlestick_patterns: bool = True
    enable_chart_patterns: bool = True
    min_pattern_strength: float = 0.6
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000

@dataclass
class DetectedPattern:
    """Represents a detected pattern on the chart"""
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
    last_touch_index: int
    is_support: bool

class EnhancedChartGenerator:
    """Enhanced chart generator with advanced pattern detection and indicators"""
    
    def __init__(self, config: Optional[ChartConfig] = None, save_charts_to_disk: bool = False, 
                 charts_dir_str: str = "logs/charts"):
        self.config = config or ChartConfig()
        self._save_charts_to_disk = save_charts_to_disk
        self._charts_dir = Path(charts_dir_str)
        
        if self._save_charts_to_disk:
            self._charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'up': '#26A69A',
            'down': '#EF5350',
            'neutral': '#787B86',
            'text': '#E0E0E0',
            'grid': '#404040',
            'bg': '#131722',
            'axes_bg': '#1A1E29',
            'volume_up': '#26A69A80',
            'volume_down': '#EF535080',
            'sma_short': '#FFA726',
            'sma_long': '#66BB6A',
            'bb': '#9C27B0',
            'rsi': '#FF5722',
            'macd': '#2196F3'
        }
        
        # Cache for indicators
        self._indicator_cache = {} if self.config.enable_caching else None
        
        # Initialize matplotlib style
        self._setup_mplfinance_style()
    
    def _setup_mplfinance_style(self):
        """Setup custom mplfinance style"""
        try:
            self.mpf_marketcolors = mpf.make_marketcolors(
                up=self.colors['up'], down=self.colors['down'], edge='inherit',
                wick={'up': self.colors['up'], 'down': self.colors['down']},
                volume={'up': self.colors['up'], 'down': self.colors['down']}, 
                alpha=0.9
            )
            
            self.style_rc_params = {
                'font.size': 8,
                'axes.labelsize': 7, 'axes.titlesize': 10,
                'axes.grid': True, 'grid.alpha': 0.2, 'grid.color': self.colors['grid'],
                'figure.facecolor': self.colors['bg'], 'axes.facecolor': self.colors['axes_bg'],
                'xtick.color': self.colors['text'], 'ytick.color': self.colors['text'],
                'axes.labelcolor': self.colors['text'], 'axes.titlecolor': self.colors['text'],
                'text.color': self.colors['text'],
                'figure.dpi': 100
            }
            
            self.custom_mpf_style = mpf.make_mpf_style(
                base_mpf_style='nightclouds',
                marketcolors=self.mpf_marketcolors,
                rc=self.style_rc_params,
                y_on_right=False
            )
        except Exception as e:
            logger.warning(f"Error creating custom style: {e}. Falling back to default.")
            self.custom_mpf_style = 'nightclouds'
    
    def _calculate_enhanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced set of technical indicators"""
        indicators = {}
        
        if df.empty or len(df) < 5:
            return indicators
        
        close_prices = df['Close'].astype(float)
        high_prices = df['High'].astype(float)
        low_prices = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
        
        try:
            # Simple Moving Averages
            if len(close_prices) >= self.config.sma_short_period:
                indicators['sma_short'] = close_prices.rolling(
                    window=self.config.sma_short_period, min_periods=1
                ).mean()
            
            if len(close_prices) >= self.config.sma_long_period:
                indicators['sma_long'] = close_prices.rolling(
                    window=self.config.sma_long_period, min_periods=1
                ).mean()
            
            # Exponential Moving Averages
            if len(close_prices) >= self.config.ema_short_period:
                indicators['ema_short'] = close_prices.ewm(
                    span=self.config.ema_short_period, adjust=False
                ).mean()
            
            if len(close_prices) >= self.config.ema_long_period:
                indicators['ema_long'] = close_prices.ewm(
                    span=self.config.ema_long_period, adjust=False
                ).mean()
            
            # Advanced indicators with TA-Lib if available
            try:
                import talib
                
                # Bollinger Bands
                if len(close_prices) >= self.config.bb_period:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(
                        close_prices.values, 
                        timeperiod=self.config.bb_period,
                        nbdevup=self.config.bb_std, 
                        nbdevdn=self.config.bb_std, 
                        matype=0
                    )
                    indicators['bb_upper'] = pd.Series(bb_upper, index=df.index)
                    indicators['bb_middle'] = pd.Series(bb_middle, index=df.index)
                    indicators['bb_lower'] = pd.Series(bb_lower, index=df.index)
                
                # RSI
                if len(close_prices) >= self.config.rsi_period + 1:
                    rsi_values = talib.RSI(close_prices.values, timeperiod=self.config.rsi_period)
                    indicators['rsi'] = pd.Series(rsi_values, index=df.index)
                
                # MACD
                if len(close_prices) >= self.config.macd_slow + self.config.macd_signal:
                    macd_line, macd_signal, macd_hist = talib.MACD(
                        close_prices.values,
                        fastperiod=self.config.macd_fast,
                        slowperiod=self.config.macd_slow,
                        signalperiod=self.config.macd_signal
                    )
                    indicators['macd_line'] = pd.Series(macd_line, index=df.index)
                    indicators['macd_signal'] = pd.Series(macd_signal, index=df.index)
                    indicators['macd_hist'] = pd.Series(macd_hist, index=df.index)
                
                # ADX for trend strength
                if len(df) >= self.config.adx_period * 2:
                    adx_values = talib.ADX(
                        high_prices.values, low_prices.values, close_prices.values,
                        timeperiod=self.config.adx_period
                    )
                    indicators['adx'] = pd.Series(adx_values, index=df.index)
                    
            except ImportError:
                logger.warning("TA-Lib not available. Using basic indicator calculations.")
                # Fallback to basic calculations
                if len(close_prices) >= self.config.bb_period:
                    sma = close_prices.rolling(window=self.config.bb_period).mean()
                    std = close_prices.rolling(window=self.config.bb_period).std()
                    indicators['bb_upper'] = sma + (std * self.config.bb_std)
                    indicators['bb_middle'] = sma
                    indicators['bb_lower'] = sma - (std * self.config.bb_std)
            
            # Support and Resistance Levels
            sr_levels = self._detect_support_resistance(df)
            if sr_levels:
                indicators['support_levels'] = [level for level in sr_levels if level.is_support]
                indicators['resistance_levels'] = [level for level in sr_levels if not level.is_support]
            
        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}", exc_info=True)
        
        return indicators
    
    def _detect_support_resistance(self, df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Advanced support and resistance detection using pivot points"""
        if len(df) < self.config.sr_lookback * 2:
            return []
        
        levels = []
        high_prices = df['High'].values
        low_prices = df['Low'].values
        close_prices = df['Close'].values
        
        try:
            # Find pivot points
            for i in range(self.config.sr_lookback, len(df) - self.config.sr_lookback):
                # Pivot High (Resistance)
                if all(high_prices[i] >= high_prices[j] for j in range(i - self.config.sr_lookback, i + self.config.sr_lookback + 1) if j != i):
                    level = self._validate_sr_level(high_prices[i], df, i, False)
                    if level:
                        levels.append(level)
                
                # Pivot Low (Support)
                if all(low_prices[i] <= low_prices[j] for j in range(i - self.config.sr_lookback, i + self.config.sr_lookback + 1) if j != i):
                    level = self._validate_sr_level(low_prices[i], df, i, True)
                    if level:
                        levels.append(level)
            
            # Merge similar levels and calculate strength
            levels = self._merge_sr_levels(levels)
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
        
        return levels
    
    def _validate_sr_level(self, price: float, df: pd.DataFrame, index: int, is_support: bool) -> Optional[SupportResistanceLevel]:
        """Validate and calculate strength of a support/resistance level"""
        tolerance = price * (self.config.sr_tolerance_pct / 100)
        touches = 0
        last_touch = index
        
        # Count touches within tolerance
        prices_to_check = df['Low'].values if is_support else df['High'].values
        
        for i, test_price in enumerate(prices_to_check):
            if abs(test_price - price) <= tolerance:
                touches += 1
                last_touch = i
        
        if touches >= self.config.sr_min_touches:
            strength = min(touches / 10.0, 1.0)  # Normalize strength
            return SupportResistanceLevel(
                price=price,
                strength=strength,
                touches=touches,
                last_touch_index=last_touch,
                is_support=is_support
            )
        return None
    
    def _merge_sr_levels(self, levels: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
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
            
            if (price_diff_pct < self.config.sr_tolerance_pct and 
                level.is_support == last_level.is_support):
                # Merge levels
                avg_price = (level.price * level.touches + last_level.price * last_level.touches) / (level.touches + last_level.touches)
                merged[-1] = SupportResistanceLevel(
                    price=avg_price,
                    strength=max(level.strength, last_level.strength),
                    touches=level.touches + last_level.touches,
                    last_touch_index=max(level.last_touch_index, last_level.last_touch_index),
                    is_support=level.is_support
                )
            else:
                merged.append(level)
        
        return merged
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect candlestick patterns"""
        if not self.config.enable_candlestick_patterns or len(df) < 3:
            return []
        
        patterns = []
        
        try:
            open_prices = df['Open'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            close_prices = df['Close'].values
            
            for i in range(2, len(df)):
                # Current candle
                o, h, l, c = open_prices[i], high_prices[i], low_prices[i], close_prices[i]
                prev_o, prev_h, prev_l, prev_c = open_prices[i-1], high_prices[i-1], low_prices[i-1], close_prices[i-1]
                
                body = abs(c - o)
                prev_body = abs(prev_c - prev_o)
                total_range = h - l
                
                # Doji pattern
                if body / total_range < 0.1 and total_range > 0:
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.CANDLESTICK,
                        name="Doji",
                        confidence=0.8,
                        start_index=i,
                        end_index=i,
                        price_level=c,
                        description="Indecision pattern - market uncertainty",
                        bullish_signal=None
                    ))
                
                # Hammer pattern
                lower_shadow = (min(o, c) - l)
                upper_shadow = (h - max(o, c))
                if (lower_shadow > body * 2 and upper_shadow < body * 0.5 and 
                    i > 10 and c < close_prices[i-5:i].mean()):
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.CANDLESTICK,
                        name="Hammer",
                        confidence=0.75,
                        start_index=i,
                        end_index=i,
                        price_level=c,
                        description="Potential bullish reversal signal",
                        bullish_signal=True
                    ))
                
                # Engulfing patterns
                if i > 0:
                    if (c > o and prev_c < prev_o and  # Current bullish, previous bearish
                        c > prev_o and o < prev_c and body > prev_body * 1.1):
                        patterns.append(DetectedPattern(
                            pattern_type=PatternType.CANDLESTICK,
                            name="Bullish Engulfing",
                            confidence=0.85,
                            start_index=i-1,
                            end_index=i,
                            price_level=c,
                            description="Strong bullish reversal pattern",
                            bullish_signal=True
                        ))
                    
                    elif (c < o and prev_c > prev_o and  # Current bearish, previous bullish
                          c < prev_o and o > prev_c and body > prev_body * 1.1):
                        patterns.append(DetectedPattern(
                            pattern_type=PatternType.CANDLESTICK,
                            name="Bearish Engulfing",
                            confidence=0.85,
                            start_index=i-1,
                            end_index=i,
                            price_level=c,
                            description="Strong bearish reversal pattern",
                            bullish_signal=False
                        ))
                
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
        
        return patterns
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect chart formation patterns"""
        if not self.config.enable_chart_patterns or len(df) < 20:
            return []
        
        patterns = []
        close_prices = df['Close'].values
        
        try:
            # Double Top/Bottom detection
            patterns.extend(self._detect_double_patterns(df))
            
            # Triangle patterns
            patterns.extend(self._detect_triangle_patterns(df))
            
            # Head and Shoulders
            patterns.extend(self._detect_head_shoulders(df))
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
        
        return patterns
    
    def _detect_double_patterns(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect double top and double bottom patterns"""
        patterns = []
        high_prices = df['High'].values
        low_prices = df['Low'].values
        
        lookback = min(20, len(df) // 4)
        
        try:
            # Double Top
            for i in range(lookback, len(df) - lookback):
                if i < len(high_prices) - lookback:
                    local_highs = []
                    for j in range(i - lookback, i + lookback):
                        if j < len(high_prices) and high_prices[j] == max(high_prices[max(0, j-2):min(len(high_prices), j+3)]):
                            local_highs.append((j, high_prices[j]))
                    
                    if len(local_highs) >= 2:
                        first_peak, second_peak = local_highs[-2], local_highs[-1]
                        if abs(first_peak[1] - second_peak[1]) / first_peak[1] < 0.02:  # Within 2%
                            patterns.append(DetectedPattern(
                                pattern_type=PatternType.CHART_FORMATION,
                                name="Double Top",
                                confidence=0.7,
                                start_index=first_peak[0],
                                end_index=second_peak[0],
                                price_level=max(first_peak[1], second_peak[1]),
                                description="Bearish reversal pattern",
                                bullish_signal=False
                            ))
            
            # Double Bottom
            for i in range(lookback, len(df) - lookback):
                if i < len(low_prices) - lookback:
                    local_lows = []
                    for j in range(i - lookback, i + lookback):
                        if j < len(low_prices) and low_prices[j] == min(low_prices[max(0, j-2):min(len(low_prices), j+3)]):
                            local_lows.append((j, low_prices[j]))
                    
                    if len(local_lows) >= 2:
                        first_trough, second_trough = local_lows[-2], local_lows[-1]
                        if abs(first_trough[1] - second_trough[1]) / first_trough[1] < 0.02:
                            patterns.append(DetectedPattern(
                                pattern_type=PatternType.CHART_FORMATION,
                                name="Double Bottom",
                                confidence=0.7,
                                start_index=first_trough[0],
                                end_index=second_trough[0],
                                price_level=min(first_trough[1], second_trough[1]),
                                description="Bullish reversal pattern",
                                bullish_signal=True
                            ))
        
        except Exception as e:
            logger.error(f"Error detecting double patterns: {e}")
        
        return patterns
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect triangle patterns (ascending, descending, symmetric)"""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        try:
            close_prices = df['Close'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            
            # Look for triangle patterns in recent data
            window = min(30, len(df) // 2)
            recent_highs = high_prices[-window:]
            recent_lows = low_prices[-window:]
            
            # Calculate trend lines
            x = np.arange(len(recent_highs))
            
            # Upper trend line (resistance)
            upper_slope = np.polyfit(x, recent_highs, 1)[0]
            
            # Lower trend line (support)  
            lower_slope = np.polyfit(x, recent_lows, 1)[0]
            
            # Classify triangle type
            if abs(upper_slope) < 0.001 and lower_slope > 0.001:  # Ascending triangle
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.CHART_FORMATION,
                    name="Ascending Triangle",
                    confidence=0.6,
                    start_index=len(df) - window,
                    end_index=len(df) - 1,
                    price_level=recent_highs[-1],
                    description="Bullish continuation pattern",
                    bullish_signal=True
                ))
            elif abs(lower_slope) < 0.001 and upper_slope < -0.001:  # Descending triangle
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.CHART_FORMATION,
                    name="Descending Triangle",
                    confidence=0.6,
                    start_index=len(df) - window,
                    end_index=len(df) - 1,
                    price_level=recent_lows[-1],
                    description="Bearish continuation pattern",
                    bullish_signal=False
                ))
            elif upper_slope < -0.001 and lower_slope > 0.001:  # Symmetric triangle
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.CHART_FORMATION,
                    name="Symmetric Triangle",
                    confidence=0.5,
                    start_index=len(df) - window,
                    end_index=len(df) - 1,
                    price_level=(recent_highs[-1] + recent_lows[-1]) / 2,
                    description="Neutral pattern - awaiting breakout",
                    bullish_signal=None
                ))
        
        except Exception as e:
            logger.error(f"Error detecting triangle patterns: {e}")
        
        return patterns
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[DetectedPattern]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        if len(df) < 40:
            return patterns
        
        try:
            high_prices = df['High'].values
            low_prices = df['Low'].values
            
            # Find potential head and shoulders in recent data
            window = min(40, len(df) // 2)
            recent_highs = high_prices[-window:]
            recent_lows = low_prices[-window:]
            
            # Find three peaks
            peaks = []
            for i in range(2, len(recent_highs) - 2):
                if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1] and
                    recent_highs[i] > recent_highs[i-2] and recent_highs[i] > recent_highs[i+2]):
                    peaks.append((i, recent_highs[i]))
            
            if len(peaks) >= 3:
                # Check for head and shoulders pattern
                left_shoulder, head, right_shoulder = peaks[-3], peaks[-2], peaks[-1]
                
                # Head should be higher than shoulders
                if (head[1] > left_shoulder[1] and head[1] > right_shoulder[1] and
                    abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05):
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.CHART_FORMATION,
                        name="Head and Shoulders",
                        confidence=0.8,
                        start_index=len(df) - window + left_shoulder[0],
                        end_index=len(df) - window + right_shoulder[0],
                        price_level=head[1],
                        description="Bearish reversal pattern",
                        bullish_signal=False
                    ))
            
            # Inverse head and shoulders
            troughs = []
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1] and
                    recent_lows[i] < recent_lows[i-2] and recent_lows[i] < recent_lows[i+2]):
                    troughs.append((i, recent_lows[i]))
            
            if len(troughs) >= 3:
                left_shoulder, head, right_shoulder = troughs[-3], troughs[-2], troughs[-1]
                
                if (head[1] < left_shoulder[1] and head[1] < right_shoulder[1] and
                    abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05):
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.CHART_FORMATION,
                        name="Inverse Head and Shoulders",
                        confidence=0.8,
                        start_index=len(df) - window + left_shoulder[0],
                        end_index=len(df) - window + right_shoulder[0],
                        price_level=head[1],
                        description="Bullish reversal pattern",
                        bullish_signal=True
                    ))
        
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
        
        return patterns
    
    def _create_enhanced_chart(self, df: pd.DataFrame, indicators: Dict[str, Any], 
                             patterns: List[DetectedPattern], symbol: str, 
                             timeframe: str, tech_metrics: Dict[str, Any]) -> str:
        """Create enhanced chart with all indicators and patterns"""
        fig = None
        
        try:
            addplots = []
            
            # Add moving averages
            if 'sma_short' in indicators and not indicators['sma_short'].isnull().all():
                addplots.append(mpf.make_addplot(
                    indicators['sma_short'], panel=0, 
                    color=self.colors['sma_short'], width=1.0, alpha=0.8,
                    linestyle='-'
                ))
            
            if 'sma_long' in indicators and not indicators['sma_long'].isnull().all():
                addplots.append(mpf.make_addplot(
                    indicators['sma_long'], panel=0,
                    color=self.colors['sma_long'], width=1.2, alpha=0.9,
                    linestyle='-'
                ))
            
            # Add EMAs
            if 'ema_short' in indicators and not indicators['ema_short'].isnull().all():
                addplots.append(mpf.make_addplot(
                    indicators['ema_short'], panel=0,
                    color=self.colors['sma_short'], width=0.8, alpha=0.7,
                    linestyle='--'
                ))
            
            # Add Bollinger Bands
            if all(key in indicators for key in ['bb_upper', 'bb_lower']) and \
               not indicators['bb_upper'].isnull().all() and not indicators['bb_lower'].isnull().all():
                addplots.append(mpf.make_addplot(
                    indicators['bb_upper'], panel=0,
                    color=self.colors['bb'], linestyle=':', width=0.6, alpha=0.6
                ))
                addplots.append(mpf.make_addplot(
                    indicators['bb_lower'], panel=0,
                    color=self.colors['bb'], linestyle=':', width=0.6, alpha=0.6
                ))
            
            # Setup panels
            panel_count = 1  # Main price panel
            panel_ratios = [3]  # Main panel gets most space
            
            # RSI panel
            rsi_panel = None
            if 'rsi' in indicators and not indicators['rsi'].isnull().all():
                rsi_panel = panel_count
                addplots.append(mpf.make_addplot(
                    indicators['rsi'], panel=rsi_panel,
                    color=self.colors['rsi'], width=1.0,
                    ylabel='RSI'
                ))
                panel_count += 1
                panel_ratios.append(1)
            
            # MACD panel
            macd_panel = None
            if all(key in indicators for key in ['macd_line', 'macd_signal']) and \
               not indicators['macd_line'].isnull().all():
                macd_panel = panel_count
                addplots.append(mpf.make_addplot(
                    indicators['macd_line'], panel=macd_panel,
                    color=self.colors['macd'], width=1.0,
                    ylabel='MACD'
                ))
                addplots.append(mpf.make_addplot(
                    indicators['macd_signal'], panel=macd_panel,
                    color=self.colors['sma_short'], width=0.8
                ))
                if 'macd_hist' in indicators:
                    addplots.append(mpf.make_addplot(
                        indicators['macd_hist'], panel=macd_panel,
                        type='bar', color=self.colors['neutral'], alpha=0.6
                    ))
                panel_count += 1
                panel_ratios.append(1)
            
            # Create the chart
            fig, axes = mpf.plot(
                df, type='candle', style=self.custom_mpf_style,
                volume=True, addplot=addplots if addplots else None,
                figsize=(12, 8), returnfig=True,
                panel_ratios=tuple(panel_ratios),
                figscale=1.0, tight_layout=True,
                show_nontrading=False,
                datetime_format='%H:%M', xrotation=45
            )
            
            ax_main = axes[0]
            
            # Add Bollinger Bands fill
            if all(key in indicators for key in ['bb_upper', 'bb_lower']):
                ax_main.fill_between(
                    df.index, indicators['bb_upper'], indicators['bb_lower'],
                    alpha=0.1, color=self.colors['bb']
                )
            
            # Add support/resistance levels
            self._add_sr_levels_to_chart(ax_main, indicators, df)
            
            # Add pattern annotations
            self._add_pattern_annotations(ax_main, patterns, df)
            
            # Add RSI levels
            if rsi_panel is not None and rsi_panel < len(axes):
                ax_rsi = axes[rsi_panel]
                ax_rsi.axhline(70, color=self.colors['down'], linestyle=':', alpha=0.7)
                ax_rsi.axhline(30, color=self.colors['up'], linestyle=':', alpha=0.7)
                ax_rsi.fill_between(df.index, 70, 100, alpha=0.1, color=self.colors['down'])
                ax_rsi.fill_between(df.index, 0, 30, alpha=0.1, color=self.colors['up'])
                ax_rsi.set_ylim(0, 100)
            
            # Add MACD zero line
            if macd_panel is not None and macd_panel < len(axes):
                ax_macd = axes[macd_panel]
                ax_macd.axhline(0, color=self.colors['text'], linestyle='-', alpha=0.5)
            
            # Add title and metrics
            self._add_enhanced_title(fig, symbol, timeframe, tech_metrics, df, patterns)
            
            # Add legend
            self._add_enhanced_legend(ax_main, indicators)
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating enhanced chart: {e}", exc_info=True)
            if fig:
                plt.close(fig)
            return self._generate_error_chart(f"Enhanced Chart Error: {str(e)[:70]}")
    
    def _add_sr_levels_to_chart(self, ax: plt.Axes, indicators: Dict[str, Any], df: pd.DataFrame):
        """Add support and resistance levels to chart"""
        try:
            # Support levels
            if 'support_levels' in indicators:
                for level in indicators['support_levels']:
                    ax.axhline(
                        y=level.price, color=self.colors['up'],
                        linestyle=':', alpha=0.6 + level.strength * 0.4,
                        linewidth=1.0 + level.strength
                    )
                    if not df.index.empty:
                        ax.text(
                            df.index[-1], level.price,
                            f' S {level.price:.2f} ({level.touches})',
                            color=self.colors['up'], fontsize=6,
                            va='bottom', ha='right'
                        )
            
            # Resistance levels
            if 'resistance_levels' in indicators:
                for level in indicators['resistance_levels']:
                    ax.axhline(
                        y=level.price, color=self.colors['down'],
                        linestyle=':', alpha=0.6 + level.strength * 0.4,
                        linewidth=1.0 + level.strength
                    )
                    if not df.index.empty:
                        ax.text(
                            df.index[-1], level.price,
                            f' R {level.price:.2f} ({level.touches})',
                            color=self.colors['down'], fontsize=6,
                            va='top', ha='right'
                        )
        except Exception as e:
            logger.warning(f"Error adding S/R levels: {e}")
    
    def _add_pattern_annotations(self, ax: plt.Axes, patterns: List[DetectedPattern], df: pd.DataFrame):
        """Add pattern annotations to chart"""
        try:
            for pattern in patterns:
                if pattern.start_index is not None and pattern.start_index < len(df):
                    x_pos = df.index[pattern.start_index]
                    y_pos = pattern.price_level or df['Close'].iloc[pattern.start_index]
                    
                    # Choose color based on bullish/bearish signal
                    if pattern.bullish_signal is True:
                        color = self.colors['up']
                    elif pattern.bullish_signal is False:
                        color = self.colors['down']
                    else:
                        color = self.colors['neutral']
                    
                    # Add annotation
                    ax.annotate(
                        pattern.name,
                        xy=(x_pos, y_pos),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=6, color='white'
                    )
        except Exception as e:
            logger.warning(f"Error adding pattern annotations: {e}")
    
    def _add_enhanced_title(self, fig: plt.Figure, symbol: str, timeframe: str,
                          tech_metrics: Dict[str, Any], df: pd.DataFrame,
                          patterns: List[DetectedPattern]):
        """Add enhanced title with key metrics and pattern count"""
        try:
            last_price = tech_metrics.get('last_price', df['Close'].iloc[-1] if not df.empty else 'N/A')
            price_str = f"${last_price:.2f}" if isinstance(last_price, (float, int)) else str(last_price)
            
            title = f"{symbol} - {timeframe} | Last: {price_str}"
            
            # Add key indicators
            rsi = tech_metrics.get('rsi')
            if rsi and isinstance(rsi, (float, int)):
                title += f" | RSI: {rsi:.1f}"
            
            # Add pattern count
            if patterns:
                bullish_patterns = sum(1 for p in patterns if p.bullish_signal is True)
                bearish_patterns = sum(1 for p in patterns if p.bullish_signal is False)
                title += f" | Patterns: {len(patterns)} (🟢{bullish_patterns} 🔴{bearish_patterns})"
            
            fig.suptitle(title, fontsize=11, color=self.colors['text'], y=0.98)
            
        except Exception as e:
            logger.warning(f"Error adding enhanced title: {e}")
    
    def _add_enhanced_legend(self, ax: plt.Axes, indicators: Dict[str, Any]):
        """Add enhanced legend with all indicators"""
        try:
            handles = []
            
            # Moving averages
            if 'sma_short' in indicators:
                handles.append(plt.Line2D([], [], color=self.colors['sma_short'], 
                                        linestyle='-', lw=1, label=f'SMA{self.config.sma_short_period}'))
            if 'sma_long' in indicators:
                handles.append(plt.Line2D([], [], color=self.colors['sma_long'], 
                                        linestyle='-', lw=1.2, label=f'SMA{self.config.sma_long_period}'))
            if 'ema_short' in indicators:
                handles.append(plt.Line2D([], [], color=self.colors['sma_short'], 
                                        linestyle='--', lw=0.8, label=f'EMA{self.config.ema_short_period}'))
            
            # Bollinger Bands
            if 'bb_upper' in indicators:
                handles.append(mpatches.Patch(color=self.colors['bb'], alpha=0.3, 
                                            label=f'BB({self.config.bb_period})'))
            
            if handles:
                ax.legend(handles=handles, loc='upper left', fontsize=7, 
                         frameon=True, fancybox=True,
                         facecolor=self.colors['axes_bg'], 
                         edgecolor=self.colors['grid'],
                         labelcolor=self.colors['text'], framealpha=0.8)
                         
        except Exception as e:
            logger.warning(f"Error adding enhanced legend: {e}")
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = None
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=self.style_rc_params.get('figure.dpi', 100),
                       facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            if fig:
                plt.close(fig)
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        finally:
            if buf:
                buf.close()
    
    def _generate_error_chart(self, error_message: str) -> str:
        """Generate error chart as fallback"""
        fig = None
        try:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor=self.colors['bg'])
            ax.set_facecolor(self.colors['axes_bg'])
            ax.text(0.5, 0.5, f"ENHANCED CHART ERROR:\n{error_message[:120]}",
                   color='#FF6B6B', ha='center', va='center', fontsize=10,
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', fc='#3E2C2C', ec='#FF6B6B', alpha=0.9))
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout(pad=0.2)
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating error chart: {e}")
            if fig:
                plt.close(fig)
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    
    def _prepare_dataframe(self, close_prices, high_prices, low_prices, volumes, 
                          lookback_periods: int, timeframe_str: str) -> Optional[pd.DataFrame]:
        """Prepare and validate DataFrame from price data"""
        try:
            # Convert to lists
            cl = list(close_prices)
            hi = list(high_prices)
            lo = list(low_prices)
            vo = list(volumes)
            
            # Get actual lookback
            min_len = min(len(cl), len(hi), len(lo), len(vo))
            actual_lookback = min(lookback_periods, min_len)
            
            if actual_lookback < 10:
                logger.error(f"Insufficient data: {actual_lookback} candles")
                return None
            
            # Slice data
            cl = cl[-actual_lookback:]
            hi = hi[-actual_lookback:]
            lo = lo[-actual_lookback:]
            vo = vo[-actual_lookback:]
            
            # Validate data
            for i in range(len(cl)):
                if not all(isinstance(x, (int, float)) and x > 0 for x in [cl[i], hi[i], lo[i]]) or \
                   not isinstance(vo[i], (int, float)) or vo[i] < 0:
                    logger.error(f"Invalid data at index {i}")
                    return None
                
                # Ensure OHLC consistency
                cl[i] = max(lo[i], min(cl[i], hi[i]))
                hi[i] = max(cl[i], hi[i], lo[i])
                lo[i] = min(cl[i], hi[i], lo[i])
            
            # Create time index
            interval_minutes = self._parse_timeframe_minutes(timeframe_str)
            end_time = datetime.now(timezone.utc)
            time_index = pd.to_datetime([
                end_time - timedelta(minutes=i * interval_minutes)
                for i in range(len(cl) - 1, -1, -1)
            ], utc=True)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': cl,  # Simplified: using close as open
                'High': hi,
                'Low': lo,
                'Close': cl,
                'Volume': vo
            }, index=time_index)
            
            # Forward fill open prices
            df['Open'] = df['Close'].shift(1).fillna(df['Close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {e}")
            return None
    
    def _parse_timeframe_minutes(self, timeframe_str: str) -> int:
        """Parse timeframe string to minutes"""
        timeframe_str = timeframe_str.lower()
        if 'm' in timeframe_str:
            return int(timeframe_str.replace('m', ''))
        elif 'h' in timeframe_str:
            return int(timeframe_str.replace('h', '')) * 60
        elif 'd' in timeframe_str:
            return int(timeframe_str.replace('d', '')) * 60 * 24
        try:
            return int(timeframe_str)
        except ValueError:
            logger.warning(f"Could not parse timeframe '{timeframe_str}'. Using 1 minute.")
            return 1
    
    def generate_enhanced_chart(self, symbol: str, timeframe: str,
                               close_prices, high_prices, low_prices, volumes,
                               tech_metrics: Optional[Dict[str, Any]] = None,
                               lookback_periods: int = 100) -> Tuple[str, str, List[DetectedPattern]]:
        """Generate enhanced chart with patterns and advanced indicators"""
        saved_filepath = ""
        detected_patterns = []
        
        try:
            # Prepare DataFrame
            df = self._prepare_dataframe(close_prices, high_prices, low_prices, 
                                       volumes, lookback_periods, timeframe)
            
            if df is None or len(df) < 15:
                logger.warning(f"Insufficient data for {symbol} chart: {len(df) if df is not None else 0} points")
                error_b64 = self._generate_error_chart(f"Insufficient data for {symbol}")
                return error_b64, "", []
            
            # Calculate indicators
            indicators = self._calculate_enhanced_indicators(df)
            
            # Detect patterns
            candlestick_patterns = self._detect_candlestick_patterns(df)
            chart_patterns = self._detect_chart_patterns(df)
            detected_patterns = candlestick_patterns + chart_patterns
            
            # Filter patterns by confidence
            detected_patterns = [p for p in detected_patterns 
                               if p.confidence >= self.config.min_pattern_strength]
            
            # Create chart
            chart_b64 = self._create_enhanced_chart(df, indicators, detected_patterns,
                                                   symbol, timeframe, tech_metrics or {})
            
            # Save to disk if enabled
            if (self._save_charts_to_disk and chart_b64 and 
                not chart_b64.startswith("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")):
                saved_filepath = self._save_chart_to_disk(symbol, timeframe, chart_b64, detected_patterns)
            
            return chart_b64, saved_filepath, detected_patterns
            
        except Exception as e:
            logger.error(f"Error generating enhanced chart for {symbol}: {e}", exc_info=True)
            error_b64 = self._generate_error_chart(f"Fatal Error: {str(e)[:80]}")
            return error_b64, "", []
    
    def _save_chart_to_disk(self, symbol: str, timeframe: str, chart_base64: str,
                           patterns: List[DetectedPattern]) -> str:
        """Save chart and pattern data to disk"""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            base_filename = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}"
            
            # Save chart image
            chart_filepath = self._charts_dir / f"{base_filename}.png"
            chart_data = base64.b64decode(chart_base64)
            with open(chart_filepath, 'wb') as f:
                f.write(chart_data)
            
            # Save pattern data
            if patterns:
                pattern_filepath = self._charts_dir / f"{base_filename}_patterns.json"
                pattern_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': timestamp,
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
                
                with open(pattern_filepath, 'w') as f:
                    json.dump(pattern_data, f, indent=2)
            
            logger.info(f"Enhanced chart saved: {chart_filepath}")
            return str(chart_filepath)
            
        except Exception as e:
            logger.error(f"Error saving enhanced chart: {e}")
            return ""


# Wrapper function for backward compatibility
def generate_enhanced_chart_for_visual_agent(
    symbol: str, timeframe: str,
    close_buf, high_buf, low_buf, vol_buf,
    tech_metrics: Optional[Dict[str, Any]] = None,
    lookback_periods: int = 100,
    config: Optional[ChartConfig] = None,
    save_chart: bool = True
) -> Tuple[str, str, List[DetectedPattern]]:
    """Enhanced wrapper function for visual agent integration"""
    try:
        generator = EnhancedChartGenerator(
            config=config or ChartConfig(),
            save_charts_to_disk=save_chart
        )
        
        return generator.generate_enhanced_chart(
            symbol=symbol, timeframe=timeframe,
            close_prices=close_buf, high_prices=high_buf,
            low_prices=low_buf, volumes=vol_buf,
            tech_metrics=tech_metrics or {},
            lookback_periods=lookback_periods
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced chart wrapper: {e}")
        try:
            error_b64 = EnhancedChartGenerator()._generate_error_chart(f"Wrapper Error: {str(e)[:60]}")
            return error_b64, "", []
        except:
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", "", []
