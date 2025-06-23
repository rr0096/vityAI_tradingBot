# tools/chart_generator.py
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot        try:
            cl = list(close_prices)
            hi = list(high_prices) 
            lo = list(low_prices)
            vo = list(volumes)
            
            min_len_data = min(len(cl), len(hi), len(lo), len(vo))
            # More restrictive limit to prevent huge images
            limited_lookback = min(lookback_periods, 100)  # Maximum 100 data points
            actual_lookback = min(limited_lookback, min_len_data)import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
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
import os
import warnings

# Suppress matplotlib warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Constants
FALLBACK_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

logger = logging.getLogger(__name__)

class ChartGeneratorError(Exception):
    """Custom exception for chart generation errors."""
    pass

class EnhancedChartGenerator:
    """
    Enhanced chart generator with improved performance, error handling,
    and additional technical indicators for better trading analysis.
    """
    def __init__(self, save_charts_to_disk: bool = False, charts_dir_str: str = "logs/charts"):
        # Enhanced color scheme for better visibility
        self.up_color = '#26A69A'
        self.down_color = '#EF5350'
        self.neutral_color = '#787B86'
        self.text_color = '#E0E0E0'
        self.grid_color = '#404040'
        self.bg_color = '#131722'
        self.axes_bg_color = '#1A1E29'
        self.accent_color = '#FF9800'  # For highlights
        self.support_color = '#4CAF50'  # Green for support
        self.resistance_color = '#F44336'  # Red for resistance

        self._save_charts_to_disk = save_charts_to_disk
        self._charts_dir = Path(charts_dir_str)
        if self._save_charts_to_disk:
            self._charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set reasonable size limits for images
        self.max_width = 1200  # Maximum width in pixels
        self.max_height = 800  # Maximum height in pixels
        self.default_figsize = (12, 8)  # Reasonable figure size

        # Enhanced market colors with better alpha
        self.mpf_marketcolors = mpf.make_marketcolors(
            up=self.up_color, down=self.down_color, edge='inherit',
            wick={'up': self.up_color, 'down': self.down_color},
            volume={'up': self.up_color, 'down': self.down_color}, alpha=0.9
        )
        
        # Improved styling for better readability
        self.style_rc_params = {
            'font.size': 8,
            'axes.labelsize': 7, 'axes.titlesize': 10,
            'axes.grid': True, 'grid.alpha': 0.15, 'grid.color': self.grid_color,
            'figure.facecolor': self.bg_color, 'axes.facecolor': self.axes_bg_color,
            'xtick.color': self.text_color, 'ytick.color': self.text_color,
            'axes.labelcolor': self.text_color, 'axes.titlecolor': self.text_color,
            'text.color': self.text_color,
            'figure.dpi': 60,  # Much lower DPI for smaller file sizes
            'savefig.dpi': 60,  # Much lower DPI
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05
        }
        
        try:
            self.custom_mpf_style = mpf.make_mpf_style(
                base_mpf_style='nightclouds',
                marketcolors=self.mpf_marketcolors,
                rc=self.style_rc_params,
                y_on_right=False
            )
        except Exception as e:
            logger.warning(f"Error creating custom mplfinance style: {e}. Falling back to 'nightclouds'.")
            self.custom_mpf_style = 'nightclouds'
    
    def _parse_timeframe_minutes(self, timeframe_str: str) -> int:
        """Parse timeframe string to minutes with enhanced validation."""
        timeframe_str = timeframe_str.lower().strip()
        try:
            if 'm' in timeframe_str:
                return max(1, int(timeframe_str.replace('m', '')))
            elif 'h' in timeframe_str:
                return max(1, int(timeframe_str.replace('h', '')) * 60)
            elif 'd' in timeframe_str:
                return max(1, int(timeframe_str.replace('d', '')) * 60 * 24)
            elif 'w' in timeframe_str:
                return max(1, int(timeframe_str.replace('w', '')) * 60 * 24 * 7)
            else:
                return max(1, int(timeframe_str))
        except ValueError:
            logger.warning(f"Could not parse timeframe '{timeframe_str}' for interval. Defaulting to 1 min.")
            return 1

    def _validate_price_data(self, close: float, high: float, low: float, volume: float) -> Tuple[float, float, float, float]:
        """Validate and correct price data to ensure OHLC rules."""
        # Ensure all values are positive (except volume can be 0)
        close = max(0.000001, float(close))
        high = max(0.000001, float(high))
        low = max(0.000001, float(low))
        volume = max(0, float(volume))
        
        # Ensure OHLC rules: High >= Close,Low and Low <= Close,High
        actual_high = max(close, high, low)
        actual_low = min(close, high, low)
        
        return close, actual_high, actual_low, volume

    def _prepare_and_validate_dataframe(
        self,
        close_prices: Union[List[float], TypingDeque[float]],
        high_prices: Union[List[float], TypingDeque[float]],
        low_prices: Union[List[float], TypingDeque[float]],
        volumes: Union[List[float], TypingDeque[float]],
        lookback_periods: int,
        timeframe_str: str
    ) -> Optional[pd.DataFrame]:
        try:
            # Limit data for reasonable chart size
            limited_lookback = self._limit_data_for_reasonable_chart(lookback_periods)
            
            cl = list(close_prices)
            hi = list(high_prices) 
            lo = list(low_prices)
            vo = list(volumes)
            
            min_len_data = min(len(cl), len(hi), len(lo), len(vo))
            actual_lookback = min(limited_lookback, min_len_data)
            
            min_candles_for_chart = 10
            if actual_lookback < min_candles_for_chart:
                 logger.error(f"Not enough data for chart. Available: {min_len_data}, Requested lookback: {lookback_periods}, Effective lookback: {actual_lookback}, Min required: {min_candles_for_chart}")
                 return None

            cl = cl[-actual_lookback:]
            hi = hi[-actual_lookback:]
            lo = lo[-actual_lookback:]
            vo = vo[-actual_lookback:]

            if not (len(cl) == len(hi) == len(lo) == len(vo) and len(cl) >= min_candles_for_chart):
                logger.error(f"Data length mismatch or too short after slicing for chart. C:{len(cl)} H:{len(hi)} L:{len(lo)} V:{len(vo)}")
                return None
            
            for i in range(len(cl)):
                c_val, h_val, l_val, v_val = cl[i], hi[i], lo[i], vo[i]
                if not (isinstance(c_val, (int, float)) and c_val > 0 and
                        isinstance(h_val, (int, float)) and h_val > 0 and
                        isinstance(l_val, (int, float)) and l_val > 0 and
                        isinstance(v_val, (int, float)) and v_val >= 0):
                    logger.error(f"Invalid data type or non-positive value at index {i}. C:{c_val} H:{h_val} L:{l_val} V:{v_val}")
                    return None

                cl[i], hi[i], lo[i], vo[i] = self._validate_price_data(c_val, h_val, l_val, v_val)
            
            interval_minutes = self._parse_timeframe_minutes(timeframe_str)
            end_time = datetime.now(timezone.utc)
            
            time_index = pd.to_datetime(
                [end_time - timedelta(minutes=i * interval_minutes) for i in range(len(cl) - 1, -1, -1)],
                utc=True
            )
            
            df = pd.DataFrame({'High': hi, 'Low': lo, 'Close': cl, 'Volume': vo}, index=time_index)
            df['Open'] = df['Close'].shift(1)
            # CORRECTED: Fix type issues with DataFrame indexing
            if not df.empty and pd.isna(df['Open'].iloc[0]):
                df.loc[df.index[0], 'Open'] = df.loc[df.index[0], 'Close']
            
            # CORRECTED: Avoid inplace=True on chained assignment
            df['Open'] = df['Open'].ffill()
            df['Open'] = df['Open'].bfill()

            if df.isnull().values.any():
                logger.error("DataFrame contains unfillable NaNs after preparation.")
                return None
            return df
        except Exception as e:
            logger.error(f"Error in _prepare_and_validate_dataframe: {e}", exc_info=True)
            return None

    def _calculate_chart_indicators(self, df: pd.DataFrame, tech_metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
        indicators: Dict[str, Any] = {}
        if df.empty or len(df) < 2: return indicators

        close_prices = df['Close'].astype(float)
        tech_metrics = tech_metrics or {}

        if len(close_prices) >= 20:
            indicators['sma_20'] = close_prices.rolling(window=20, min_periods=1).mean()
        if len(close_prices) >= 50:
            indicators['sma_50'] = close_prices.rolling(window=50, min_periods=1).mean()

        if len(close_prices) >= 20:
            try:
                import talib
                bb_upper, bb_middle, bb_lower = talib.BBANDS(np.array(close_prices.values, dtype=np.float64), timeperiod=20, nbdevup=2, nbdevdn=2)
                indicators['bb_upper'] = pd.Series(bb_upper, index=df.index)
                indicators['bb_middle'] = pd.Series(bb_middle, index=df.index)
                indicators['bb_lower'] = pd.Series(bb_lower, index=df.index)
            except ImportError: logger.warning("TA-Lib not found. Bollinger Bands will not be plotted.")
            except Exception as e_bb: logger.warning(f"Error calculating Bollinger Bands for chart: {e_bb}")

        if len(close_prices) >= 15:
             try:
                import talib
                rsi_values = talib.RSI(np.array(close_prices.values, dtype=np.float64), timeperiod=14)
                indicators['rsi'] = pd.Series(rsi_values, index=df.index)
             except ImportError: logger.warning("TA-Lib not found. RSI panel will not be plotted.")
             except Exception as e_rsi: logger.warning(f"Error calculating RSI for chart: {e_rsi}")
        
        if len(df) >= 10:
            indicators['support_plot'] = df['Low'].rolling(10, min_periods=1).min().iloc[-1]
            indicators['resistance_plot'] = df['High'].rolling(10, min_periods=1).max().iloc[-1]
        
        return indicators

    def _create_chart(
        self, df: pd.DataFrame, indicators_data: Dict[str, Any],
        symbol: str, timeframe: str, current_metrics: Dict[str, Any]
    ) -> str:
        fig = None
        try:
            addplots = []
            
            if 'sma_20' in indicators_data and not indicators_data['sma_20'].isnull().all():
                addplots.append(mpf.make_addplot(indicators_data['sma_20'], panel=0, color=self.neutral_color, width=0.7, alpha=0.6, linestyle='-.'))
            if 'sma_50' in indicators_data and not indicators_data['sma_50'].isnull().all():
                addplots.append(mpf.make_addplot(indicators_data['sma_50'], panel=0, color=self.neutral_color, width=0.9, alpha=0.8, linestyle='-'))
            if 'bb_upper' in indicators_data and not indicators_data['bb_upper'].isnull().all():
                addplots.append(mpf.make_addplot(indicators_data['bb_upper'], panel=0, color=self.grid_color, linestyle=':', width=0.6, alpha=0.5))
            if 'bb_lower' in indicators_data and not indicators_data['bb_lower'].isnull().all():
                addplots.append(mpf.make_addplot(indicators_data['bb_lower'], panel=0, color=self.grid_color, linestyle=':', width=0.6, alpha=0.5))
            
            rsi_series = indicators_data.get('rsi')
            has_rsi_panel = rsi_series is not None and not rsi_series.isnull().all()
            
            panel_ids = {'price': 0}; next_panel_id = 1
            
            if has_rsi_panel:
                panel_ids['rsi'] = next_panel_id
                addplots.append(mpf.make_addplot(rsi_series, panel=panel_ids['rsi'], ylabel='RSI', color=self.text_color, width=0.9, y_on_right=False, secondary_y=False))
                next_panel_id += 1

            panel_ratios_list = [3]
            if has_rsi_panel: panel_ratios_list.append(1)
            panel_ratios_config = tuple(panel_ratios_list)
            if len(panel_ratios_config) == 1: panel_ratios_config = (3,1)

            fig, axes = mpf.plot(
                df, type='candle', style=self.custom_mpf_style,
                volume=True, addplot=addplots if addplots else None,
                figsize=(8, 5), returnfig=True, panel_ratios=panel_ratios_config,
                figscale=0.8, tight_layout=True, show_nontrading=False,
                datetime_format=' %H:%M', xrotation=30,
                warn_too_much_data=50  # Limit data to prevent huge images
            )
            
            ax_main = axes[0]
            if 'bb_upper' in indicators_data and 'bb_lower' in indicators_data and \
               not indicators_data['bb_upper'].isnull().all() and not indicators_data['bb_lower'].isnull().all():
                ax_main.fill_between(df.index, indicators_data['bb_upper'], indicators_data['bb_lower'], alpha=0.08, color=self.neutral_color)

            self._annotate_sr_levels_on_ax(ax_main, df, indicators_data.get('support_plot'), indicators_data.get('resistance_plot'))
            
            if has_rsi_panel and 'rsi' in panel_ids and panel_ids['rsi'] < len(axes):
                ax_rsi = axes[panel_ids['rsi']]
                ax_rsi.axhline(70, color=self.down_color, linestyle=':', lw=0.7, alpha=0.6)
                ax_rsi.axhline(30, color=self.up_color, linestyle=':', lw=0.7, alpha=0.6)
                ax_rsi.fill_between(df.index, 70, 100, alpha=0.07, color=self.down_color)
                ax_rsi.fill_between(df.index, 0, 30, alpha=0.07, color=self.up_color)
                ax_rsi.set_ylim(0, 100); ax_rsi.tick_params(axis='y', labelcolor=self.text_color, labelsize=6)
                ax_rsi.set_ylabel('RSI', color=self.text_color, fontsize=7)
            
            self._add_title_and_metrics_to_fig(fig, symbol, timeframe, current_metrics, df)
            self._add_legend_to_main_ax(ax_main, indicators_data)

            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Error in _create_chart for {symbol}: {e}", exc_info=True)
            if fig: plt.close(fig)
            return self._generate_error_chart(f"Plot Creation Error: {str(e)[:70]}")

    def _annotate_sr_levels_on_ax(self, ax: Axes, df: pd.DataFrame, support: Optional[float], resistance: Optional[float]):
        try:
            if support is not None and support > 0:
                ax.axhline(y=support, color=self.support_color, linestyle=':', alpha=0.7, linewidth=1.0)
                if not df.index.empty:
                    ax.text(df.index[-1], support, f' S {support:.2f}', color=self.support_color, fontsize=6, va='bottom', ha='right')
            if resistance is not None and resistance > 0:
                ax.axhline(y=resistance, color=self.resistance_color, linestyle=':', alpha=0.7, linewidth=1.0)
                if not df.index.empty:
                    ax.text(df.index[-1], resistance, f' R {resistance:.2f}', color=self.resistance_color, fontsize=6, va='top', ha='right')
        except Exception as e:
            logger.warning(f"Error annotating S/R levels: {e}")

    def _add_title_and_metrics_to_fig(self, fig: Figure, symbol: str, timeframe: str, metrics: Dict[str, Any], df: pd.DataFrame):
        try:
            last_price_val = metrics.get('last_price', df['Close'].iloc[-1] if not df.empty and not df['Close'].empty else 'N/A')
            price_str = f"${last_price_val:.2f}" if isinstance(last_price_val, (float, int)) else str(last_price_val)
            title = f"{symbol} - {timeframe}   |   Last: {price_str}"
            
            rsi_val = metrics.get('rsi')
            if rsi_val is not None and isinstance(rsi_val, (float, int)) and np.isfinite(rsi_val):
                title += f"   RSI: {rsi_val:.1f}"
            
            fig.suptitle(title, fontsize=10, color=self.text_color, y=0.98)
        except Exception as e:
            logger.warning(f"Error adding title/metrics to chart: {e}")

    def _add_legend_to_main_ax(self, ax: Axes, indicators: Dict[str, Any]):
        try:
            handles = []
            if 'sma_20' in indicators and not indicators['sma_20'].isnull().all():
                handles.append(Line2D([], [], color=self.neutral_color, linestyle='-.', lw=1, label='SMA20'))
            if 'sma_50' in indicators and not indicators['sma_50'].isnull().all():
                handles.append(Line2D([], [], color=self.neutral_color, linestyle='-', lw=1.2, label='SMA50'))
            if 'bb_upper' in indicators and not indicators['bb_upper'].isnull().all():
                 handles.append(mpatches.Patch(color=self.neutral_color, alpha=0.1, label='BBands'))
            
            if handles:
                ax.legend(handles=handles, loc='upper left', fontsize=6, frameon=True, fancybox=True,
                          facecolor=self.axes_bg_color, edgecolor=self.grid_color,
                          labelcolor=self.text_color, framealpha=0.7)
        except Exception as e:
            logger.warning(f"Error adding legend to chart: {e}")
            
    def _fig_to_base64(self, fig: Figure) -> str:
        buf = None
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=self.style_rc_params.get('savefig.dpi', 80),
                        facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}", exc_info=True)
            if fig:
                plt.close(fig)
            return FALLBACK_IMAGE_B64
        finally:
            if buf:
                buf.close()

    def _generate_error_chart(self, error_message: str) -> str:
        fig = None
        try:
            fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=self.bg_color, dpi=80)
            ax.set_facecolor(self.axes_bg_color)
            ax.text(0.5, 0.5, f"CHART ERROR:\n{error_message[:100]}", color='#FF6B6B',
                   ha='center', va='center', fontsize=9, transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.4', fc='#3E2C2C', ec='#FF6B6B', alpha=0.8))
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout(pad=0.1)
            return self._fig_to_base64(fig)
        except Exception as e:
            logger.error(f"Critical error generating error chart itself: {e}", exc_info=True)
            if fig:
                plt.close(fig)
            return FALLBACK_IMAGE_B64

    def generate_chart_image(
        self, symbol: str, timeframe: str,
        close_prices: Union[List[float], TypingDeque[float]],
        high_prices: Union[List[float], TypingDeque[float]],
        low_prices: Union[List[float], TypingDeque[float]],
        volumes: Union[List[float], TypingDeque[float]],
        tech_metrics: Optional[Dict[str, Any]] = None,
        lookback_periods: int = 100,
        annotate_patterns: bool = True
    ) -> Tuple[str, str]:
        saved_filepath = ""
        try:
            df = self._prepare_and_validate_dataframe(
                close_prices, high_prices, low_prices, volumes, lookback_periods, timeframe
            )
            if df is None or len(df) < 15:
                logger.warning(f"Insufficient data for {symbol} chart after validation: {len(df) if df is not None else 0} points.")
                error_b64 = self._generate_error_chart(f"Insufficient data for {symbol} ({len(df) if df is not None else 0} pts).")
                return error_b64, ""
            
            current_metrics_for_title = tech_metrics or {}
            indicators_data = self._calculate_chart_indicators(df.copy(), current_metrics_for_title) # type: ignore

            chart_b64 = self._create_chart(df, indicators_data, symbol, timeframe, current_metrics_for_title)
            
            if self._save_charts_to_disk and chart_b64 and chart_b64 != FALLBACK_IMAGE_B64:
                saved_filepath = self._save_chart_to_file_internal(symbol, timeframe, chart_b64)

            return chart_b64, saved_filepath

        except Exception as e:
            logger.error(f"Top-level error in generate_chart_image for {symbol}: {e}", exc_info=True)
            error_b64 = self._generate_error_chart(f"Fatal Error Creating Chart: {str(e)[:80]}")
            return error_b64, ""

    def _save_chart_to_file_internal(self, symbol: str, timeframe: str, chart_base64: str) -> str:
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}.png"
            filepath = self._charts_dir / filename
            
            chart_data = base64.b64decode(chart_base64)
            with open(filepath, 'wb') as f:
                f.write(chart_data)
            
            logger.info(f"Chart image saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving chart image to file: {e}", exc_info=True)
            return ""

    def _limit_data_for_reasonable_chart(self, lookback_periods: int) -> int:
        """Limit the data points to create reasonable-sized charts"""
        # Maximum data points to prevent huge images
        max_data_points = 200  # Reasonable limit for visual analysis
        
        if lookback_periods > max_data_points:
            logger.info(f"Limiting lookback from {lookback_periods} to {max_data_points} points for reasonable chart size")
            return max_data_points
        
        return lookback_periods

# Global function to be called by agents
def generate_chart_for_visual_agent(
    symbol: str,
    timeframe: str,
    close_buf: Union[TypingDeque[float], List[float]],
    high_buf: Union[TypingDeque[float], List[float]],
    low_buf: Union[TypingDeque[float], List[float]],
    vol_buf: Union[TypingDeque[float], List[float]],
    tech_metrics: Optional[Dict[str, Any]] = None,
    lookback_periods: int = 100,
    save_chart: bool = True
) -> Tuple[str, str]:
    try:
        generator = EnhancedChartGenerator(save_charts_to_disk=save_chart)
        metrics_to_pass = tech_metrics if isinstance(tech_metrics, dict) else {}

        base64_image, saved_filepath = generator.generate_chart_image(
            symbol=symbol, timeframe=timeframe,
            close_prices=close_buf, high_prices=high_buf, low_prices=low_buf, volumes=vol_buf,
            tech_metrics=metrics_to_pass,
            lookback_periods=lookback_periods,
            annotate_patterns=True
        )
        return base64_image, saved_filepath
    except Exception as e:
        logger.error(f"Error in generate_chart_for_visual_agent wrapper: {e}", exc_info=True)
        try:
            error_b64 = EnhancedChartGenerator()._generate_error_chart(f"Chart Service Error: {str(e)[:60]}")
            return error_b64, ""
        except Exception:
            return FALLBACK_IMAGE_B64, ""

