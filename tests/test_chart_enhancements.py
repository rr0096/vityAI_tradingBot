# tests/test_chart_enhancements.py
"""
Tests for the enhanced chart generator functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

try:
    from tools.chart_enhancements import (
        EnhancedPatternDetector,
        EnhancedIndicatorCalculator,
        EnhancedChartGenerator,
        ChartPattern,
        PatternType,
        SupportResistanceLevel,
        analyze_chart_patterns,
        get_pattern_summary,
        generate_enhanced_chart_for_visual_agent
    )
except ImportError as e:
    pytest.skip(f"Chart enhancements module not available: {e}", allow_module_level=True)

class TestEnhancedPatternDetector:
    """Test suite for enhanced pattern detection"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        np.random.seed(42)  # For reproducible tests
        
        # Create realistic OHLC data
        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        open_prices = close_prices + np.random.randn(50) * 0.1
        high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(50) * 0.2)
        low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(50) * 0.2)
        volumes = np.random.randint(1000, 10000, 50)
        
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def pattern_detector(self):
        """Create pattern detector instance"""
        return EnhancedPatternDetector(min_confidence=0.5)
    
    def test_pattern_detector_initialization(self, pattern_detector):
        """Test pattern detector initialization"""
        assert pattern_detector.min_confidence == 0.5
    
    def test_detect_candlestick_patterns_empty_df(self, pattern_detector):
        """Test pattern detection with empty DataFrame"""
        empty_df = pd.DataFrame()
        patterns = pattern_detector.detect_candlestick_patterns(empty_df)
        assert patterns == []
    
    def test_detect_candlestick_patterns_small_df(self, pattern_detector):
        """Test pattern detection with insufficient data"""
        small_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1000]
        })
        patterns = pattern_detector.detect_candlestick_patterns(small_df)
        assert patterns == []
    
    def test_detect_candlestick_patterns_valid_df(self, pattern_detector, sample_df):
        """Test pattern detection with valid DataFrame"""
        patterns = pattern_detector.detect_candlestick_patterns(sample_df)
        
        # Should return a list of ChartPattern objects
        assert isinstance(patterns, list)
        for pattern in patterns:
            assert isinstance(pattern, ChartPattern)
            assert pattern.confidence >= pattern_detector.min_confidence
            assert pattern.pattern_type == PatternType.CANDLESTICK
    
    def test_doji_pattern_detection(self, pattern_detector):
        """Test specific Doji pattern detection"""
        # Create a DataFrame with a clear Doji pattern
        df = pd.DataFrame({
            'Open': [100, 100, 100.01],  # Small body
            'High': [102, 102, 101],
            'Low': [98, 98, 99],
            'Close': [101, 101, 100.02],  # Almost same as open
            'Volume': [1000, 1000, 1000]
        })
        
        patterns = pattern_detector.detect_candlestick_patterns(df)
        doji_patterns = [p for p in patterns if p.name == "Doji"]
        
        # Should detect at least one Doji
        assert len(doji_patterns) > 0
        doji = doji_patterns[0]
        assert doji.bullish_signal is None  # Doji is neutral
    
    def test_hammer_pattern_detection(self, pattern_detector):
        """Test Hammer pattern detection"""
        # Create a DataFrame with a Hammer pattern
        closes = [105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 100]  # Downtrend then hammer
        df = pd.DataFrame({
            'Open': [c + 0.5 for c in closes],
            'High': [c + 1 for c in closes[:-1]] + [100.2],  # Small upper shadow for hammer
            'Low': [c - 1 for c in closes[:-1]] + [97],      # Long lower shadow for hammer
            'Close': closes,
            'Volume': [1000] * 11
        })
        
        patterns = pattern_detector.detect_candlestick_patterns(df)
        hammer_patterns = [p for p in patterns if p.name == "Hammer"]
        
        # Might detect hammer pattern
        if hammer_patterns:
            hammer = hammer_patterns[0]
            assert hammer.bullish_signal is True
    
    def test_support_resistance_detection(self, pattern_detector, sample_df):
        """Test support and resistance level detection"""
        sr_levels = pattern_detector.detect_support_resistance_levels(sample_df)
        
        # Should return a list of SupportResistanceLevel objects
        assert isinstance(sr_levels, list)
        for level in sr_levels:
            assert isinstance(level, SupportResistanceLevel)
            assert isinstance(level.price, float)
            assert 0 <= level.strength <= 1
            assert level.touches >= 2  # Minimum touches
            assert isinstance(level.is_support, bool)
    
    def test_support_resistance_insufficient_data(self, pattern_detector):
        """Test S/R detection with insufficient data"""
        small_df = pd.DataFrame({
            'High': [100, 101, 102],
            'Low': [99, 100, 101],
            'Close': [100, 101, 102]
        })
        
        sr_levels = pattern_detector.detect_support_resistance_levels(small_df)
        assert sr_levels == []


class TestEnhancedIndicatorCalculator:
    """Test suite for enhanced indicator calculations"""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price series"""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))
        return prices
    
    def test_calculate_ema(self, sample_prices):
        """Test EMA calculation"""
        ema = EnhancedIndicatorCalculator.calculate_ema(sample_prices, 12)
        
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_prices)
        assert not ema.isnull().all()
    
    def test_calculate_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = EnhancedIndicatorCalculator.calculate_bollinger_bands(sample_prices, 20, 2.0)
        
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(sample_prices)
        
        # Upper should be above middle, middle above lower (where not NaN)
        valid_mask = ~(upper.isnull() | middle.isnull() | lower.isnull())
        if valid_mask.any():
            assert (upper[valid_mask] >= middle[valid_mask]).all()
            assert (middle[valid_mask] >= lower[valid_mask]).all()
    
    def test_calculate_rsi_simple(self, sample_prices):
        """Test simple RSI calculation"""
        rsi = EnhancedIndicatorCalculator.calculate_rsi_simple(sample_prices, 14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_prices)
        
        # RSI should be between 0 and 100 (where not NaN)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()
    
    def test_calculate_macd_simple(self, sample_prices):
        """Test simple MACD calculation"""
        macd_line, signal_line, histogram = EnhancedIndicatorCalculator.calculate_macd_simple(sample_prices, 12, 26, 9)
        
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd_line) == len(sample_prices)
        
        # Histogram should equal MACD line minus signal line (where not NaN)
        valid_mask = ~(macd_line.isnull() | signal_line.isnull() | histogram.isnull())
        if valid_mask.any():
            np.testing.assert_array_almost_equal(
                histogram[valid_mask].values,
                (macd_line - signal_line)[valid_mask].values,
                decimal=6
            )


class TestEnhancedChartGenerator:
    """Test suite for enhanced chart generator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        np.random.seed(42)
        size = 100
        
        close_prices = list(100 + np.cumsum(np.random.randn(size) * 0.5))
        high_prices = [c + abs(np.random.randn()) * 0.5 for c in close_prices]
        low_prices = [c - abs(np.random.randn()) * 0.5 for c in close_prices]
        volumes = list(np.random.randint(1000, 10000, size))
        
        return close_prices, high_prices, low_prices, volumes
    
    @pytest.fixture
    def enhanced_generator(self):
        """Create enhanced chart generator instance"""
        return EnhancedChartGenerator(save_charts_to_disk=False)
    
    def test_enhanced_generator_initialization(self, enhanced_generator):
        """Test enhanced generator initialization"""
        assert isinstance(enhanced_generator, EnhancedChartGenerator)
        assert hasattr(enhanced_generator, 'pattern_detector')
        assert hasattr(enhanced_generator, 'indicator_calculator')
    
    @patch('tools.chart_enhancements.EnhancedChartGenerator._generate_error_chart')
    def test_generate_enhanced_chart_insufficient_data(self, mock_error_chart, enhanced_generator):
        """Test enhanced chart generation with insufficient data"""
        mock_error_chart.return_value = "error_chart_b64"
        
        # Very small dataset
        close_prices = [100, 101]
        high_prices = [101, 102]
        low_prices = [99, 100]
        volumes = [1000, 1000]
        
        chart_b64, filepath, patterns = enhanced_generator.generate_enhanced_chart_image(
            "BTCUSDT", "1h", close_prices, high_prices, low_prices, volumes
        )
        
        assert chart_b64 == "error_chart_b64"
        assert filepath == ""
        assert patterns == []
        mock_error_chart.assert_called_once()
    
    @patch('tools.chart_enhancements.EnhancedChartGenerator._create_chart')
    def test_generate_enhanced_chart_valid_data(self, mock_create_chart, enhanced_generator, sample_data):
        """Test enhanced chart generation with valid data"""
        mock_create_chart.return_value = "valid_chart_b64"
        close_prices, high_prices, low_prices, volumes = sample_data
        
        chart_b64, filepath, patterns = enhanced_generator.generate_enhanced_chart_image(
            "BTCUSDT", "1h", close_prices, high_prices, low_prices, volumes,
            detect_patterns=True
        )
        
        assert isinstance(chart_b64, str)
        assert isinstance(filepath, str)
        assert isinstance(patterns, list)
        mock_create_chart.assert_called_once()


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        dates = pd.date_range('2024-01-01', periods=30, freq='1H')
        np.random.seed(42)
        
        close_prices = 100 + np.cumsum(np.random.randn(30) * 0.5)
        open_prices = close_prices + np.random.randn(30) * 0.1
        high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(30) * 0.2)
        low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(30) * 0.2)
        volumes = np.random.randint(1000, 10000, 30)
        
        return pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
    
    def test_analyze_chart_patterns(self, sample_df):
        """Test chart pattern analysis function"""
        analysis = analyze_chart_patterns(sample_df)
        
        assert isinstance(analysis, dict)
        required_keys = [
            'patterns_detected', 'bullish_patterns', 'bearish_patterns',
            'neutral_patterns', 'support_levels', 'resistance_levels',
            'patterns', 'sr_levels'
        ]
        
        for key in required_keys:
            assert key in analysis
        
        assert isinstance(analysis['patterns_detected'], int)
        assert isinstance(analysis['patterns'], list)
        assert isinstance(analysis['sr_levels'], list)
    
    def test_get_pattern_summary_empty(self):
        """Test pattern summary with no patterns"""
        summary = get_pattern_summary([])
        assert summary == "No significant patterns detected"
    
    def test_get_pattern_summary_with_patterns(self):
        """Test pattern summary with patterns"""
        patterns = [
            ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Doji",
                confidence=0.8,
                bullish_signal=None
            ),
            ChartPattern(
                pattern_type=PatternType.CANDLESTICK,
                name="Hammer",
                confidence=0.9,
                bullish_signal=True
            )
        ]
        
        summary = get_pattern_summary(patterns)
        
        assert "2 patterns" in summary
        assert "1 bullish" in summary
        assert "1 neutral" in summary
        assert "Hammer" in summary  # Strongest pattern
        assert "90%" in summary or "0.9" in summary


class TestIntegrationFunctions:
    """Test suite for integration functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        size = 50
        
        close_prices = list(100 + np.cumsum(np.random.randn(size) * 0.5))
        high_prices = [c + abs(np.random.randn()) * 0.3 for c in close_prices]
        low_prices = [c - abs(np.random.randn()) * 0.3 for c in close_prices]
        volumes = list(np.random.randint(1000, 10000, size))
        
        return close_prices, high_prices, low_prices, volumes
    
    @patch('tools.chart_enhancements.EnhancedChartGenerator')
    def test_generate_enhanced_chart_for_visual_agent(self, mock_generator_class, sample_data):
        """Test the main integration function"""
        # Setup mock
        mock_generator = MagicMock()
        mock_generator.generate_enhanced_chart_image.return_value = ("chart_b64", "filepath", [])
        mock_generator_class.return_value = mock_generator
        
        close_prices, high_prices, low_prices, volumes = sample_data
        
        result = generate_enhanced_chart_for_visual_agent(
            "BTCUSDT", "1h", close_prices, high_prices, low_prices, volumes,
            detect_patterns=True
        )
        
        assert len(result) == 3
        chart_b64, filepath, patterns = result
        assert isinstance(chart_b64, str)
        assert isinstance(filepath, str)
        assert isinstance(patterns, list)
        
        mock_generator_class.assert_called_once_with(save_charts_to_disk=True)
        mock_generator.generate_enhanced_chart_image.assert_called_once()
    
    @patch('tools.chart_enhancements.EnhancedChartGenerator')
    def test_generate_enhanced_chart_error_handling(self, mock_generator_class, sample_data):
        """Test error handling in integration function"""
        # Setup mock to raise exception
        mock_generator_class.side_effect = Exception("Test error")
        
        close_prices, high_prices, low_prices, volumes = sample_data
        
        result = generate_enhanced_chart_for_visual_agent(
            "BTCUSDT", "1h", close_prices, high_prices, low_prices, volumes
        )
        
        chart_b64, filepath, patterns = result
        assert isinstance(chart_b64, str)  # Should return fallback
        assert filepath == ""
        assert patterns == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
