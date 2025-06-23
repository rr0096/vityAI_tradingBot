#!/usr/bin/env python3

"""
TradingView Auto-Capture System
Automatically captures charts from TradingView with your specified settings.
Works with Safari, Firefox, DuckDuckGo Browser, or any available browser.
"""

import time
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import base64

# Try importing selenium components
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.safari.service import Service as SafariService
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradingViewCapture:
    """
    Captures live TradingView charts with technical indicators automatically.
    """
    
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.driver = None
        self.wait = None
        
    def _detect_available_browser(self) -> Optional[str]:
        """Detect which browsers are available on the system."""
        browsers = {
            'safari': '/Applications/Safari.app',
            'firefox': '/Applications/Firefox.app',
            'duckduckgo': '/Applications/DuckDuckGo Privacy Browser.app',
        }
        
        available = []
        for browser, path in browsers.items():
            if Path(path).exists():
                available.append(browser)
                logger.info(f"✅ Found {browser.title()} browser")
        
        # Also check for Firefox in common locations
        firefox_paths = [
            '/usr/bin/firefox',
            '/usr/local/bin/firefox',
            shutil.which('firefox')
        ]
        
        for firefox_path in firefox_paths:
            if firefox_path and Path(firefox_path).exists():
                if 'firefox' not in available:
                    available.append('firefox')
                break
        
        logger.info(f"Available browsers: {available}")
        return available[0] if available else None
    
    def _setup_driver(self, browser: str = None) -> bool:
        """Setup the browser driver."""
        if not SELENIUM_AVAILABLE:
            logger.error("❌ Selenium not installed. Install with: pip install selenium")
            return False
        
        if not browser:
            browser = self._detect_available_browser()
            
        if not browser:
            logger.error("❌ No compatible browser found!")
            return False
        
        try:
            if browser == 'safari':
                # Safari requires enabling Developer menu and allowing remote automation
                logger.info("🚀 Setting up Safari driver...")
                self.driver = webdriver.Safari()
                
            elif browser == 'firefox':
                logger.info("🚀 Setting up Firefox driver...")
                options = FirefoxOptions()
                # Run in background (optional)
                # options.add_argument('--headless')
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                
                # Try to find Firefox binary
                firefox_binary = shutil.which('firefox')
                if firefox_binary:
                    options.binary_location = firefox_binary
                
                self.driver = webdriver.Firefox(options=options)
            
            else:
                logger.error(f"❌ Browser {browser} not supported yet")
                return False
            
            self.driver.set_window_size(1920, 1080)  # Full HD for better charts
            self.wait = WebDriverWait(self.driver, 15)
            logger.info(f"✅ {browser.title()} driver setup successful!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup {browser} driver: {e}")
            return False
    
    def capture_symbol_chart(self, 
                           symbol: str = "SOLUSDT", 
                           timeframe: str = "1m",
                           indicators: list = None) -> Optional[str]:
        """
        Capture a TradingView chart for the specified symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., "SOLUSDT", "BTCUSDT")
            timeframe: Chart timeframe (1m, 5m, 15m, 1h, 4h, 1D)
            indicators: List of indicators to add (e.g., ["RSI", "MACD", "EMA"])
        
        Returns:
            Path to the captured image file, or None if failed
        """
        if not self.driver:
            if not self._setup_driver():
                return None
        
        if indicators is None:
            indicators = ["RSI", "MACD", "EMA(20)", "EMA(50)"]
        
        try:
            # Build TradingView URL
            base_url = "https://www.tradingview.com/chart/"
            url = f"{base_url}?symbol=BINANCE:{symbol}&interval={timeframe}"
            
            logger.info(f"🌐 Opening TradingView for {symbol} ({timeframe})...")
            self.driver.get(url)
            
            # Wait for chart to load
            time.sleep(5)
            
            # Try to dismiss any popups/modals
            self._dismiss_popups()
            
            # Wait for chart canvas to be present
            try:
                chart_element = self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "canvas, .chart-container, .chart-widget"))
                )
                logger.info("📊 Chart loaded successfully!")
            except Exception:
                logger.warning("⚠️ Could not detect chart element, proceeding anyway...")
            
            # Add indicators (optional - TradingView might have them by default)
            self._add_indicators(indicators)
            
            # Wait a bit for everything to render
            time.sleep(3)
            
            # Take screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}.png"
            filepath = self.output_dir / filename
            
            logger.info(f"📸 Taking screenshot...")
            self.driver.save_screenshot(str(filepath))
            
            # Verify file was created and has reasonable size
            if filepath.exists() and filepath.stat().st_size > 1000:  # At least 1KB
                logger.info(f"✅ Chart captured successfully: {filepath}")
                return str(filepath)
            else:
                logger.error("❌ Screenshot file is missing or too small")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to capture chart: {e}")
            return None
    
    def _dismiss_popups(self):
        """Try to dismiss common TradingView popups."""
        try:
            # Common selectors for popups/modals
            popup_selectors = [
                "button[data-name='close']",
                ".close-button",
                ".modal-close",
                "[aria-label='Close']",
                ".tv-dialog__close",
                ".js-dialog__close",
            ]
            
            for selector in popup_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            element.click()
                            logger.info(f"🚫 Dismissed popup: {selector}")
                            time.sleep(1)
                except Exception:
                    continue
                    
            # Press ESC key to dismiss any remaining popups
            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
            
        except Exception as e:
            logger.debug(f"Popup dismissal: {e}")
    
    def _add_indicators(self, indicators: list):
        """Try to add technical indicators to the chart."""
        try:
            logger.info(f"📈 Attempting to add indicators: {indicators}")
            
            # Look for indicators button/menu
            indicator_selectors = [
                "[data-name='indicators']",
                ".chart-toolbar-button[title*='Indicators']",
                ".js-indicators-button",
                "button[aria-label*='Indicators']"
            ]
            
            indicator_button = None
            for selector in indicator_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            indicator_button = element
                            break
                    if indicator_button:
                        break
                except Exception:
                    continue
            
            if indicator_button:
                indicator_button.click()
                time.sleep(2)
                logger.info("📊 Opened indicators menu")
                
                # For now, we'll rely on TradingView's default indicators
                # Advanced indicator selection would require more complex automation
                
            else:
                logger.info("ℹ️ Could not find indicators menu, using default view")
                
        except Exception as e:
            logger.debug(f"Indicator addition: {e}")
    
    def capture_multiple_timeframes(self, symbol: str, timeframes: list = None) -> Dict[str, str]:
        """Capture charts for multiple timeframes."""
        if timeframes is None:
            timeframes = ["1m", "5m", "15m", "1h"]
        
        results = {}
        for tf in timeframes:
            logger.info(f"📊 Capturing {symbol} - {tf}")
            filepath = self.capture_symbol_chart(symbol, tf)
            if filepath:
                results[tf] = filepath
            time.sleep(2)  # Brief pause between captures
        
        return results
    
    def close(self):
        """Close the browser driver."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🔒 Browser closed")
            except Exception:
                pass
            self.driver = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Test the TradingView capture system."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🎯 TradingView Auto-Capture Test")
    print("=" * 40)
    
    # Test the capture system
    with TradingViewCapture() as capture:
        # Test single capture
        chart_path = capture.capture_symbol_chart(
            symbol="SOLUSDT",
            timeframe="1m",
            indicators=["RSI", "MACD", "EMA"]
        )
        
        if chart_path:
            print(f"✅ SUCCESS! Chart saved to: {chart_path}")
            print(f"📁 File size: {Path(chart_path).stat().st_size / 1024:.1f} KB")
        else:
            print("❌ FAILED to capture chart")


if __name__ == "__main__":
    main()
