#!/usr/bin/env python3
"""
FenixTradingBot - Paper Trading Demo
Sistema de paper trading usando la arquitectura heterogénea completa
"""

import logging
import asyncio
import json
from typing import Dict, Any
from datetime import datetime, timezone
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PaperTradingDemo:
    """
    Demo de paper trading que integra todos los componentes del sistema
    """
    
    def __init__(self):
        # Portfolio inicial
        self.initial_balance = 10000.0  # $10,000 USD
        self.current_balance = self.initial_balance
        self.positions = {}  # {symbol: {amount, avg_price, value}}
        self.trade_history = []
        
        # Configuración
        self.symbols = ['BTC', 'ETH']  # Símbolos para trading
        self.trading_active = True
        
        logger.info("🚀 Paper Trading Demo initialized")
        logger.info(f"💰 Initial balance: ${self.initial_balance:,.2f}")
    
    async def initialize_system(self):
        """Inicializa todos los componentes del sistema"""
        try:
            logger.info("🔧 Initializing trading system components...")
            
            # 1. Sistema de memoria
            from system.dynamic_memory_manager import DynamicMemoryManager
            from system.memory_aware_agent_manager import MemoryAwareAgentManager
            
            self.memory_manager = DynamicMemoryManager()
            self.agent_manager = MemoryAwareAgentManager()
            
            stats = self.memory_manager.get_memory_stats()
            logger.info(f"📊 Memory: {stats.total_mb}MB total, {stats.available_mb:.1f}MB available")
            
            # 2. Agentes principales
            from agents.sentiment_enhanced import EnhancedSentimentAnalyst
            from agents.technical_v_enhanced_fixed import EnhancedTechnicalAnalyst
            from agents.QABBAValidatorAgent import EnhancedQABBAAgent
            from agents.decision import EnhancedDecisionAgent
            
            self.sentiment_agent = EnhancedSentimentAnalyst()
            self.technical_agent = EnhancedTechnicalAnalyst()
            self.qabba_agent = EnhancedQABBAAgent()
            self.decision_agent = EnhancedDecisionAgent()
            
            logger.info("✅ All agents initialized")
            
            # 3. Fuentes de datos
            from tools.improved_news_scraper import ImprovedNewsScraper
            from tools.twitter_scraper import TwitterScraper
            
            self.news_scraper = ImprovedNewsScraper()
            self.twitter_scraper = TwitterScraper()
            
            logger.info("✅ Data sources initialized")
            
            # 4. Validador JSON
            from agents.json_validator import TradingSignalValidator
            self.validator = TradingSignalValidator()
            
            logger.info("✅ JSON Validator initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_mock_market_data(self, symbol: str) -> Dict[str, Any]:
        """Simula datos de mercado para paper trading"""
        import random
        
        # Precios base (simulados)
        base_prices = {'BTC': 43000, 'ETH': 2600}
        base_price = base_prices.get(symbol, 1000)
        
        # Añadir variación aleatoria
        variation = random.uniform(-0.05, 0.05)  # ±5%
        current_price = base_price * (1 + variation)
        
        return {
            'symbol': symbol,
            'price': round(current_price, 2),
            'volume': random.randint(1000000, 5000000),
            'change_24h': round(variation * 100, 2),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analiza un símbolo usando todos los agentes"""
        logger.info(f"🔍 Analyzing {symbol}...")
        
        # Obtener datos de mercado simulados
        market_data = self.get_mock_market_data(symbol)
        logger.info(f"📈 {symbol}: ${market_data['price']:.2f} ({market_data['change_24h']:+.2f}%)")
        
        analysis_results = {
            'symbol': symbol,
            'market_data': market_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # 1. Análisis de sentimiento (simplificado para demo)
            logger.info(f"🧠 Running sentiment analysis for {symbol}...")
            # En un sistema real, aquí llamaríamos al agent con datos reales
            sentiment_result = {
                'sentiment': 'NEUTRAL',
                'confidence': 0.75,
                'reasoning': f'Market sentiment for {symbol} appears neutral based on recent data'
            }
            analysis_results['sentiment'] = sentiment_result
            logger.info(f"✅ Sentiment: {sentiment_result['sentiment']} (confidence: {sentiment_result['confidence']:.2f})")
            
            # 2. Análisis técnico (simulado)
            logger.info(f"📊 Running technical analysis for {symbol}...")
            technical_result = {
                'signal': 'HOLD',
                'confidence': 0.68,
                'reasoning': f'Technical indicators for {symbol} suggest holding position'
            }
            analysis_results['technical'] = technical_result
            logger.info(f"✅ Technical: {technical_result['signal']} (confidence: {technical_result['confidence']:.2f})")
            
            # 3. Validación QABBA (simulada)
            logger.info(f"🔬 Running QABBA validation for {symbol}...")
            qabba_result = {
                'validation': 'APPROVED',
                'risk_level': 'MEDIUM',
                'reasoning': f'QABBA analysis approves trading signal for {symbol}'
            }
            analysis_results['qabba'] = qabba_result
            logger.info(f"✅ QABBA: {qabba_result['validation']} (risk: {qabba_result['risk_level']})")
            
            # 4. Decisión final
            logger.info(f"🎯 Making final decision for {symbol}...")
            decision_result = {
                'action': 'HOLD',
                'confidence': 0.70,
                'reasoning': f'Based on all analyses, holding {symbol} is recommended'
            }
            analysis_results['final_decision'] = decision_result
            logger.info(f"✅ Decision: {decision_result['action']} (confidence: {decision_result['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def execute_paper_trade(self, symbol: str, action: str, amount: float, price: float):
        """Ejecuta un trade simulado"""
        if action.upper() == 'BUY':
            cost = amount * price
            if cost <= self.current_balance:
                self.current_balance -= cost
                if symbol in self.positions:
                    # Promedio ponderado
                    total_amount = self.positions[symbol]['amount'] + amount
                    total_value = self.positions[symbol]['value'] + cost
                    self.positions[symbol] = {
                        'amount': total_amount,
                        'avg_price': total_value / total_amount,
                        'value': total_value
                    }
                else:
                    self.positions[symbol] = {
                        'amount': amount,
                        'avg_price': price,
                        'value': cost
                    }
                
                trade_record = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'amount': amount,
                    'price': price,
                    'cost': cost,
                    'balance_after': self.current_balance
                }
                self.trade_history.append(trade_record)
                
                logger.info(f"✅ BOUGHT {amount:.6f} {symbol} at ${price:.2f} (cost: ${cost:.2f})")
                logger.info(f"💰 Balance: ${self.current_balance:.2f}")
            else:
                logger.warning(f"❌ Insufficient balance for {symbol} purchase")
        
        elif action.upper() == 'SELL':
            if symbol in self.positions and self.positions[symbol]['amount'] >= amount:
                revenue = amount * price
                self.current_balance += revenue
                self.positions[symbol]['amount'] -= amount
                self.positions[symbol]['value'] -= amount * self.positions[symbol]['avg_price']
                
                if self.positions[symbol]['amount'] <= 0:
                    del self.positions[symbol]
                
                trade_record = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'amount': amount,
                    'price': price,
                    'revenue': revenue,
                    'balance_after': self.current_balance
                }
                self.trade_history.append(trade_record)
                
                logger.info(f"✅ SOLD {amount:.6f} {symbol} at ${price:.2f} (revenue: ${revenue:.2f})")
                logger.info(f"💰 Balance: ${self.current_balance:.2f}")
            else:
                logger.warning(f"❌ Insufficient {symbol} position for sale")
    
    def show_portfolio_status(self):
        """Muestra el estado del portfolio"""
        logger.info("=" * 50)
        logger.info("📊 PORTFOLIO STATUS")
        logger.info("=" * 50)
        logger.info(f"💰 Cash Balance: ${self.current_balance:,.2f}")
        
        total_portfolio_value = self.current_balance
        
        if self.positions:
            logger.info("📈 POSITIONS:")
            for symbol, pos in self.positions.items():
                current_price = self.get_mock_market_data(symbol)['price']
                current_value = pos['amount'] * current_price
                pnl = current_value - pos['value']
                pnl_pct = (pnl / pos['value']) * 100 if pos['value'] > 0 else 0
                
                total_portfolio_value += current_value
                
                logger.info(f"  {symbol}: {pos['amount']:.6f} @ ${pos['avg_price']:.2f}")
                logger.info(f"    Current: ${current_price:.2f} | Value: ${current_value:.2f}")
                logger.info(f"    P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        else:
            logger.info("📈 POSITIONS: None")
        
        total_pnl = total_portfolio_value - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        logger.info("=" * 50)
        logger.info(f"🎯 TOTAL PORTFOLIO VALUE: ${total_portfolio_value:,.2f}")
        logger.info(f"📊 TOTAL P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
        logger.info("=" * 50)
    
    async def run_paper_trading_demo(self, cycles: int = 3):
        """Ejecuta el demo de paper trading"""
        logger.info("🚀 Starting Paper Trading Demo")
        logger.info("=" * 50)
        
        # Inicializar sistema
        if not await self.initialize_system():
            logger.error("❌ Failed to initialize system. Exiting.")
            return
        
        logger.info("✅ System initialization complete!")
        logger.info("💹 Starting paper trading cycles...")
        
        for cycle in range(1, cycles + 1):
            logger.info("=" * 50)
            logger.info(f"🔄 CYCLE {cycle}/{cycles}")
            logger.info("=" * 50)
            
            # Analizar cada símbolo
            for symbol in self.symbols:
                analysis = await self.analyze_symbol(symbol)
                
                # Simular decisión de trading basada en análisis
                market_data = analysis['market_data']
                if cycle == 1 and symbol == 'BTC':
                    # Demo: comprar BTC en el primer ciclo
                    amount = 0.1  # 0.1 BTC
                    self.execute_paper_trade(symbol, 'BUY', amount, market_data['price'])
                elif cycle == 2 and symbol == 'ETH':
                    # Demo: comprar ETH en el segundo ciclo  
                    amount = 1.0  # 1 ETH
                    self.execute_paper_trade(symbol, 'BUY', amount, market_data['price'])
                
                await asyncio.sleep(1)  # Pausa entre análisis
            
            # Mostrar estado del portfolio
            self.show_portfolio_status()
            
            if cycle < cycles:
                logger.info("⏳ Waiting for next cycle...")
                await asyncio.sleep(3)  # Pausa entre ciclos
        
        logger.info("🎉 Paper Trading Demo completed!")
        
        # Resumen final
        logger.info("📋 FINAL SUMMARY:")
        logger.info(f"   Trades executed: {len(self.trade_history)}")
        logger.info(f"   Initial balance: ${self.initial_balance:,.2f}")
        final_value = self.current_balance + sum(
            pos['amount'] * self.get_mock_market_data(symbol)['price'] 
            for symbol, pos in self.positions.items()
        )
        logger.info(f"   Final portfolio value: ${final_value:,.2f}")
        logger.info(f"   Total return: ${final_value - self.initial_balance:+,.2f}")

async def main():
    """Función principal"""
    print("🤖 FenixTradingBot - Paper Trading Demo")
    print("🏛️ Heterogeneous Agent Architecture")
    print("💹 Simulated Trading Environment")
    print()
    
    demo = PaperTradingDemo()
    await demo.run_paper_trading_demo(cycles=3)

if __name__ == "__main__":
    asyncio.run(main())
