"""
FenixTradingBot - Sistema Completo de Paper Trading
Integra simulador de órdenes, datos de mercado y agentes para paper trading realista
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
from pathlib import Path

# Imports del sistema existente
from config.config_loader import APP_CONFIG
from memory.trade_memory import TradeMemory
from agents.sentiment_enhanced import EnhancedSentimentAnalyst, SentimentOutput
from agents.technical_v_enhanced_fixed import EnhancedTechnicalAnalyst, EnhancedTechnicalAnalysisOutput
from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent, EnhancedVisualChartAnalysisOutput
from agents.decision import EnhancedDecisionAgent, FinalDecisionOutput
from agents.risk import AdvancedRiskManager, RiskAssessment, OrderDetails, PortfolioState, RiskParameters
from agents.QABBAValidatorAgent import EnhancedQABBAAgent, QABBAAnalysisOutput

# Imports del sistema de paper trading
from paper_trading.order_simulator import BinanceOrderSimulator, OrderType, OrderStatus
from paper_trading.market_simulator import MarketDataSimulator, Kline

logger = logging.getLogger(__name__)

class PaperTradingSystem:
    """
    Sistema completo de paper trading que replica el comportamiento del live trading
    pero usando simuladores realistas en lugar de órdenes reales
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Componentes del sistema
        self.order_simulator = BinanceOrderSimulator()
        self.market_simulator = MarketDataSimulator()
        self.trade_memory = TradeMemory()
        
        # Agentes (se inicializarán después)
        self.sentiment_agent: Optional[EnhancedSentimentAnalyst] = None
        self.technical_agent: Optional[EnhancedTechnicalAnalyst] = None
        self.visual_agent: Optional[EnhancedVisualAnalystAgent] = None
        self.qabba_agent: Optional[EnhancedQABBAAgent] = None
        self.decision_agent: Optional[EnhancedDecisionAgent] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        
        # Estado del trading
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.is_trading_active = False
        self.last_trade_time = datetime.now(timezone.utc)
        
        # Configuración
        self.symbols = APP_CONFIG.trading.symbol.split(",") if "," in APP_CONFIG.trading.symbol else [APP_CONFIG.trading.symbol]
        self.timeframe = APP_CONFIG.trading.timeframe
        self.cooldown_seconds = APP_CONFIG.trading.trade_cooldown_after_close_seconds
        
        # Archivos de log
        self.trade_log_path = Path("logs/paper_trading_trades.jsonl")
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎭 PaperTradingSystem initialized")
        logger.info(f"   Initial balance: ${initial_balance:,.2f}")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Timeframe: {self.timeframe}")
    
    async def initialize(self) -> bool:
        """Inicializa todos los componentes del sistema"""
        try:
            logger.info("🔧 Initializing paper trading system...")
            
            # 1. Inicializar agentes
            logger.info("   Initializing agents...")
            self.sentiment_agent = EnhancedSentimentAnalyst(agent_type='sentiment')
            self.technical_agent = EnhancedTechnicalAnalyst(
                agent_type='technical',
                sequence_length_llm4fts=20
            )
            self.visual_agent = EnhancedVisualAnalystAgent(
                agent_type='visual',
                save_charts_to_disk=APP_CONFIG.tools.chart_generator.save_charts_to_disk
            )
            self.qabba_agent = EnhancedQABBAAgent(agent_type='qabba')
            self.decision_agent = EnhancedDecisionAgent()
            
            # 2. Inicializar risk manager
            logger.info("   Initializing risk manager...")
            initial_risk_params = RiskParameters(**APP_CONFIG.risk_management.model_dump())
            self.risk_manager = AdvancedRiskManager(
                symbol_tick_size=0.01,  # Se actualizará con datos reales
                symbol_step_size=0.001,
                min_notional=5.0,
                initial_risk_params=initial_risk_params,
                portfolio_state_file="logs/paper_trading_portfolio.json"
            )
            
            # 3. Inicializar simuladores
            logger.info("   Initializing market data simulator...")
            await self.market_simulator.start_streaming(self.symbols, self.timeframe)
            
            # Suscribirse a updates de precio
            self.market_simulator.subscribe_to_price_updates(self._on_price_update)
            
            logger.info("✅ Paper trading system initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize paper trading system: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def start_trading(self):
        """Inicia el sistema de paper trading"""
        if not all([self.sentiment_agent, self.technical_agent, self.visual_agent, 
                   self.qabba_agent, self.decision_agent, self.risk_manager]):
            logger.error("❌ Cannot start trading: system not properly initialized")
            return
        
        self.is_trading_active = True
        logger.info("🚀 Paper trading started!")
        
        # Iniciar bucle principal de trading
        await self._trading_loop()
    
    async def _trading_loop(self):
        """Bucle principal de análisis y trading"""
        while self.is_trading_active:
            try:
                # Verificar cooldown
                time_since_last_trade = (datetime.now(timezone.utc) - self.last_trade_time).total_seconds()
                if time_since_last_trade < self.cooldown_seconds:
                    await asyncio.sleep(5)
                    continue
                
                # Procesar cada símbolo
                for symbol in self.symbols:
                    await self._analyze_and_trade_symbol(symbol)
                
                # Actualizar posiciones activas
                await self._update_active_positions()
                
                # Actualizar balance basado en precio actual
                await self._update_portfolio_value()
                
                # Esperar antes del siguiente ciclo
                await asyncio.sleep(10)  # Análisis cada 10 segundos
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)  # Pausa más larga en caso de error
    
    async def _analyze_and_trade_symbol(self, symbol: str):
        """Analiza un símbolo y ejecuta trades si es necesario"""
        try:
            # Obtener datos de mercado
            current_price = self.market_simulator.get_current_price(symbol)
            if not current_price:
                logger.warning(f"No current price available for {symbol}")
                return
            
            recent_klines = self.market_simulator.get_recent_klines(symbol, 100)
            if len(recent_klines) < 50:
                logger.info(f"Insufficient historical data for {symbol} ({len(recent_klines)} klines)")
                return
            
            # Verificar si ya hay posición activa
            if symbol in self.active_positions:
                logger.debug(f"Position already active for {symbol}, skipping analysis")
                return
            
            logger.info(f"🔍 Analyzing {symbol} @ ${current_price:.6f}")
            
            # Preparar datos técnicos
            close_prices = [k.close_price for k in recent_klines]
            high_prices = [k.high_price for k in recent_klines]
            low_prices = [k.low_price for k in recent_klines]
            volumes = [k.volume for k in recent_klines]
            
            # Simulamos métricas técnicas (en producción vendrían de technical_tools)
            tech_metrics = {
                'last_price': current_price,
                'sma_20': sum(close_prices[-20:]) / 20 if len(close_prices) >= 20 else current_price,
                'rsi': 50.0,  # Placeholder
                'atr': abs(sum(high_prices[-14:]) - sum(low_prices[-14:])) / 14 if len(high_prices) >= 14 else 0.01
            }
            
            # 1. Análisis de sentimiento
            sentiment_result: SentimentOutput = self.sentiment_agent.run()
            logger.info(f"   Sentiment: {sentiment_result.overall_sentiment} (conf: {sentiment_result.confidence_score:.2f})")
            
            # 2. Análisis técnico
            technical_result: EnhancedTechnicalAnalysisOutput = self.technical_agent.run(
                current_tech_metrics=tech_metrics,
                indicator_sequences={},  # Placeholder
                sentiment_label=sentiment_result.overall_sentiment,
                symbol_tick_size=0.01
            )
            logger.info(f"   Technical: {technical_result.signal} (conf: {technical_result.confidence_level})")
            
            # 3. Análisis visual (simplificado para paper trading)
            visual_result = EnhancedVisualChartAnalysisOutput(
                overall_visual_assessment="NEUTRAL",
                reasoning="Paper trading visual analysis placeholder",
                chart_timeframe_analyzed=self.timeframe,
                pattern_clarity_score=0.5
            )
            
            # 4. Validación QABBA
            qabba_result: QABBAAnalysisOutput = self.qabba_agent.get_qabba_analysis(
                tech_metrics=tech_metrics,
                price_data_sequence=close_prices
            )
            logger.info(f"   QABBA: {qabba_result.qabba_signal} (conf: {qabba_result.qabba_confidence:.2f})")
            
            # 5. Decisión final
            final_decision: FinalDecisionOutput = self.decision_agent.run(
                sentiment_analysis=sentiment_result,
                numerical_technical_analysis=technical_result,
                visual_technical_analysis=visual_result,
                qabba_validation_analysis=qabba_result,
                current_tech_metrics=tech_metrics
            )
            logger.info(f"   Decision: {final_decision.final_decision} (conf: {final_decision.confidence_in_decision})")
            
            # 6. Evaluación de riesgo
            confidence_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}
            decision_conf_numeric = confidence_map.get(final_decision.confidence_in_decision, 0.5)
            
            risk_assessment: RiskAssessment = self.risk_manager.run(
                proposal_decision=final_decision.final_decision,
                current_balance=self.current_balance,
                tech_metrics=tech_metrics,
                decision_confidence=decision_conf_numeric,
                active_positions_list=list(self.active_positions.values())
            )
            logger.info(f"   Risk: {risk_assessment.verdict} - {risk_assessment.reason}")
            
            # 7. Ejecutar trade si está aprobado
            if risk_assessment.verdict in ["APPROVE", "APPROVE_REDUCED"] and risk_assessment.order_details:
                if final_decision.final_decision in ["BUY", "SELL"]:
                    await self._execute_paper_trade(symbol, final_decision.final_decision, risk_assessment.order_details)
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    async def _execute_paper_trade(self, symbol: str, direction: str, order_details: OrderDetails):
        """Ejecuta un trade en el simulador"""
        try:
            logger.info(f"🎯 Executing {direction} trade for {symbol}")
            
            # 1. Colocar orden de entrada (market order)
            entry_result = await self.order_simulator.place_order(
                symbol=symbol,
                side=direction,
                order_type=OrderType.MARKET,
                quantity=order_details.position_size_contracts
            )
            
            if entry_result.get("status") == "REJECTED":
                logger.error(f"❌ Entry order rejected: {entry_result.get('message')}")
                return
            
            entry_order_id = entry_result["orderId"]
            logger.info(f"✅ Entry order placed: {entry_order_id}")
            
            # Simular fill inmediato para market order
            await asyncio.sleep(0.1)
            
            # 2. Verificar fill de entrada
            entry_order = await self.order_simulator.get_order(entry_order_id)
            if not entry_order or entry_order["status"] != "FILLED":
                logger.error(f"❌ Entry order not filled: {entry_order}")
                return
            
            fill_price = float(entry_order["avgPrice"])
            fill_quantity = float(entry_order["executedQty"])
            
            # 3. Colocar órdenes de stop loss y take profit
            sl_side = "SELL" if direction == "BUY" else "BUY"
            
            # Stop Loss
            sl_result = await self.order_simulator.place_order(
                symbol=symbol,
                side=sl_side,
                order_type=OrderType.STOP_MARKET,
                quantity=fill_quantity,
                stop_price=order_details.stop_loss_price,
                reduce_only=True
            )
            
            # Take Profit
            tp_result = await self.order_simulator.place_order(
                symbol=symbol,
                side=sl_side,
                order_type=OrderType.TAKE_PROFIT_MARKET,
                quantity=fill_quantity,
                stop_price=order_details.take_profit_price,
                reduce_only=True
            )
            
            # 4. Registrar posición activa
            position = {
                "symbol": symbol,
                "side": direction,
                "entry_price": fill_price,
                "quantity": fill_quantity,
                "entry_order_id": entry_order_id,
                "sl_order_id": sl_result.get("orderId"),
                "tp_order_id": tp_result.get("orderId"),
                "sl_price": order_details.stop_loss_price,
                "tp_price": order_details.take_profit_price,
                "entry_time": datetime.now(timezone.utc),
                "status": "ACTIVE"
            }
            
            self.active_positions[symbol] = position
            
            # 5. Actualizar balance
            trade_cost = fill_quantity * fill_price
            if direction == "BUY":
                self.current_balance -= trade_cost
            else:
                self.current_balance += trade_cost
            
            # 6. Log del trade
            await self._log_trade_event("POSITION_OPENED", position)
            
            logger.info(f"✅ Position opened: {direction} {fill_quantity:.6f} {symbol} @ ${fill_price:.6f}")
            logger.info(f"   SL: ${order_details.stop_loss_price:.6f} | TP: ${order_details.take_profit_price:.6f}")
            logger.info(f"   Balance: ${self.current_balance:.2f}")
            
            self.last_trade_time = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
    
    async def _update_active_positions(self):
        """Actualiza el estado de las posiciones activas"""
        positions_to_close = []
        
        for symbol, position in self.active_positions.items():
            if position["status"] != "ACTIVE":
                continue
            
            try:
                # Verificar órdenes SL/TP
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                sl_order = await self.order_simulator.get_order(sl_order_id) if sl_order_id else None
                tp_order = await self.order_simulator.get_order(tp_order_id) if tp_order_id else None
                
                # Verificar si alguna orden fue ejecutada
                if sl_order and sl_order["status"] == "FILLED":
                    await self._close_position(symbol, "STOP_LOSS", float(sl_order["avgPrice"]))
                    positions_to_close.append(symbol)
                elif tp_order and tp_order["status"] == "FILLED":
                    await self._close_position(symbol, "TAKE_PROFIT", float(tp_order["avgPrice"]))
                    positions_to_close.append(symbol)
                
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
        
        # Remover posiciones cerradas
        for symbol in positions_to_close:
            if symbol in self.active_positions:
                del self.active_positions[symbol]
    
    async def _close_position(self, symbol: str, reason: str, exit_price: float):
        """Cierra una posición y calcula P&L"""
        position = self.active_positions[symbol]
        
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        side = position["side"]
        
        # Calcular P&L
        if side == "BUY":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # Actualizar balance
        self.current_balance += pnl
        
        # Actualizar posición
        position.update({
            "status": "CLOSED",
            "exit_price": exit_price,
            "exit_time": datetime.now(timezone.utc),
            "close_reason": reason,
            "pnl": pnl
        })
        
        # Log del cierre
        await self._log_trade_event("POSITION_CLOSED", position)
        
        logger.info(f"🔒 Position closed: {symbol} - {reason}")
        logger.info(f"   Entry: ${entry_price:.6f} | Exit: ${exit_price:.6f}")
        logger.info(f"   P&L: ${pnl:+.2f} | Balance: ${self.current_balance:.2f}")
    
    async def _update_portfolio_value(self):
        """Actualiza el valor del portfolio con precios actuales"""
        total_value = self.current_balance
        
        for symbol, position in self.active_positions.items():
            if position["status"] != "ACTIVE":
                continue
            
            current_price = self.market_simulator.get_current_price(symbol)
            if not current_price:
                continue
            
            # Calcular valor unrealized
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            side = position["side"]
            
            if side == "BUY":
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity
            
            total_value += unrealized_pnl
            position["unrealized_pnl"] = unrealized_pnl
        
        # Log periódico del portfolio
        if hasattr(self, '_last_portfolio_log'):
            time_since_log = (datetime.now(timezone.utc) - self._last_portfolio_log).total_seconds()
            if time_since_log > 300:  # Log cada 5 minutos
                await self._log_portfolio_status(total_value)
                self._last_portfolio_log = datetime.now(timezone.utc)
        else:
            self._last_portfolio_log = datetime.now(timezone.utc)
    
    async def _on_price_update(self, symbol: str, kline: Kline):
        """Callback para actualizaciones de precio del market simulator"""
        # Actualizar precio en el order simulator
        self.order_simulator.update_market_price(symbol, kline.close_price)
        
        logger.debug(f"Price update: {symbol} @ ${kline.close_price:.6f}")
    
    async def _log_trade_event(self, event_type: str, data: Dict[str, Any]):
        """Log de eventos de trading"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        try:
            with open(self.trade_log_path, "a") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Error logging trade event: {e}")
    
    async def _log_portfolio_status(self, total_value: float):
        """Log del estado del portfolio"""
        total_return = total_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        logger.info("📊 PORTFOLIO STATUS")
        logger.info(f"   Cash: ${self.current_balance:.2f}")
        logger.info(f"   Total Value: ${total_value:.2f}")
        logger.info(f"   Total Return: ${total_return:+.2f} ({return_pct:+.2f}%)")
        logger.info(f"   Active Positions: {len(self.active_positions)}")
    
    def stop_trading(self):
        """Detiene el sistema de paper trading"""
        self.is_trading_active = False
        self.market_simulator.stop_streaming()
        logger.info("🛑 Paper trading stopped")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Resumen completo del portfolio"""
        total_value = self.current_balance
        unrealized_pnl = 0.0
        
        positions_summary = []
        for symbol, position in self.active_positions.items():
            if position["status"] == "ACTIVE":
                current_price = self.market_simulator.get_current_price(symbol)
                if current_price:
                    entry_price = position["entry_price"]
                    quantity = position["quantity"]
                    side = position["side"]
                    
                    if side == "BUY":
                        position_pnl = (current_price - entry_price) * quantity
                    else:
                        position_pnl = (entry_price - current_price) * quantity
                    
                    total_value += position_pnl
                    unrealized_pnl += position_pnl
                    
                    positions_summary.append({
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "unrealized_pnl": position_pnl
                    })
        
        total_return = total_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        return {
            "initial_balance": self.initial_balance,
            "current_cash": self.current_balance,
            "unrealized_pnl": unrealized_pnl,
            "total_portfolio_value": total_value,
            "total_return": total_return,
            "return_percentage": return_pct,
            "active_positions": positions_summary,
            "is_trading_active": self.is_trading_active
        }

# Función principal para usar el sistema
async def run_paper_trading(initial_balance: float = 10000.0, duration_minutes: int = 60):
    """Ejecuta el sistema de paper trading por un tiempo determinado"""
    system = PaperTradingSystem(initial_balance)
    
    try:
        # Inicializar sistema
        if not await system.initialize():
            logger.error("Failed to initialize paper trading system")
            return
        
        # Iniciar trading
        trading_task = asyncio.create_task(system.start_trading())
        
        # Ejecutar por el tiempo especificado
        await asyncio.sleep(duration_minutes * 60)
        
        # Detener trading
        system.stop_trading()
        
        # Cancelar task de trading
        trading_task.cancel()
        
        # Mostrar resumen final
        summary = system.get_portfolio_summary()
        logger.info("🎉 Paper Trading Complete!")
        logger.info(f"   Final Portfolio Value: ${summary['total_portfolio_value']:.2f}")
        logger.info(f"   Total Return: ${summary['total_return']:+.2f} ({summary['return_percentage']:+.2f}%)")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in paper trading: {e}")
        system.stop_trading()
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run_paper_trading(initial_balance=10000.0, duration_minutes=30))
