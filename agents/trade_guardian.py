# agents/trade_guardian.py
"""
TradeGuardianAgent: Agente de monitoreo dinámico de posiciones abiertas.
Monitorea el mercado periódicamente y consulta a los agentes principales para decidir si mantener, cerrar o ajustar una posición.
"""
from __future__ import annotations
from typing import Any, ClassVar, Dict, Optional, Literal
from datetime import datetime
import logging
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

class TradeGuardianDecision(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    action: Literal["HOLD", "CLOSE", "ADJUST_SL", "ADJUST_TP"]
    reasoning: str
    confidence: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class TradeGuardianAgent:
    name: ClassVar[str] = "TradeGuardianAgent"
    role: ClassVar[str] = "Dynamic Position Monitor and Guardian"
    goal: ClassVar[str] = (
        "Monitor open trades, consult all agents periodically, and decide whether to hold, close, or adjust the position to minimize loss strikes and maximize profit."
    )
    backstory: ClassVar[str] = (
        "A vigilant AI agent that protects the trading system from prolonged loss strikes by actively monitoring open positions and adapting to new market signals."
    )
    llm_model: str = "qwen2.5:7b-instruct-q5_k_m"
    last_decision: Optional[TradeGuardianDecision] = None
    max_open_minutes: int = 30  # trigger cierre si la posición lleva mucho tiempo
    drawdown_threshold: float = 0.01  # 1% drawdown trigger

    def run(self, position: Dict[str, Any],
            tech_analysis: Any,
            qabba_analysis: Any,
            sentiment_analysis: Any,
            visual_analysis: Optional[Any] = None) -> TradeGuardianDecision:
        signals = {
            'technical': getattr(tech_analysis, 'signal', None),
            'qabba': getattr(qabba_analysis, 'qabba_signal', None),
            'sentiment': getattr(sentiment_analysis, 'overall_sentiment', None),
            'visual': getattr(visual_analysis, 'overall_visual_assessment', None) if visual_analysis else None
        }
        open_direction = position.get('direction')  # 'BUY' o 'SELL'
        close_votes = 0
        for agent, signal in signals.items():
            if open_direction == 'BUY' and signal in ['SELL', 'NEGATIVE', 'BEARISH', 'SELL_QABBA']:
                close_votes += 1
            elif open_direction == 'SELL' and signal in ['BUY', 'POSITIVE', 'BULLISH', 'BUY_QABBA']:
                close_votes += 1
        # Trigger adicional: drawdown
        entry_price = position.get('entry_price_requested') or position.get('filled_price_actual')
        last_price = position.get('last_price') or position.get('current_price')
        if entry_price and last_price:
            if open_direction == 'BUY' and (last_price < entry_price * (1 - self.drawdown_threshold)):
                return TradeGuardianDecision(action="CLOSE", reasoning="Drawdown > 1% para BUY.", confidence=0.95)
            if open_direction == 'SELL' and (last_price > entry_price * (1 + self.drawdown_threshold)):
                return TradeGuardianDecision(action="CLOSE", reasoning="Drawdown > 1% para SELL.", confidence=0.95)
        # Trigger adicional: tiempo máximo abierto
        entry_time = position.get('entry_timestamp_utc')
        if entry_time:
            try:
                from datetime import datetime, timezone
                entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                minutes_open = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60
                if minutes_open > self.max_open_minutes:
                    return TradeGuardianDecision(action="CLOSE", reasoning=f"Posición abierta > {self.max_open_minutes} minutos.", confidence=0.9)
            except Exception:
                pass
        # Trigger: SL muy cercano
        sl_price = position.get('sl_price_set')
        if open_direction == 'BUY' and sl_price and last_price and entry_price is not None and (last_price - sl_price) < (entry_price * 0.002):
            return TradeGuardianDecision(action="CLOSE", reasoning="SL muy cercano al precio actual para BUY.", confidence=0.85)
        if open_direction == 'SELL' and sl_price and last_price and entry_price is not None and (sl_price - last_price) < (entry_price * 0.002):
            return TradeGuardianDecision(action="CLOSE", reasoning="SL muy cercano al precio actual para SELL.", confidence=0.85)
        # Trigger: tiempo abierto sin beneficio
        if entry_price and last_price and abs(last_price - entry_price) < (entry_price * 0.001):
            if entry_time:
                from datetime import datetime, timezone
                entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                minutes_open = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60
                if minutes_open > 10:
                    return TradeGuardianDecision(action="CLOSE", reasoning="Posición sin beneficio tras 10 minutos.", confidence=0.8)
        if close_votes >= 2:
            decision = TradeGuardianDecision(action="CLOSE", reasoning=f"Al menos {close_votes} agentes cambiaron de señal respecto a la apertura.", confidence=0.9)
        else:
            decision = TradeGuardianDecision(action="HOLD", reasoning="No hay suficiente evidencia para cerrar la posición.", confidence=0.7)
        self.last_decision = decision
        logger.info(f"[TradeGuardian] Decisión: {decision.action} | Confianza: {decision.confidence:.2f} | Razonamiento: {decision.reasoning}")
        return decision
