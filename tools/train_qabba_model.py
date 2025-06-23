# agents/risk.py
from __future__ import annotations

import logging
import math # Para cálculos como floor
from typing import Any, ClassVar, Dict, Tuple, List, Optional

# No se necesita ollama ni instructor aquí si es puramente basado en reglas
from crewai import Agent
from pydantic import BaseModel, Field, ConfigDict, field_validator

# Asumimos que FinalDecisionOutput se importa o define aquí si es necesario
# from agents.decision_v_phi3 import FinalDecisionOutput
# Por ahora, asumiremos que la 'proposal' es solo el string "BUY", "SELL", "HOLD"
# y la explicación viene por separado.

class RiskCalculationDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    entry_price: Optional[float] = None
    stop_loss_price: float
    take_profit_price: float
    position_size_contracts: float # o units, o $, dependiendo de tu exchange y cómo calcules
    risk_per_trade_pct: float
    potential_reward_usd: Optional[float] = None # Si calculas en USD
    potential_risk_usd: Optional[float] = None  # Si calculas en USD
    reward_risk_ratio: Optional[float] = None

class RiskManagerOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: Literal["APPROVE", "VETO"]
    reason: str
    order_details: Optional[RiskCalculationDetails] = None


class RiskManager(Agent): # Hereda de Agent si quieres que sea parte de un Crew de CrewAI
    """
    Risk Manager que valida propuestas de trading y calcula parámetros de orden
    basándose en reglas matemáticas y de gestión de riesgo. No usa LLM.
    """
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    @field_validator("tools", mode="before", check_fields=False)
    def validate_tools(cls, v: Any) -> Any:
        return v

    tools: ClassVar[List[Any]] = []
    name: ClassVar[str] = "RiskManager"
    role: ClassVar[str] = "Quantitative Risk Overseer and Trade Execution Planner"
    goal: ClassVar[str] = "Approve or veto trades based on predefined risk parameters, calculate optimal position size, Stop Loss, and Take Profit levels."
    backstory: ClassVar[str] = (
        "A meticulous and purely quantitative risk officer. It operates strictly on mathematical "
        "principles and predefined risk management rules to safeguard capital and optimize "
        "trade execution parameters. It does not rely on sentiment or subjective LLM opinions for its core logic."
    )

    # --- Parámetros de Riesgo Configurables ---
    # Estos podrían cargarse desde un archivo de configuración
    default_risk_per_trade_percentage: float = 0.02  # Ej: 2% del balance por operación
    min_reward_risk_ratio: float = 1.5             # Ej: Mínimo R:R de 1.5:1
    atr_sl_multiplier: float = 1.5                 # Ej: SL a 1.5 * ATR
    atr_tp_multiplier: float = 3.0                 # Ej: TP a 3.0 * ATR (para un R:R de 2:1 si SL es 1.5 ATR)
    
    # Necesitarás una forma de obtener el stepSize y tickSize para el símbolo actual
    # Esto normalmente vendría de la información del exchange
    # Por ahora, placeholders. Deberías pasarlos o obtenerlos dinámicamente.
    symbol_tick_size: float = 0.01 # Placeholder
    symbol_step_size: float = 0.001 # Placeholder


    def _round_price(self, price: float) -> float:
        """Redondea el precio al tickSize correcto."""
        if self.symbol_tick_size == 0: return price
        return round(round(price / self.symbol_tick_size) * self.symbol_tick_size, 8) # 8 decimales como ejemplo

    def _round_quantity(self, quantity: float) -> float:
        """Redondea la cantidad al stepSize correcto."""
        if self.symbol_step_size == 0: return quantity
        # Usar math.floor para no exceder el balance al redondear hacia arriba la cantidad
        return math.floor(quantity / self.symbol_step_size) * self.symbol_step_size


    def run(
        self,
        proposal_decision: Literal["BUY", "SELL", "HOLD"], # La decisión del DecisionAgent
        # proposal_reasoning: str, # El razonamiento combinado del DecisionAgent (para logging)
        current_balance: float,
        # current_drawdown_pct: float, # Podrías usarlo para ajustar el riesgo
        tech_metrics: Dict[str, Any], # Necesita 'last_price' y 'atr' (Average True Range)
        # Opcional: symbol_info para tick_size, step_size, min_notional etc.
        # symbol_info: Optional[Dict[str, Any]] = None
    ) -> RiskManagerOutput:
        """
        Valida la propuesta y calcula los parámetros de la orden.
        """
        
        # Actualizar tick_size y step_size si se pasan dinámicamente
        # if symbol_info:
        #     self.symbol_tick_size = float(symbol_info.get('tickSize', self.symbol_tick_size))
        #     self.symbol_step_size = float(symbol_info.get('stepSize', self.symbol_step_size))
            # min_notional = float(symbol_info.get('minNotional', 0))


        if proposal_decision == "HOLD":
            return RiskManagerOutput(verdict="VETO", reason="Proposal is HOLD, no trade action.")

        last_price = tech_metrics.get("last_price")
        atr = tech_metrics.get("atr") # Asegúrate que 'atr' esté en tech_metrics

        if last_price is None:
            return RiskManagerOutput(verdict="VETO", reason="Critical error: 'last_price' is missing from tech_metrics.")
        if atr is None or atr <= 0: # ATR debe ser positivo
            logging.warning("RiskManager: 'atr' is missing or invalid in tech_metrics. Cannot calculate dynamic SL/TP. Using fixed percentage (less ideal).")
            # Fallback a SL/TP porcentual si ATR no está disponible (NO RECOMENDADO PARA PRODUCCIÓN)
            sl_price_calculated = last_price * (1 - 0.01) if proposal_decision == "BUY" else last_price * (1 + 0.01)
            tp_price_calculated = last_price * (1 + 0.018) if proposal_decision == "BUY" else last_price * (1 - 0.018)
            atr_fallback_reason = "ATR missing, using fixed % SL/TP."
        else:
            atr_fallback_reason = ""
            if proposal_decision == "BUY":
                sl_price_calculated = last_price - (self.atr_sl_multiplier * atr)
                tp_price_calculated = last_price + (self.atr_tp_multiplier * atr)
            elif proposal_decision == "SELL":
                sl_price_calculated = last_price + (self.atr_sl_multiplier * atr)
                tp_price_calculated = last_price - (self.atr_tp_multiplier * atr)
            else: # No debería ocurrir si ya filtramos HOLD
                return RiskManagerOutput(verdict="VETO", reason="Invalid proposal decision for SL/TP calculation.")

        sl_price = self._round_price(sl_price_calculated)
        tp_price = self._round_price(tp_price_calculated)

        # Validar que SL y TP no sean ilógicos (ej. SL > entry para BUY)
        if proposal_decision == "BUY" and (sl_price >= last_price or tp_price <= last_price):
            return RiskManagerOutput(verdict="VETO", reason=f"Invalid SL/TP for BUY: Entry={last_price}, SL={sl_price}, TP={tp_price}. {atr_fallback_reason}")
        if proposal_decision == "SELL" and (sl_price <= last_price or tp_price >= last_price):
            return RiskManagerOutput(verdict="VETO", reason=f"Invalid SL/TP for SELL: Entry={last_price}, SL={sl_price}, TP={tp_price}. {atr_fallback_reason}")

        # Calcular Riesgo y Recompensa por unidad (o contrato)
        potential_risk_per_unit = abs(last_price - sl_price)
        potential_reward_per_unit = abs(tp_price - last_price)

        if potential_risk_per_unit == 0: # Evitar división por cero
             return RiskManagerOutput(verdict="VETO", reason=f"Potential risk is zero (entry and SL are too close or equal). Entry={last_price}, SL={sl_price}.")

        current_reward_risk_ratio = potential_reward_per_unit / potential_risk_per_unit

        # Veto si el R:R es muy bajo
        if current_reward_risk_ratio < self.min_reward_risk_ratio:
            reason = (
                f"Reward/Risk ratio ({current_reward_risk_ratio:.2f}) is below minimum required ({self.min_reward_risk_ratio:.2f}). "
                f"Potential Reward: {potential_reward_per_unit:.2f}, Potential Risk: {potential_risk_per_unit:.2f}. {atr_fallback_reason}"
            )
            return RiskManagerOutput(verdict="VETO", reason=reason)

        # Calcular Tamaño de Posición
        # capital_at_risk = current_balance * self.default_risk_per_trade_percentage
        # position_size_calculated = capital_at_risk / potential_risk_per_unit
        # position_size_final = self._round_quantity(position_size_calculated)
        
        # Simplificación para el tamaño de posición: asumir que queremos arriesgar X USD fijos o un % del capital
        # Para futuros, el tamaño de la posición se calcula diferente (valor del contrato, etc.)
        # Aquí un ejemplo MUY SIMPLIFICADO si operas con cantidad de "monedas" (ej. 0.1 BTC)
        # Y el riesgo se define por la distancia al SL.
        # Este cálculo de tamaño de posición DEBE ser revisado y adaptado a tu exchange y activo.
        # Ejemplo: Arriesgar X% del balance.
        amount_to_risk_usd = current_balance * self.default_risk_per_trade_percentage
        
        # Si potential_risk_per_unit es la pérdida en USD por 1 unidad del activo:
        position_size_units_calculated = amount_to_risk_usd / potential_risk_per_unit
        position_size_final_units = self._round_quantity(position_size_units_calculated)


        if position_size_final_units <= 0:
            return RiskManagerOutput(verdict="VETO", reason=f"Calculated position size is zero or negative ({position_size_final_units}). Check risk parameters or balance. Amount to risk: {amount_to_risk_usd}, Risk per unit: {potential_risk_per_unit}")

        # Validar contra MIN_NOTIONAL (si operas futuros, esto es crucial)
        # notional_value = last_price * position_size_final_units
        # if notional_value < min_notional:
        #     return RiskManagerOutput(verdict="VETO", reason=f"Notional value {notional_value:.2f} is below exchange minimum {min_notional:.2f}.")


        order_details_obj = RiskCalculationDetails(
            entry_price=last_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            position_size_contracts=position_size_final_units, # Renombrar a _units si es más apropiado
            risk_per_trade_pct=self.default_risk_per_trade_percentage,
            potential_reward_usd=potential_reward_per_unit * position_size_final_units,
            potential_risk_usd=potential_risk_per_unit * position_size_final_units,
            reward_risk_ratio=current_reward_risk_ratio
        )
        
        final_reason = (
            f"Trade Approved. R:R={current_reward_risk_ratio:.2f}. "
            f"Size={position_size_final_units}. SL={sl_price}, TP={tp_price}. {atr_fallback_reason}"
        )
        logging.info(f"RiskManager: {final_reason}")
        return RiskManagerOutput(verdict="APPROVE", reason=final_reason, order_details=order_details_obj)

