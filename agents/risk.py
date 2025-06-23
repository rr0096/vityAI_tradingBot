# agents/risk.py
# This file contains the primary implementation of AdvancedRiskManager
# used by live_trading.py.

from __future__ import annotations

import logging
import math
import json
import numpy as np
from typing import Any, ClassVar, Dict, Tuple, List, Optional, Literal, Deque as TypingDeque
from datetime import datetime, timedelta, timezone
from collections import deque
from pathlib import Path

from crewai import Agent
from pydantic import BaseModel, Field, ConfigDict, ValidationError, PrivateAttr, field_validator

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

import threading

class RiskParameters(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    base_risk_per_trade: float = Field(default=0.01, description="Base risk percentage per trade of available balance.")
    max_risk_per_trade: float = Field(default=0.02, description="Absolute maximum risk percentage per trade.")
    min_risk_per_trade: float = Field(default=0.005, description="Absolute minimum risk percentage per trade.")
    
    atr_sl_multiplier: float = Field(default=1.5, description="Multiplier for ATR to set Stop Loss distance.")
    atr_tp_multiplier: float = Field(default=2.0, description="Multiplier for ATR to set Take Profit distance (relative to SL distance or entry).")
    
    min_reward_risk_ratio: float = Field(default=1.5, description="Minimum acceptable Reward/Risk ratio for a trade.")
    target_reward_risk_ratio: float = Field(default=2.0, description="Target Reward/Risk ratio.")
    
    max_daily_loss_pct: float = Field(default=0.03, description="Maximum percentage of total balance allowed for daily loss.")
    max_consecutive_losses: int = Field(default=3, description="Maximum consecutive losses before pausing trading.")
    max_trades_per_day: int = Field(default=10, description="Maximum number of trades allowed per day.")
    
    volatility_adjustment_factor: float = Field(default=1.0, description="Factor to adjust risk based on market volatility.")
    performance_adjustment_factor: float = Field(default=1.0, description="Factor to adjust risk based on recent performance.")
    market_condition_factor: float = Field(default=1.0, description="Factor to adjust risk based on perceived market conditions.")
    time_of_day_factor: float = Field(default=1.0, description="Factor to adjust risk based on trading session.")
    confidence_adjustment_factor: float = Field(default=1.0, description="Factor to adjust risk based on decision confidence.")


class OrderDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    position_size_contracts: float
    position_size_usd: float
    risk_per_trade_pct: float
    risk_amount_usd: float
    potential_reward_usd: float
    reward_risk_ratio: float
    
    use_trailing_stop: bool = False
    trailing_stop_distance_pct: Optional[float] = Field(None, description="Trailing stop distance as percentage of entry price")
    trailing_stop_activation_pct: Optional[float] = Field(None, description="Price move percentage to activate trailing stop")
    
    max_holding_time_hours: Optional[int] = Field(None, description="Suggested maximum holding time for the trade")
    reduce_position_levels: Optional[List[Tuple[float, float]]] = Field(default_factory=list, description="List of (price_level, percentage_to_close)")
    risk_score_at_open: float
    confidence_adjusted_size: bool = False
    market_conditions_at_open: Dict[str, Any] = Field(default_factory=dict)


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    verdict: Literal["APPROVE", "VETO", "APPROVE_REDUCED"]
    reason: str
    risk_score: float
    order_details: Optional[OrderDetails] = None
    risk_factors_summary: Dict[str, float] = Field(default_factory=dict, description="Summary of key factors contributing to risk score")
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    position_size_reduction_pct: float = Field(default=0.0, description="Percentage by which original size was reduced, if any")
    final_risk_parameters: Optional[RiskParameters] = Field(None, description="The risk parameters used for this specific assessment")


class PortfolioState(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    total_balance: float = Field(default=10000.0)
    available_balance: float = Field(default=10000.0)
    open_positions: List[Dict[str, Any]] = Field(default_factory=list)
    total_exposure_usd: float = Field(default=0.0)
    total_exposure_pct: float = Field(default=0.0, description="Total exposure as percentage of total_balance")
    
    daily_pnl_usd: float = Field(default=0.0)
    daily_pnl_pct: float = Field(default=0.0)
    trades_today: int = Field(default=0)
    
    consecutive_losses: int = Field(default=0)
    equity_curve: List[float] = Field(default_factory=list)
    current_drawdown_pct: float = Field(default=0.0)
    max_drawdown_seen_pct: float = Field(default=0.0)

    win_rate_last_n: float = Field(default=0.50, description="Win rate of the last N trades")
    avg_pnl_last_n: float = Field(default=0.0, description="Average PnL of the last N trades")
    
    last_updated: Optional[datetime] = None # This will be correctly serialized by model_dump_json

# ============================================================================
# ADVANCED RISK MANAGER AGENT
# ============================================================================

class AdvancedRiskManager(Agent):
    name: ClassVar[str] = "AdvancedRiskManager"
    role: ClassVar[str] = "Quantitative Risk Manager with Dynamic Adjustment"
    goal: ClassVar[str] = (
        "Protect capital through sophisticated risk management while "
        "optimizing position sizing for maximum risk-adjusted returns, based on rules and dynamic factors."
    )
    backstory: ClassVar[str] = (
        "An advanced risk management system that combines traditional risk metrics "
        "with dynamic adjustments based on market conditions, portfolio state, and recent performance. "
        "It operates purely on quantitative rules and does not use LLMs for its core risk decisions."
    )
    
    _symbol_tick_size: float = PrivateAttr()
    _symbol_step_size: float = PrivateAttr()
    _min_notional_value: float = PrivateAttr()

    _base_risk_params: RiskParameters = PrivateAttr()
    _current_risk_params: RiskParameters = PrivateAttr()

    _portfolio_state_file: Path = PrivateAttr()
    _portfolio_state: PortfolioState = PrivateAttr()
    _portfolio_lock: threading.RLock = PrivateAttr() # For thread-safe _portfolio_state access
    
    _trade_history: TypingDeque[Dict[str, Any]] = PrivateAttr(default_factory=lambda: deque(maxlen=100))
    _market_analysis_cache: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _MARKET_CACHE_TTL_SECONDS: int = PrivateAttr(default=180)

    @field_validator("tools", mode="before", check_fields=False)
    @classmethod
    def _validate_agent_tools(cls, v: Any) -> Any:
        if v is None: return []
        if not isinstance(v, list):
            logger.warning("Tools attribute for AdvancedRiskManager was not a list, defaulting to empty list.")
            return []
        return v
    
    def __init__(
        self,
        symbol_tick_size: float,
        symbol_step_size: float,
        min_notional: float,
        initial_risk_params: Optional[RiskParameters] = None,
        portfolio_state_file: str = "portfolio_state.json",
        **kwargs
    ):
        agent_kwargs_for_super = {
            'name': kwargs.pop('name', AdvancedRiskManager.name),
            'role': kwargs.pop('role', AdvancedRiskManager.role),
            'goal': kwargs.pop('goal', AdvancedRiskManager.goal),
            'backstory': kwargs.pop('backstory', AdvancedRiskManager.backstory),
            **kwargs
        }
        if 'tools' not in agent_kwargs_for_super:
            agent_kwargs_for_super['tools'] = []

        super().__init__(**agent_kwargs_for_super)
        
        self._symbol_tick_size = symbol_tick_size
        self._symbol_step_size = symbol_step_size
        self._min_notional_value = min_notional

        self._base_risk_params = initial_risk_params.model_copy(deep=True) if initial_risk_params else RiskParameters()
        self._current_risk_params = self._base_risk_params.model_copy(deep=True)

        self._portfolio_state_file = Path(portfolio_state_file)
        self._portfolio_lock = threading.RLock() # Initialize lock
        self._portfolio_state = self._load_portfolio_state() # Load state after lock is initialized
        
        # Initialize private attributes for caching
        self._trade_history = deque(maxlen=100)
        self._market_analysis_cache = {}
        self._MARKET_CACHE_TTL_SECONDS = 180
        
        if not self._portfolio_state.equity_curve:
            self._portfolio_state.equity_curve.append(self._portfolio_state.total_balance)

        logger.info(
            f"[{self.name}] Initialized. Tick: {self._symbol_tick_size}, Step: {self._symbol_step_size}, MinNotional: {self._min_notional_value}. "
            f"Portfolio State File: {self._portfolio_state_file}. Initial Balance: ${self._portfolio_state.total_balance:.2f}"
        )

    def _round_price(self, price: float) -> float:
        if self._symbol_tick_size <= 1e-9:
            return round(price, 8)
        return round(round(price / self._symbol_tick_size) * self._symbol_tick_size, 8)

    def _round_quantity(self, quantity: float) -> float:
        if self._symbol_step_size <= 1e-9:
            return round(quantity, 8)
        if 0 < quantity < self._symbol_step_size:
            return self._symbol_step_size
        return math.floor(quantity / self._symbol_step_size) * self._symbol_step_size

    def _load_portfolio_state(self) -> PortfolioState:
        with self._portfolio_lock: # Acquire lock before accessing file
            if self._portfolio_state_file.exists() and self._portfolio_state_file.is_file():
                try:
                    with open(self._portfolio_state_file, 'r') as f:
                        data = json.load(f)
                    # Pydantic will handle datetime parsing from ISO string
                    loaded_state = PortfolioState(**data)
                    logger.info(f"[{self.name}] Portfolio state loaded successfully from {self._portfolio_state_file}")
                    return loaded_state
                except (json.JSONDecodeError, ValidationError, Exception) as e:
                    logger.error(f"[{self.name}] Error loading/parsing portfolio state from {self._portfolio_state_file}: {e}. Using default state.")
            else:
                logger.info(f"[{self.name}] Portfolio state file {self._portfolio_state_file} not found. Initializing with default state.")
            return PortfolioState()

    def _save_portfolio_state(self) -> None:
        with self._portfolio_lock: # Acquire lock before modifying and saving
            self._portfolio_state.last_updated = datetime.now(timezone.utc)
            try:
                self._portfolio_state_file.parent.mkdir(parents=True, exist_ok=True)
                # Use model_dump_json for correct datetime serialization
                json_data = self._portfolio_state.model_dump_json(indent=2, exclude_none=True)
                with open(self._portfolio_state_file, 'w') as f:
                    f.write(json_data)
                logger.debug(f"[{self.name}] Portfolio state saved to {self._portfolio_state_file}")
            except IOError as e:
                logger.error(f"[{self.name}] Error saving portfolio state to {self._portfolio_state_file}: {e}")
            except Exception as e:
                logger.error(f"[{self.name}] Unexpected error saving portfolio state: {e}", exc_info=True)

    def update_portfolio_on_trade_close(self, pnl_usd: float, symbol: str, side: str, entry_price: float, exit_price: float, quantity: float):
        with self._portfolio_lock: # Protect access to _portfolio_state
            self._portfolio_state.total_balance += pnl_usd
            self._portfolio_state.available_balance = self._portfolio_state.total_balance # Assuming full balance is available after close

            self._portfolio_state.daily_pnl_usd += pnl_usd
            if self._portfolio_state.total_balance > 1e-9:
                self._portfolio_state.daily_pnl_pct = (self._portfolio_state.daily_pnl_usd / self._portfolio_state.total_balance) * 100
            else:
                self._portfolio_state.daily_pnl_pct = 0.0

            if pnl_usd < 0:
                self._portfolio_state.consecutive_losses += 1
            elif pnl_usd > 0:
                self._portfolio_state.consecutive_losses = 0
            
            self._portfolio_state.equity_curve.append(self._portfolio_state.total_balance)
            if len(self._portfolio_state.equity_curve) > 2000: # Limit equity curve history
                self._portfolio_state.equity_curve = self._portfolio_state.equity_curve[-2000:]
            
            if self._portfolio_state.equity_curve:
                peak_equity = max(self._portfolio_state.equity_curve)
                current_equity = self._portfolio_state.equity_curve[-1]
                if peak_equity > 1e-9:
                    self._portfolio_state.current_drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
                else:
                     self._portfolio_state.current_drawdown_pct = 0.0
                self._portfolio_state.max_drawdown_seen_pct = max(self._portfolio_state.max_drawdown_seen_pct, self._portfolio_state.current_drawdown_pct)
            else:
                self._portfolio_state.current_drawdown_pct = 0.0

            self._trade_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol, "side": side, "entry": entry_price, "exit": exit_price, "qty": quantity, "pnl_usd": pnl_usd
            })
            self._update_short_term_performance_metrics()
            # _save_portfolio_state will be called by the main loop or after a batch of updates
            # to avoid too frequent writes if multiple updates happen quickly.
            # For now, let's keep it here for immediate persistence after a trade.
            self._save_portfolio_state_internal() # Call internal save that assumes lock is held

    def _save_portfolio_state_internal(self) -> None:
        # Internal version of save that assumes lock is already held
        # This is to be called by methods that already acquired the lock
        self._portfolio_state.last_updated = datetime.now(timezone.utc)
        try:
            self._portfolio_state_file.parent.mkdir(parents=True, exist_ok=True)
            json_data = self._portfolio_state.model_dump_json(indent=2, exclude_none=True)
            with open(self._portfolio_state_file, 'w') as f:
                f.write(json_data)
            logger.debug(f"[{self.name}] Portfolio state (internal save) to {self._portfolio_state_file}")
        except IOError as e:
            logger.error(f"[{self.name}] Error saving portfolio state (internal save) to {self._portfolio_state_file}: {e}")
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error saving portfolio state (internal save): {e}", exc_info=True)


    def _update_short_term_performance_metrics(self, last_n_trades: int = 20):
        # Assumes lock is held if called from a locked context
        if not self._trade_history:
            self._portfolio_state.win_rate_last_n = 0.50
            self._portfolio_state.avg_pnl_last_n = 0.0
            return

        recent_trades = list(self._trade_history)[-last_n_trades:]
        if not recent_trades:
            self._portfolio_state.win_rate_last_n = 0.50
            self._portfolio_state.avg_pnl_last_n = 0.0
            return

        wins = sum(1 for trade in recent_trades if trade.get('pnl_usd', 0) > 0)
        total_pnl = sum(trade.get('pnl_usd', 0) for trade in recent_trades)
        
        self._portfolio_state.win_rate_last_n = wins / len(recent_trades)
        self._portfolio_state.avg_pnl_last_n = total_pnl / len(recent_trades)
        logger.debug(f"[{self.name}] Updated short-term performance: Win Rate (last {len(recent_trades)}) = {self._portfolio_state.win_rate_last_n:.2%}, Avg PnL = ${self._portfolio_state.avg_pnl_last_n:.2f}")


    def run(
        self,
        proposal_decision: Literal["BUY", "SELL", "HOLD"],
        current_balance: float, # Balance passed from live_trading after fetching
        tech_metrics: Dict[str, Any],
        decision_confidence: Optional[float] = 0.7,
        market_depth_snapshot: Optional[Dict[str, Any]] = None,
        active_positions_list: Optional[List[Dict]] = None # Current open positions from exchange
    ) -> RiskAssessment:
        
        with self._portfolio_lock: # Ensure all reads and potential writes to _portfolio_state are safe
            self._update_portfolio_state(current_balance, active_positions_list) # Update internal state first
            
            if proposal_decision == "HOLD":
                return RiskAssessment(
                    verdict="VETO", 
                    reason="Proposal is HOLD, no trade action required by Risk Manager.", 
                    risk_score=100.0,
                    final_risk_parameters=self._current_risk_params
                )
            
            circuit_breaker_status = self._check_circuit_breakers() # Uses self._portfolio_state
            if circuit_breaker_status['triggered']:
                logger.warning(f"[{self.name}] VETO due to Circuit Breaker: {circuit_breaker_status['reason']}")
                return RiskAssessment(
                    verdict="VETO", 
                    reason=f"Circuit breaker: {circuit_breaker_status['reason']}",
                    risk_score=0.0, 
                    warnings=[circuit_breaker_status['reason']],
                    final_risk_parameters=self._current_risk_params
                )
            
            if not self._validate_critical_inputs(tech_metrics):
                return RiskAssessment(
                    verdict="VETO", 
                    reason="Critical technical metrics missing or invalid for risk assessment.", 
                    risk_score=10.0,
                    final_risk_parameters=self._current_risk_params
                )
            
            market_analysis = self._analyze_market_conditions(tech_metrics, market_depth_snapshot)
            self._current_risk_params = self._adjust_risk_parameters_dynamically(market_analysis, decision_confidence or 0.7)
            
            sl_tp_levels = self._calculate_sl_tp_levels(
                proposal_decision, tech_metrics, self._current_risk_params, market_analysis )
            if sl_tp_levels.get("error"):
                return RiskAssessment(
                    verdict="VETO", 
                    reason=sl_tp_levels["error"], 
                    risk_score=15.0,
                    final_risk_parameters=self._current_risk_params
                )

            position_sizing_details = self._calculate_optimal_position_size(
                tech_metrics['last_price'], sl_tp_levels['sl_price'], self._current_risk_params, market_analysis
            ) # Uses self._portfolio_state.available_balance
            if position_sizing_details['size_contracts'] <= 1e-9:
                reason = position_sizing_details.get('reason', "Calculated position size is zero or too small.")
                logger.warning(f"[{self.name}] VETO: {reason}")
                return RiskAssessment(
                    verdict="VETO", 
                    reason=reason, 
                    risk_score=20.0,
                    final_risk_parameters=self._current_risk_params
                )

            risk_evaluation = self._evaluate_trade_risk_profile(
                position_sizing_details, sl_tp_levels, market_analysis, tech_metrics, proposal_decision
            ) # Uses self._portfolio_state.current_drawdown_pct
            
            verdict: Literal["APPROVE", "VETO", "APPROVE_REDUCED"] = "VETO"
            reason_for_verdict: str = risk_evaluation['main_risk_factor'] if risk_evaluation['risk_score'] < 50 else "Risk deemed acceptable."
            final_position_sizing = position_sizing_details.copy()
            position_size_reduction_pct = 0.0

            if risk_evaluation['risk_score'] < 30:
                verdict = "VETO"
                reason_for_verdict = f"High risk score ({risk_evaluation['risk_score']:.0f}/100). Main factor: {risk_evaluation['main_risk_factor']}."
            elif risk_evaluation['risk_score'] < 50:
                original_size = final_position_sizing['size_contracts']
                reduction_factor = 0.5
                reduced_size = self._round_quantity(original_size * (1 - reduction_factor))
                reduced_notional = reduced_size * tech_metrics['last_price']

                if reduced_size > 0 and reduced_notional >= self._min_notional_value:
                    verdict = "APPROVE_REDUCED"
                    final_position_sizing['size_contracts'] = reduced_size
                    final_position_sizing['notional_usd'] = reduced_notional # Typo fixed: was 'notional_usd'
                    risk_per_unit = abs(tech_metrics['last_price'] - sl_tp_levels['sl_price'])
                    final_position_sizing['risk_usd'] = risk_per_unit * reduced_size
                    reason_for_verdict = f"Moderate-high risk ({risk_evaluation['risk_score']:.0f}/100). Size reduced by {reduction_factor*100:.0f}%."
                    position_size_reduction_pct = reduction_factor * 100
                    logger.info(f"[{self.name}] Position size reduced due to risk score. Original: {original_size}, New: {reduced_size}")
                else:
                    verdict = "VETO"
                    reason_for_verdict = f"Moderate-high risk ({risk_evaluation['risk_score']:.0f}/100). Size reduction not feasible (min_notional or zero size)."
            else:
                verdict = "APPROVE"
                reason_for_verdict = f"Risk acceptable (Score: {risk_evaluation['risk_score']:.1f}/100)."
            
            order_details_obj: Optional[OrderDetails] = None
            if verdict in ["APPROVE", "APPROVE_REDUCED"]:
                order_details_obj = self._build_order_details_output(
                    entry_price=tech_metrics['last_price'],
                    sl_price=sl_tp_levels['sl_price'],
                    tp_price=sl_tp_levels['tp_price'],
                    position_size_contracts=final_position_sizing['size_contracts'],
                    position_size_usd=final_position_sizing['notional_usd'], # Typo fixed: was 'notional_usd'
                    risk_params_used=self._current_risk_params,
                    reward_risk_ratio=sl_tp_levels['actual_rr'],
                    risk_score_at_open=risk_evaluation['risk_score'],
                    market_conditions_at_open=market_analysis,
                    sl_tp_details=sl_tp_levels
                )
            
            final_assessment = RiskAssessment(
                verdict=verdict,
                reason=reason_for_verdict,
                risk_score=risk_evaluation['risk_score'],
                order_details=order_details_obj,
                risk_factors_summary=risk_evaluation['factors_contribution'],
                warnings=risk_evaluation.get('warnings_list', []),
                suggestions=self._generate_risk_suggestions(risk_evaluation, market_analysis),
                position_size_reduction_pct=position_size_reduction_pct,
                final_risk_parameters=self._current_risk_params.model_copy(deep=True)
            )
            
            self._log_risk_decision(final_assessment, tech_metrics.get('last_price', 0.0))
            # Save state after all updates within the locked block
            self._save_portfolio_state_internal()
            return final_assessment

    def _check_circuit_breakers(self) -> Dict[str, Any]:
        # Assumes _portfolio_state is accessed under lock by caller
        if not self._portfolio_state:
            logger.error(f"[{self.name}] Portfolio state not available for circuit breaker check.")
            return {'triggered': True, 'reason': "Portfolio state unavailable."}

        params = self._base_risk_params # Use base params for fixed limits

        # REACTIVADO: CIRCUIT BREAKERS ACTIVOS
        # Check daily loss percentage
        if self._portfolio_state.daily_pnl_usd < 0 and \
           abs(self._portfolio_state.daily_pnl_usd) >= (self._portfolio_state.total_balance * params.max_daily_loss_pct):
            return {'triggered': True, 'reason': f"Daily loss limit hit ({self._portfolio_state.daily_pnl_usd:.2f} USD is >= {params.max_daily_loss_pct*100:.2f}% of balance)."}

        # Check consecutive losses
        if self._portfolio_state.consecutive_losses >= params.max_consecutive_losses:
            return {'triggered': True, 'reason': f"Max consecutive losses ({self._portfolio_state.consecutive_losses}) reached limit ({params.max_consecutive_losses})."}

        # Check max trades per day
        if self._portfolio_state.trades_today >= params.max_trades_per_day:
            return {'triggered': True, 'reason': f"Max trades per day ({self._portfolio_state.trades_today}) reached limit ({params.max_trades_per_day})."}

        # Drawdown check
        if self._portfolio_state.current_drawdown_pct >= 15.0: # Example threshold from log
            return {'triggered': True, 'reason': f"Significant current drawdown ({self._portfolio_state.current_drawdown_pct:.2f}%) reached."}

        return {'triggered': False, 'reason': None}

    # ... (resto de los métodos _validate_critical_inputs, _analyze_market_conditions, etc. sin cambios significativos respecto a la versión anterior,
    #      asegurándose de que cualquier acceso a self._portfolio_state esté dentro de un contexto que ya tiene el lock, o adquiriéndolo si es necesario)

    def _update_portfolio_state(self, current_balance: Optional[float], active_positions: Optional[List[Dict[str, Any]]]):
        # Assumes lock is already held by the calling method (e.g., run)
        if current_balance is not None and isinstance(current_balance, (int, float)) and current_balance > 0:
            self._portfolio_state.total_balance = float(current_balance)
            
            if not self._portfolio_state.equity_curve or self._portfolio_state.equity_curve[-1] != self._portfolio_state.total_balance:
                self._portfolio_state.equity_curve.append(self._portfolio_state.total_balance)
                if len(self._portfolio_state.equity_curve) > 2000:
                    self._portfolio_state.equity_curve = self._portfolio_state.equity_curve[-2000:]

            if self._portfolio_state.equity_curve:
                peak_equity = max(self._portfolio_state.equity_curve)
                current_equity = self._portfolio_state.equity_curve[-1]
                if peak_equity > 1e-9:
                    self._portfolio_state.current_drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
                else:
                     self._portfolio_state.current_drawdown_pct = 0.0
                self._portfolio_state.max_drawdown_seen_pct = max(self._portfolio_state.max_drawdown_seen_pct, self._portfolio_state.current_drawdown_pct)
            else:
                self._portfolio_state.current_drawdown_pct = 0.0
        else:
            logger.warning(f"[{self.name}] Invalid current_balance ({current_balance}) for portfolio update. Using last known.")

        if active_positions is not None:
            self._portfolio_state.open_positions = active_positions
            total_notional_open = sum(
                pos.get('notional_value_usd', float(pos.get('current_price', 0)) * float(pos.get('contracts', 0)))
                for pos in active_positions
            )
            self._portfolio_state.total_exposure_usd = total_notional_open
            if self._portfolio_state.total_balance > 1e-9:
                self._portfolio_state.total_exposure_pct = (total_notional_open / self._portfolio_state.total_balance) * 100
            else:
                self._portfolio_state.total_exposure_pct = 0.0
        
        self._portfolio_state.available_balance = self._portfolio_state.total_balance - self._portfolio_state.total_exposure_usd
        if self._portfolio_state.available_balance < 0:
             logger.warning(f"[{self.name}] Calculated available balance is negative. Check exposure tracking.")
             self._portfolio_state.available_balance = 0

        self._portfolio_state.last_updated = datetime.now(timezone.utc) # This is fine, will be serialized by model_dump_json
        logger.debug(f"[{self.name}] Portfolio state updated. Balance: ${self._portfolio_state.total_balance:.2f}, Exposure: ${self._portfolio_state.total_exposure_usd:.2f} ({self._portfolio_state.total_exposure_pct:.2f}%)")

    def reset_daily_stats(self) -> None:
        with self._portfolio_lock:
            self._portfolio_state.daily_pnl_usd = 0.0
            self._portfolio_state.daily_pnl_pct = 0.0
            self._portfolio_state.trades_today = 0
            # No resetear consecutive_losses aquí, se maneja por trade.
            # No resetear drawdown aquí, es una métrica continua.
            logger.info(f"[{self.name}] Daily portfolio stats (PnL, trade count) have been reset.")
            self._save_portfolio_state_internal() # Save after reset

    # --- Métodos restantes ( _validate_critical_inputs, _analyze_market_conditions, _adjust_risk_parameters_dinamically,
    # _calculate_sl_tp_levels, _calculate_optimal_position_size, _evaluate_trade_risk_profile,
    # _build_order_details_output, _get_current_trading_session, _calculate_liquidity_score,
    # _generate_risk_suggestions, _log_risk_decision ) se mantienen como en la versión anterior,
    # asegurando que si acceden a self._portfolio_state, lo hagan en un contexto donde el lock ya fue adquirido
    # (ej. dentro del método `run` que ya tiene el `with self._portfolio_lock:`).
    # Si alguno de estos métodos es llamado desde fuera de `run` y modifica el estado, necesitaría su propio lock.
    # Por ahora, la estructura principal de `run` parece cubrir esto.

    def _validate_critical_inputs(self, tech_metrics: Dict[str, Any]) -> bool:
        required_numeric_positive = {
            "last_price": tech_metrics.get("last_price"),
            "atr": tech_metrics.get("atr")
        }
        for name, val in required_numeric_positive.items():
            if not isinstance(val, (int, float)) or val <= 0:
                logger.error(f"[{self.name}] Critical input validation failed: '{name}' is invalid or missing (value: {val}).")
                return False
        return True
    
    def _analyze_market_conditions(self, tech_metrics: Dict[str, Any], market_depth: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        price_key_num = tech_metrics.get('last_price', 0.0)
        price_key_str = f"{price_key_num:.4f}"
        cache_key = f"{price_key_str}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
        
        cached_data = self._market_analysis_cache.get(cache_key)
        if cached_data and (datetime.now(timezone.utc).timestamp() - cached_data['timestamp'] < self._MARKET_CACHE_TTL_SECONDS):
            logger.debug(f"[{self.name}] Using cached market analysis.")
            return cached_data['analysis']

        analysis: Dict[str, Any] = {
            'volatility_regime': 'normal', 'trend_strength': 'neutral',
            'liquidity_score': 0.5, 'risk_environment': 'moderate',
            'time_session': self._get_current_trading_session(),
            'atr_pct': 0.0
        }
        
        price = float(tech_metrics.get('last_price', 1.0))
        atr = float(tech_metrics.get('atr', price * 0.015))
        if price <= 1e-9: price = 1.0 # Avoid division by zero

        atr_pct = (atr / price) * 100 if price > 1e-9 else 0.0 # Added check
        analysis['atr_pct'] = round(atr_pct, 2)

        if atr_pct > 2.5: analysis['volatility_regime'] = 'high'
        elif atr_pct < 0.75: analysis['volatility_regime'] = 'low'
        
        adx = float(tech_metrics.get('adx', 20.0))
        if adx > 35: analysis['trend_strength'] = 'strong'
        elif adx > 20: analysis['trend_strength'] = 'moderate'
        else: analysis['trend_strength'] = 'weak'
            
        if market_depth: analysis['liquidity_score'] = self._calculate_liquidity_score(market_depth, price)
        
        risk_points = 0
        if analysis['volatility_regime'] == 'high': risk_points += 2
        if analysis['trend_strength'] == 'weak' and analysis['volatility_regime'] != 'low': risk_points += 1
        if analysis['liquidity_score'] < 0.3: risk_points += 2
        if analysis['time_session'] in ['rollover_pacific', 'pre_market_important_news']: risk_points +=1

        if risk_points >= 4: analysis['risk_environment'] = 'high'
        elif risk_points >= 2: analysis['risk_environment'] = 'elevated'
        else: analysis['risk_environment'] = 'normal'
        
        self._market_analysis_cache[cache_key] = {'timestamp': datetime.now(timezone.utc).timestamp(), 'analysis': analysis}
        logger.debug(f"[{self.name}] Market analysis updated: {analysis}")
        return analysis
    
    def _adjust_risk_parameters_dynamically(self, market_analysis: Dict[str, Any], decision_confidence: float) -> RiskParameters:
        # Assumes lock is held by caller if _portfolio_state is read here (it is via win_rate_last_n)
        adjusted_params = self._base_risk_params.model_copy(deep=True)
        
        win_rate = self._portfolio_state.win_rate_last_n
        if win_rate > 0.65: adjusted_params.performance_adjustment_factor = 1.2
        elif win_rate < 0.40: adjusted_params.performance_adjustment_factor = 0.8

        if market_analysis['volatility_regime'] == 'high':
            adjusted_params.volatility_adjustment_factor = 0.75
            adjusted_params.atr_sl_multiplier = max(self._base_risk_params.atr_sl_multiplier, 1.8)
            adjusted_params.min_reward_risk_ratio = max(self._base_risk_params.min_reward_risk_ratio, 1.8)
        elif market_analysis['volatility_regime'] == 'low':
            adjusted_params.volatility_adjustment_factor = 1.1
            adjusted_params.atr_sl_multiplier = min(self._base_risk_params.atr_sl_multiplier, 1.2)

        if market_analysis['risk_environment'] == 'high': adjusted_params.market_condition_factor = 0.7
        elif market_analysis['risk_environment'] == 'elevated': adjusted_params.market_condition_factor = 0.85
        
        session = market_analysis['time_session']
        if session in ['london_open_early', 'ny_session_active', 'london_ny_overlap']: adjusted_params.time_of_day_factor = 1.1
        elif session in ['asian_quiet', 'post_ny_close', 'rollover_pacific', 'sydney_early_asian']: adjusted_params.time_of_day_factor = 0.9

        if decision_confidence >= 0.85: adjusted_params.confidence_adjustment_factor = 1.2
        elif decision_confidence >= 0.65: adjusted_params.confidence_adjustment_factor = 1.0
        else: adjusted_params.confidence_adjustment_factor = 0.8

        total_adjustment = (
            adjusted_params.volatility_adjustment_factor *
            adjusted_params.performance_adjustment_factor *
            adjusted_params.market_condition_factor *
            adjusted_params.time_of_day_factor *
            adjusted_params.confidence_adjustment_factor
        )
        
        adjusted_params.base_risk_per_trade = self._base_risk_params.base_risk_per_trade * total_adjustment
        adjusted_params.base_risk_per_trade = max(
            self._base_risk_params.min_risk_per_trade,
            min(self._base_risk_params.max_risk_per_trade, adjusted_params.base_risk_per_trade)
        )
        logger.info(f"[{self.name}] Risk parameters adjusted. New base_risk_per_trade: {adjusted_params.base_risk_per_trade:.4%}")
        return adjusted_params

    def _calculate_sl_tp_levels(
        self,
        direction: Literal["BUY", "SELL"],
        tech_metrics: Dict[str, Any],
        risk_params: RiskParameters,
        market_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        entry_price = float(tech_metrics['last_price'])
        atr = float(tech_metrics.get('atr', entry_price * 0.015)) # Default ATR if not present
        
        if atr <= 1e-9: # Ensure ATR is positive
            atr = entry_price * 0.005 # Fallback ATR
            logger.warning(f"[{self.name}] ATR was zero/invalid, defaulted to {atr:.4f} for SL/TP.")

        sl_distance = atr * risk_params.atr_sl_multiplier
        # Ensure SL distance is at least a few ticks or a small percentage of price
        min_sl_dist_ticks = self._symbol_tick_size * 3 if self._symbol_tick_size > 0 else entry_price * 0.001
        sl_distance = max(sl_distance, min_sl_dist_ticks)

        base_sl_price = entry_price - sl_distance if direction == "BUY" else entry_price + sl_distance
        # TP calculation based on SL distance for a target R:R
        base_tp_price = entry_price + (sl_distance * risk_params.target_reward_risk_ratio) if direction == "BUY" \
                        else entry_price - (sl_distance * risk_params.target_reward_risk_ratio)

        sl_price = self._round_price(base_sl_price)
        tp_price = self._round_price(base_tp_price)
        
        # Ensure SL and TP are not on the wrong side of entry or too close after rounding
        min_price_move_from_entry = self._symbol_tick_size * 2 if self._symbol_tick_size > 0 else entry_price * 0.0002
        if direction == "BUY":
            if sl_price >= entry_price - min_price_move_from_entry:
                sl_price = self._round_price(entry_price - min_price_move_from_entry)
            if tp_price <= entry_price + min_price_move_from_entry: # TP must be above entry
                tp_price = self._round_price(entry_price + (max(sl_distance, min_price_move_from_entry) * risk_params.min_reward_risk_ratio))
        else: # SELL
            if sl_price <= entry_price + min_price_move_from_entry:
                sl_price = self._round_price(entry_price + min_price_move_from_entry)
            if tp_price >= entry_price - min_price_move_from_entry: # TP must be below entry
                tp_price = self._round_price(entry_price - (max(sl_distance, min_price_move_from_entry) * risk_params.min_reward_risk_ratio))

        final_risk_dist = abs(entry_price - sl_price)
        final_reward_dist = abs(tp_price - entry_price)

        if final_risk_dist < (self._symbol_tick_size if self._symbol_tick_size > 0 else 1e-9):
             logger.warning(f"[{self.name}] Final risk distance is too small or zero. SL={sl_price}, Entry={entry_price}")
             return {"error": f"Calculated risk distance too small. SL={sl_price}, Entry={entry_price}."}
        
        actual_rr = final_reward_dist / final_risk_dist if final_risk_dist > 1e-9 else 0.0
        
        # If actual R:R is below minimum after all adjustments, adjust TP to meet minimum R:R
        if actual_rr < risk_params.min_reward_risk_ratio * 0.95: # Use a small tolerance (0.95)
            logger.warning(f"[{self.name}] Calculated SL/TP R:R ({actual_rr:.2f}) < min ({risk_params.min_reward_risk_ratio}). Adjusting TP to meet min R:R.")
            if direction == "BUY":
                tp_price = self._round_price(entry_price + (final_risk_dist * risk_params.min_reward_risk_ratio))
            else: # SELL
                tp_price = self._round_price(entry_price - (final_risk_dist * risk_params.min_reward_risk_ratio))
            final_reward_dist = abs(tp_price - entry_price) # Recalculate reward distance
            actual_rr = final_reward_dist / final_risk_dist if final_risk_dist > 1e-9 else 0.0 # Recalculate R:R

        use_trailing = market_analysis.get('trend_strength') == 'strong' and market_analysis.get('volatility_regime') != 'high'
        trailing_params = {}
        if use_trailing and entry_price > 0 and final_risk_dist > 0 and atr > 0:
            trailing_params['trailing_stop_distance_pct'] = round( (atr * 0.75) / entry_price * 100, 2) if entry_price > 1e-9 else 0.0
            trailing_params['trailing_stop_activation_pct'] = round( (final_risk_dist * 0.5) / entry_price * 100, 2) if entry_price > 1e-9 else 0.0
        
        return {
            'sl_price': sl_price, 'tp_price': tp_price, 'actual_rr': actual_rr,
            'use_trailing': use_trailing, **trailing_params, 'error': None # Ensure error is None if successful
        }

    def _calculate_optimal_position_size(
        self, entry_price: float, sl_price: float, risk_params: RiskParameters, market_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Assumes lock is held by caller as it reads self._portfolio_state
        available_balance = self._portfolio_state.available_balance
        risk_per_unit = abs(entry_price - sl_price)

        if risk_per_unit < (self._symbol_tick_size if self._symbol_tick_size > 0 else 1e-9):
            return {'size_contracts': 0.0, 'notional_usd': 0.0, 'risk_usd': 0.0, 'reason': "Risk per unit is too small (SL too close to entry or invalid tick size)."}

        capital_to_risk_usd = available_balance * risk_params.base_risk_per_trade
        position_size_contracts = self._round_quantity(capital_to_risk_usd / risk_per_unit if risk_per_unit > 1e-9 else 0.0)
        notional_value_usd = position_size_contracts * entry_price

        # Max exposure check
        max_allowed_new_exposure = (self._portfolio_state.total_balance * 0.50) - self._portfolio_state.total_exposure_usd
        if notional_value_usd > max_allowed_new_exposure:
            if max_allowed_new_exposure > self._min_notional_value : # Check if we can even open a min notional trade with remaining exposure
                position_size_contracts = self._round_quantity(max_allowed_new_exposure / entry_price if entry_price > 1e-9 else 0.0)
                notional_value_usd = position_size_contracts * entry_price
                logger.info(f"[{self.name}] Position size capped by max portfolio exposure. New size: {position_size_contracts}")
            else: # Not enough room for even a min_notional trade
                return {'size_contracts': 0.0, 'notional_usd': 0.0, 'risk_usd': 0.0, 'reason': "Max portfolio exposure limit reached, not enough for min notional."}


        # Min notional check
        if notional_value_usd < self._min_notional_value and position_size_contracts > 0 : # If current size is too small but positive
            size_for_min_notional = self._round_quantity(self._min_notional_value / entry_price if entry_price > 1e-9 else 0.0)
            risk_if_min_notional = risk_per_unit * size_for_min_notional
            
            # Allow sizing up to min_notional if it doesn't grossly exceed the intended capital_to_risk_usd
            # (e.g., if risk_if_min_notional is within 150% of capital_to_risk_usd)
            # OR if the capital_to_risk_usd was already very small relative to what a min_notional trade would risk
            if risk_if_min_notional <= capital_to_risk_usd * 1.5 or \
               capital_to_risk_usd < ( (self._min_notional_value / entry_price if entry_price > 1e-9 else 0.0) * risk_per_unit * 0.2 ): # if target risk was < 20% of risk from min_notional
                position_size_contracts = size_for_min_notional
                notional_value_usd = position_size_contracts * entry_price
                logger.info(f"[{self.name}] Pos size adjusted UP to meet min_notional. New: {position_size_contracts}, Risk: ${risk_if_min_notional:.2f}")
            else:
                return {'size_contracts': 0.0, 'notional_usd': 0.0, 'risk_usd': 0.0, 'reason': f"Notional ${notional_value_usd:.2f} < min ${self._min_notional_value:.2f}. Adjusting UP exceeds risk budget significantly."}
        
        if position_size_contracts <= 1e-9: # Final check after all adjustments
             return {'size_contracts': 0.0, 'notional_usd': 0.0, 'risk_usd': 0.0, 'reason': "Final position size is zero."}

        actual_risk_taken_usd = risk_per_unit * position_size_contracts
        actual_risk_pct_of_balance = (actual_risk_taken_usd / available_balance) * 100 if available_balance > 1e-9 else 0.0

        # Final check: ensure actual risk percentage is within acceptable bounds (e.g., not exceeding max_risk_per_trade due to min_notional adjustments)
        if actual_risk_pct_of_balance / 100 > risk_params.max_risk_per_trade * 1.1: # Allow 10% overshoot for min_notional
            return {'size_contracts': 0.0, 'notional_usd': 0.0, 'risk_usd': 0.0, 'reason': f"Adjusted size for min_notional results in risk {actual_risk_pct_of_balance:.2f}% > max {risk_params.max_risk_per_trade*100:.2f}%."}

        return {
            'size_contracts': position_size_contracts,
            'notional_usd': notional_value_usd,
            'risk_usd': actual_risk_taken_usd,
            'risk_pct_of_balance': actual_risk_pct_of_balance,
            'reason': "Position size calculated successfully."
        }

    def _evaluate_trade_risk_profile(
        self, position_sizing: Dict[str, Any], sl_tp_levels: Dict[str, Any],
        market_analysis: Dict[str, Any], tech_metrics: Dict[str, Any],
        direction: Literal["BUY", "SELL"]
    ) -> Dict[str, Any]:
        # Assumes lock is held by caller as it reads self._portfolio_state
        risk_points = 0.0
        factors_contribution: Dict[str, float] = {}
        warnings_list: List[str] = []

        rr_ratio = sl_tp_levels.get('actual_rr', 0.0)
        if rr_ratio < self._current_risk_params.min_reward_risk_ratio:
            points = (self._current_risk_params.min_reward_risk_ratio - rr_ratio) * 40.0 # Heavier penalty for poor R:R
            risk_points += points
            factors_contribution['Poor R:R'] = points
            warnings_list.append(f"Low R:R ({rr_ratio:.2f} vs min {self._current_risk_params.min_reward_risk_ratio:.2f})")

        env = market_analysis.get('risk_environment', 'moderate')
        if env == 'high':
            risk_points += 30.0; factors_contribution['High Risk Market Env'] = 30.0
            warnings_list.append("Market environment: high risk.")
        elif env == 'elevated':
            risk_points += 15.0; factors_contribution['Elevated Risk Market Env'] = 15.0
            warnings_list.append("Market environment: elevated risk.")

        if self._portfolio_state.current_drawdown_pct > 8.0: # If in significant drawdown
            points = self._portfolio_state.current_drawdown_pct * 0.75 # Scale penalty with drawdown
            risk_points += points
            factors_contribution['Portfolio Drawdown'] = points
            warnings_list.append(f"Portfolio drawdown ({self._portfolio_state.current_drawdown_pct:.1f}%).")
        
        last_price = tech_metrics.get('last_price')
        ma50 = tech_metrics.get('ma50')
        if isinstance(last_price, (int,float)) and isinstance(ma50, (int,float)) and ma50 > 1e-9:
            price_ma50_dist_pct = abs(last_price - ma50) / ma50 * 100
            if price_ma50_dist_pct > 5.0: # Price is >5% away from MA50
                points = price_ma50_dist_pct * 0.5 # Penalty for overextension
                risk_points += points
                factors_contribution['Price Far From MA50'] = points
                warnings_list.append(f"Price is {price_ma50_dist_pct:.1f}% away from MA50 (potential overextension).")

        # Check for ATR percentage if it's very high or low
        atr_pct = market_analysis.get('atr_pct', 1.5) # Default to normal if not found
        if atr_pct > 3.5 : # Very high volatility
            risk_points += 10; factors_contribution['Extreme ATR Pct'] = 10.0
            warnings_list.append(f"ATR is very high ({atr_pct:.1f}%), indicating extreme volatility.")
        elif atr_pct < 0.4: # Very low volatility, breakouts might fail
            risk_points += 5; factors_contribution['Very Low ATR Pct'] = 5.0
            warnings_list.append(f"ATR is very low ({atr_pct:.1f}%), breakouts might lack follow-through.")


        risk_score = max(0.0, 100.0 - risk_points) # Score out of 100
        
        main_risk_factor = "Multiple factors"
        if factors_contribution:
            main_risk_factor = max(factors_contribution, key=lambda k: factors_contribution[k]) # type: ignore
            
        return {
            'risk_score': round(risk_score,1),
            'factors_contribution': factors_contribution,
            'warnings_list': warnings_list,
            'main_risk_factor': main_risk_factor
        }

    def _build_order_details_output(
        self, entry_price: float, sl_price: float, tp_price: float,
        position_size_contracts: float, position_size_usd: float,
        risk_params_used: RiskParameters, reward_risk_ratio: float,
        risk_score_at_open: float, market_conditions_at_open: Dict[str, Any],
        sl_tp_details: Dict[str, Any]
    ) -> OrderDetails:
        
        risk_amount_usd = abs(entry_price - sl_price) * position_size_contracts
        potential_reward_usd = abs(tp_price - entry_price) * position_size_contracts
        
        return OrderDetails(
            entry_price=entry_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            position_size_contracts=position_size_contracts,
            position_size_usd=position_size_usd,
            risk_per_trade_pct=risk_params_used.base_risk_per_trade * 100, # Store as percentage
            risk_amount_usd=risk_amount_usd,
            potential_reward_usd=potential_reward_usd,
            reward_risk_ratio=reward_risk_ratio,
            use_trailing_stop=sl_tp_details.get('use_trailing', False),
            trailing_stop_distance_pct=sl_tp_details.get('trailing_stop_distance_pct'),
            trailing_stop_activation_pct=sl_tp_details.get('trailing_stop_activation_pct'),
            max_holding_time_hours=12, # Example, could be dynamic
            risk_score_at_open=risk_score_at_open,
            market_conditions_at_open=market_conditions_at_open,
        )

    def _get_current_trading_session(self) -> str:
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        # Simplified session mapping
        if 0 <= hour < 1: return 'sydney_early_asian' # Sydney open, early Asia
        elif 1 <= hour < 7: return 'asian_active'      # Tokyo, HK, Singapore active
        elif 7 <= hour < 9: return 'asian_london_overlap' # Asia closing, London opening
        elif 9 <= hour < 12: return 'london_open_early' # London main session
        elif 12 <= hour < 13: return 'london_mid'      # London lunch / pre-NY
        elif 13 <= hour < 17: return 'london_ny_overlap' # London closing, NY active
        elif 17 <= hour < 21: return 'ny_session_active' # NY main session
        elif 21 <= hour < 22: return 'ny_close'        # NY close
        elif 22 <= hour < 24: return 'rollover_pacific' # Post-NY, Pacific session
        return 'unknown_session' # Should not happen

    def _calculate_liquidity_score(self, market_depth: Dict[str, Any], current_price: float) -> float:
        if not market_depth or not market_depth.get('bids') or not market_depth.get('asks') or \
           not isinstance(market_depth['bids'], list) or not isinstance(market_depth['asks'], list) or \
           not market_depth['bids'] or not market_depth['asks']:
            logger.debug(f"[{self.name}] Insufficient market depth data for liquidity score.")
            return 0.3 # Default to moderate-low if no data

        try:
            best_bid_price = float(market_depth['bids'][0][0])
            best_ask_price = float(market_depth['asks'][0][0])
            
            if best_bid_price <= 0 or best_ask_price <= 0 or best_ask_price <= best_bid_price or current_price <= 1e-9:
                logger.debug(f"[{self.name}] Invalid best bid/ask or current price for liquidity score. B:{best_bid_price} A:{best_ask_price} P:{current_price}")
                return 0.1 # Very low liquidity

            spread_pct = ((best_ask_price - best_bid_price) / current_price) * 100 if current_price > 1e-9 else float('inf')
            
            depth_range_pct = 0.2 # Look at depth within 0.2% of mid_price
            mid_price = (best_bid_price + best_ask_price) / 2
            
            bid_depth_qty = sum(float(qty_str) for price_str, qty_str in market_depth['bids'] if float(price_str) >= mid_price * (1 - depth_range_pct / 100))
            ask_depth_qty = sum(float(qty_str) for price_str, qty_str in market_depth['asks'] if float(price_str) <= mid_price * (1 + depth_range_pct / 100))

            # Score spread: Lower is better. 0.05% spread = good score. >0.15% = poor.
            spread_score = max(0, 1 - (spread_pct / 0.15))
            
            # Score depth: More USD depth is better. $1M depth = good score.
            depth_in_usd = (bid_depth_qty + ask_depth_qty) * current_price
            depth_score = min(1, depth_in_usd / 1000000) # Cap at 1 for $1M depth

            return round((spread_score * 0.5 + depth_score * 0.5), 2)
        except (IndexError, TypeError, ValueError) as e:
            logger.warning(f"[{self.name}] Error calculating liquidity score from market depth: {e}")
            return 0.2 # Low score on error
        
    def _generate_risk_suggestions(self, risk_evaluation: Dict[str, Any], market_analysis: Dict[str, Any]) -> List[str]:
        suggestions = []
        if risk_evaluation['risk_score'] < 40:
            suggestions.append("Overall risk is significantly elevated. Strongly consider waiting or minimal exposure.")
        if 'Poor R:R' in risk_evaluation['factors_contribution'] and risk_evaluation['factors_contribution']['Poor R:R'] > 15:
            suggestions.append("Reward/Risk ratio is notably unfavorable. Re-evaluate SL/TP targets or skip trade.")
        if market_analysis.get('risk_environment') == 'high':
            suggestions.append("Market environment currently assessed as high risk; extreme caution advised if proceeding.")
        if self._portfolio_state.current_drawdown_pct > 10.0:
            suggestions.append(f"Portfolio is in a drawdown of {self._portfolio_state.current_drawdown_pct:.1f}%. Consider reducing risk or pausing.")
        return suggestions

    def _log_risk_decision(self, assessment: RiskAssessment, last_price: float):
        logger.info(
            f"[{self.name}] Risk Assessment: Verdict={assessment.verdict}, Score={assessment.risk_score:.1f}, "
            f"Reason='{assessment.reason}'. Last Price=${last_price:.4f}."
        )
        if assessment.order_details:
            od = assessment.order_details
            logger.info(
                f"  Order Details: Size={od.position_size_contracts:.4f} ({od.position_size_usd:.2f} USD), "
                f"SL=${od.stop_loss_price:.4f}, TP=${od.take_profit_price:.4f}, R:R={od.reward_risk_ratio:.2f}, "
                f"Risked ${od.risk_amount_usd:.2f} ({od.risk_per_trade_pct:.2f}% of bal)"
            )
        if assessment.warnings:
            logger.warning(f"  Risk Warnings: {'; '.join(assessment.warnings)}")
        if assessment.suggestions:
            logger.info(f"  Risk Suggestions: {'; '.join(assessment.suggestions)}")
