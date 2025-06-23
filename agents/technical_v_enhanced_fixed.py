# agents/technical_v_enhanced_fixed.py
from __future__ import annotations

import logging
import json # For schema in prompt
from typing import Any, ClassVar, Dict, Literal, Optional, List, Deque as TypingDeque
from collections import deque

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, ValidationError

from .llm4fts_implementation import LLM4FTSConverter
from tools.technical_tools import close_buf, high_buf, low_buf, vol_buf
from config.config_loader import APP_CONFIG
from utils.agent_memory import get_last_agent_decision

from .enhanced_base_llm_agent import EnhancedBaseLLMAgent

logger = logging.getLogger(__name__)

class EnhancedTechnicalAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    signal: Literal["BUY", "SELL", "HOLD"] = Field(...)
    reasoning: str = Field(..., min_length=10)
    confidence_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = Field(None)
    key_patterns_observed: Optional[List[str]] = Field(default_factory=list)
    temporal_analysis: Optional[str] = Field(None)
    price_target: Optional[float] = Field(None)
    stop_loss_suggestion: Optional[float] = Field(None)
    market_phase: Optional[Literal["ACCUMULATION", "MARKUP", "DISTRIBUTION", "MARKDOWN", "UNCERTAIN"]] = Field(None)

class EnhancedTechnicalAnalyst(EnhancedBaseLLMAgent):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="ignore", validate_assignment=False)

    name: ClassVar[str] = "EnhancedTechnicalAnalyst"
    role: ClassVar[str] = "Advanced Technical Analysis with Temporal Pattern Recognition"
    goal: ClassVar[str] = (
        "Provide structured technical analysis output (BUY, SELL, HOLD signal) with confidence, "
        "key patterns, temporal insights derived from LLM4FTS, detected market phase, and "
        "suggested price/stop-loss levels, all in a strict JSON format."
    )
    backstory: ClassVar[str] = (
        "An AI technical analyst leveraging LLM4FTS for nuanced temporal price action understanding. "
        "It synthesizes numerical indicators, textual price action descriptions, market phase context, "
        "and historical patterns to generate comprehensive and actionable trading insights."
    )
    
    _sequence_length_for_llm4fts: int = PrivateAttr(default=20)
    _pattern_memory: TypingDeque[EnhancedTechnicalAnalysisOutput] = PrivateAttr(default_factory=lambda: deque(maxlen=50))
    _llm4fts_converter: LLM4FTSConverter = PrivateAttr()

    def __init__(self, sequence_length_llm4fts: Optional[int] = None, **kwargs):
        init_data = kwargs.copy()
        init_data.setdefault('agent_type', 'technical')
        super().__init__(**init_data)
        self._llm4fts_converter = LLM4FTSConverter()
        self._sequence_length_for_llm4fts = sequence_length_llm4fts if sequence_length_llm4fts is not None else 20
        # Usar atributo estándar, no PrivateAttr
        self._pattern_memory = deque(maxlen=50)
        
    def run(
        self,
        current_tech_metrics: Dict[str, Any],
        indicator_sequences: Dict[str, List[float]],
        sentiment_label: str = "NEUTRAL",
        symbol_tick_size: float = 0.01 # Added parameter
    ) -> EnhancedTechnicalAnalysisOutput:
        
        # Recuperar memoria reciente
        last_decision, last_ts = get_last_agent_decision('technical', APP_CONFIG.trading.symbol)
        if last_decision and last_ts:
            logger.info(f"[EnhancedTechnicalAnalyst] Última decisión reciente: {last_decision} a las {last_ts}")
        
        if not current_tech_metrics or not isinstance(current_tech_metrics.get('last_price'), (int, float)):
            logger.warning(f"[{self.name}] Insufficient current_tech_metrics or missing/invalid 'last_price'.")
            return self._get_default_output("Insufficient or invalid technical metrics provided (missing last_price).")
        
        temporal_description = self._generate_temporal_analysis(current_tech_metrics, indicator_sequences)
        market_phase_val = self._detect_market_phase(current_tech_metrics)
        
        # Pass symbol_tick_size to _calculate_suggested_price_levels
        price_levels_suggestions = self._calculate_suggested_price_levels(
            current_tech_metrics, market_phase_val, symbol_tick_size
        )
        
        prompt = self._create_enhanced_prompt(
            current_tech_metrics,
            sentiment_label,
            temporal_description,
            market_phase_val,
            price_levels_suggestions
        )

        response = self._query_llm_with_validation(
            prompt,
            EnhancedTechnicalAnalysisOutput,
            schema_type="trading",  # Technical uses trading schema
            use_constitutional=True,
            temperature=0.1
        )
        
        if not response or not isinstance(response, EnhancedTechnicalAnalysisOutput):
            logger.error(f"[{self.name}] Failed to get a valid structured response from LLM. Response type: {type(response)}. Response: {str(response)[:200]}")
            reason = getattr(response, 'reasoning', "LLM query failed or returned an invalid structure for technical analysis.") if isinstance(response, BaseModel) else "LLM query failed."
            return self._get_default_output(reason)
            
        if response.signal == "HOLD":
            response.price_target = None
            response.stop_loss_suggestion = None
        
        if response.temporal_analysis is None:
            response.temporal_analysis = temporal_description
        if response.market_phase is None:
            response.market_phase = market_phase_val

        self._update_pattern_memory(response)
        logger.info(f"[{self.name}] Analysis complete. Signal: {response.signal}, Confidence: {response.confidence_level}, Phase: {response.market_phase}")
        return response

    def _generate_temporal_analysis(self, tech_metrics: Dict[str, Any], indicator_sequences: Dict[str, List[float]]) -> str:
        sequence_length = self._sequence_length_for_llm4fts
        if len(close_buf) < sequence_length:
            return "Insufficient historical data for full temporal analysis based on LLM4FTS."
        
        candles_for_llm4fts = deque(maxlen=sequence_length)
        min_len = min(len(close_buf), len(high_buf), len(low_buf), len(vol_buf))
        if min_len < sequence_length:
             logger.warning(f"[{self.name}] Data buffer length mismatch or insufficient for LLM4FTS. Min available: {min_len}")
             return "Data buffer length mismatch or insufficient for LLM4FTS sequence."

        cl = list(close_buf)[-sequence_length:]
        hi = list(high_buf)[-sequence_length:]
        lo = list(low_buf)[-sequence_length:]
        vo = list(vol_buf)[-sequence_length:]

        for i in range(len(cl)):
            candles_for_llm4fts.append((cl[i], hi[i], lo[i], vo[i])) # type: ignore
            
        if not candles_for_llm4fts:
            return "Could not prepare candle data for LLM4FTS due to buffer issues."
            
        return self._llm4fts_converter.convert_ohlcv_to_text(candles_for_llm4fts, tech_metrics)

    def _detect_market_phase(self, tech_metrics: Dict[str, Any]) -> Literal["ACCUMULATION", "MARKUP", "DISTRIBUTION", "MARKDOWN", "UNCERTAIN"]:
        rsi = tech_metrics.get('rsi', 50.0)
        adx = tech_metrics.get('adx', 20.0)
        price = tech_metrics.get('last_price', 0.0)
        ma50 = tech_metrics.get('ma50', price if price > 0 else 0.0)
        
        if not all(isinstance(v, (int, float)) for v in [rsi, adx, price, ma50]):
            logger.warning(f"[{self.name}] Invalid types for market phase detection: RSI({type(rsi)}), ADX({type(adx)}), Price({type(price)}), MA50({type(ma50)})")
            return "UNCERTAIN"
        if price <= 0:
            return "UNCERTAIN"

        volume_now = tech_metrics.get('curr_vol', 1.0)
        volume_avg_20 = tech_metrics.get('avg_vol_20', 1.0)
        volume_ratio = volume_now / (volume_avg_20 + 1e-9)

        if adx < 20:
            if rsi < 40 and volume_ratio > 1.1:
                return "ACCUMULATION"
            elif rsi > 60 and volume_ratio > 1.1:
                return "DISTRIBUTION"
            return "ACCUMULATION"
        elif adx > 25:
            if ma50 <= 0:
                return "UNCERTAIN"
            if price > ma50 * 1.005:
                return "MARKUP"
            elif price < ma50 * 0.995:
                return "MARKDOWN"
            return "UNCERTAIN"
        return "UNCERTAIN"
    
    def _calculate_suggested_price_levels(
        self,
        tech_metrics: Dict[str, Any],
        market_phase: str,
        symbol_tick_size: float # Added parameter
    ) -> Dict[str, Optional[float]]:
        current_price = tech_metrics.get('last_price', 0.0)
        atr = tech_metrics.get('atr', 0.0)

        if current_price <= 0 or atr <= 0:
            logger.warning(f"[{self.name}] Cannot calculate price levels: current_price={current_price}, atr={atr}. Critical data missing or invalid.")
            return {'target_buy': None, 'stop_loss_buy': None, 'target_sell': None, 'stop_loss_sell': None, 'atr_value_at_calc': atr}

        sl_multiplier = 1.5
        tp_multiplier_base = 2.0
        
        if market_phase == "MARKUP":
            tp_multiplier_base = 2.5
        elif market_phase == "MARKDOWN":
            tp_multiplier_base = 2.5
        elif market_phase == "ACCUMULATION" or market_phase == "DISTRIBUTION":
            tp_multiplier_base = 1.8

        sl_distance = atr * sl_multiplier
        
        # Use the passed symbol_tick_size
        min_sl_distance = max(0.0005 * current_price, 2 * symbol_tick_size)
        sl_distance = max(sl_distance, min_sl_distance)
        tp_distance = max(atr * tp_multiplier_base, sl_distance * 1.2) # Ensure TP is at least a bit beyond SL

        levels = {
            'target_buy': round(current_price + tp_distance, 8) if current_price > 0 else None,
            'stop_loss_buy': round(current_price - sl_distance, 8) if current_price > 0 else None,
            'target_sell': round(current_price - tp_distance, 8) if current_price > 0 else None,
            'stop_loss_sell': round(current_price + sl_distance, 8) if current_price > 0 else None,
            'atr_value_at_calc': atr
        }
        return levels
    
    def _create_enhanced_prompt(
        self,
        tech_metrics: Dict[str, Any],
        sentiment_label: str,
        temporal_description: str,
        market_phase: str,
        price_levels_suggestions: Dict[str, Optional[float]]
    ) -> str:
        
        metrics_str_list = [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in tech_metrics.items()]
        metrics_str = "\n".join(metrics_str_list)
        
        pattern_insights = self._get_pattern_insights()
        
        price_levels_str = "Price level suggestions:\n"
        for key, val in price_levels_suggestions.items():
            if key == 'atr_value_at_calc': # Handle ATR separately
                formatted = f"{val:.4f}" if isinstance(val, float) else "N/A"
                price_levels_str += f"  - (ATR at calculation: {formatted})\n"
            else:
                price_levels_str += f"  - {key}: {f'${val:.4f}' if val is not None else 'N/A'}\n"
        
        schema_json = json.dumps(EnhancedTechnicalAnalysisOutput.model_json_schema(), indent=2)
        example_output = {
            "signal": "BUY",
            "reasoning": "Strong bullish indicators observed. RSI is trending up but not overbought. MACD shows bullish crossover. Price broke above a key resistance level, now acting as support. LLM4FTS temporal analysis confirms short-term upward momentum. Market is in MARKUP phase.",
            "confidence_level": "HIGH",
            "key_patterns_observed": ["Resistance breakout", "Bullish MACD Crossover"],
            "temporal_analysis": "The price shows a consistent upward micro-trend in the last 15 periods, with increasing volume on up-moves, suggesting buying interest.",
            "price_target": 175.50,
            "stop_loss_suggestion": 173.20,
            "market_phase": "MARKUP"
        }
        example_json = json.dumps(example_output, indent=2)

        prompt = f"""You are Plutus, an expert technical analyst for financial markets.
Your task is to analyze the provided market data and generate a trading signal (BUY, SELL, or HOLD) along with supporting details.

**CURRENT MARKET DATA:**
1.  **Key Numerical Technical Indicators:**
{metrics_str}
2.  **General Market Sentiment (External):** {sentiment_label}
3.  **Recent Historical Patterns (Internal Memory):**
{pattern_insights}
4.  **Suggested Price Levels (based on ATR):**
{price_levels_str}

**CONTEXTUAL ANALYSIS (to be incorporated into your reasoning and JSON output):**
-   **Detected Market Phase:** {market_phase}
-   **Temporal Price Action Description (LLM4FTS):**
    {temporal_description}

**INSTRUCTIONS FOR YOUR RESPONSE:**
Based on the INTEGRATION of ALL data provided (numerical indicators, sentiment, historical patterns, suggested levels, market phase, and temporal description), determine the most appropriate trading signal.
Your reasoning must explain HOW you arrived at the signal, which factors were most influential, and if there are any conflicting signals.
The `temporal_analysis` field in your JSON response should summarize or incorporate insights from the 'Temporal Price Action Description' provided above.
The `market_phase` field in your JSON response should reflect the 'Detected Market Phase' provided above.
If the signal is BUY or SELL, the `price_target` and `stop_loss_suggestion` fields in your JSON should be derived from the 'Suggested Price Levels' or your analysis. If HOLD, these should be `null`.
Confidence level (LOW, MEDIUM, HIGH) should reflect the confluence of indicators and clarity of the setup.

CRITICAL RESPONSE FORMATTING:
1. Your response MUST BE ONLY a single, valid JSON object that strictly adheres to the Pydantic schema below.
2. Do NOT include any text, explanations, or markdown (like ```json ... ```) outside the JSON object.
3. The entire response must be the JSON object.

Pydantic Schema to follow:
{schema_json}

Example of the exact JSON output format required:
{example_json}

Now, provide ONLY the JSON object based on your analysis:"""
        return prompt
    
    def _get_pattern_insights(self) -> str:
        # Usar atributo estándar, no PrivateAttr
        pattern_memory_value = getattr(self, '_pattern_memory', None)
        if not isinstance(pattern_memory_value, deque):
            pattern_memory_value = deque(maxlen=50)
            self._pattern_memory = pattern_memory_value
        if not pattern_memory_value or len(pattern_memory_value) < 3:
            return "Insufficient recent pattern data in memory for strong insights."
        
        recent_patterns_to_analyze = list(pattern_memory_value)[-10:]
        if not recent_patterns_to_analyze:
            return "No recent patterns to analyze."

        signals = [p.signal for p in recent_patterns_to_analyze if hasattr(p, 'signal')]
        if not signals:
            return "No valid signals in recent pattern memory."
            
        dominant_signal = max(set(signals), key=signals.count)
        count = signals.count(dominant_signal)
        avg_confidence = sum([1 if getattr(p, 'confidence_level', 'LOW') == 'HIGH' else 0.5 if getattr(p, 'confidence_level', 'LOW') == 'MEDIUM' else 0 for p in recent_patterns_to_analyze]) / len(recent_patterns_to_analyze)
        if avg_confidence > 0.7:
            avg_confidence_text = "with generally HIGH confidence"
        elif avg_confidence > 0.4:
            avg_confidence_text = "with generally MEDIUM confidence"
        else:
            avg_confidence_text = "with generally LOW confidence"

        return (f"Dominant signal in last {len(signals)} recorded patterns: {dominant_signal} "
                f"(observed {count} times) {avg_confidence_text}.")
    
    def _update_pattern_memory(self, response: EnhancedTechnicalAnalysisOutput):
        # Usar atributo estándar, no PrivateAttr
        pattern_memory_value = getattr(self, '_pattern_memory', None)
        if pattern_memory_value is not None:
            pattern_memory_value.append(response)
        else:
            self._pattern_memory = deque([response], maxlen=50)
            logging.warning("[EnhancedTechnicalAnalyst] _pattern_memory was None, reinitialized.")
    
    def _get_default_output(self, reason: str = "Technical analysis could not be performed.") -> EnhancedTechnicalAnalysisOutput:
        logger.warning(f"[{self.name}] Generating default technical output. Reason: {reason}")
        try:
            return EnhancedTechnicalAnalysisOutput(
                signal="HOLD",
                reasoning=reason[:500],
                confidence_level="LOW",
                key_patterns_observed=["N/A due to error"],
                temporal_analysis="Temporal analysis unavailable due to error or insufficient data.",
                market_phase="UNCERTAIN",
                price_target=None,
                stop_loss_suggestion=None
            )
        except ValidationError as e:
            logger.critical(f"[{self.name}] Pydantic validation error even when creating default technical output: {e}")
            return EnhancedTechnicalAnalysisOutput(
                signal="HOLD", 
                reasoning="Critical error creating default output.",
                confidence_level="LOW",
                key_patterns_observed=[],
                temporal_analysis="Error in analysis",
                market_phase="UNCERTAIN",
                price_target=None,
                stop_loss_suggestion=None
            )
