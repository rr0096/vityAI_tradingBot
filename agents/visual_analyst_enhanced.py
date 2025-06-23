# agents/visual_analyst_enhanced.py
from __future__ import annotations
from typing import Any, ClassVar, Dict, List, Literal, Optional, Deque as TypingDeque  # Tuple unused, removed
from datetime import datetime, timezone  # timedelta unused, removed
from collections import deque, Counter
import statistics
import logging
import json # For schema in prompt
import os

from pydantic import BaseModel, Field, ConfigDict, ValidationError
from .enhanced_base_llm_agent import EnhancedBaseLLMAgent
from config.config_loader import APP_CONFIG
from utils.agent_memory import get_last_agent_decision

logger = logging.getLogger(__name__)

# CONFIGURACIÓN: Usar SIEMPRE captura real
try:
    from tools.chart_generator_real import generate_chart_for_visual_agent_real as generate_chart_for_visual_agent
    logger.info("🎯 CONFIGURADO PARA USAR CAPTURA REAL DE TRADINGVIEW")
    USE_REAL_CAPTURE = True
except ImportError as e:
    logger.error(f"❌ No se pudo importar captura real: {e}")
    from tools.chart_generator import generate_chart_for_visual_agent
    logger.info("📊 USANDO CHART GENERATOR TRADICIONAL COMO FALLBACK")
    USE_REAL_CAPTURE = False

class EnhancedVisualChartAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    
    overall_visual_assessment: Literal["BULLISH", "BEARISH", "NEUTRAL", "UNCLEAR"] = Field(...)
    key_candlestick_patterns: Optional[List[str]] = Field(default_factory=list)
    chart_patterns: Optional[List[str]] = Field(default_factory=list)
    trend_analysis: Optional[Dict[str, str]] = Field(default_factory=dict) # E.g., {"primary": "uptrend", "strength": "moderate"}
    indicator_interpretation: Optional[Dict[str, str]] = Field(default_factory=dict) # E.g., {"RSI": "approaching_oversold"}
    volume_analysis: Optional[Dict[str, Optional[str]]] = Field(default_factory=dict) # E.g., {"trend": "increasing_on_upmoves", "confirmation": "confirms_bullish_breakout"}
    support_resistance_levels: Optional[Dict[str, List[float]]] = Field(default_factory=dict) # E.g., {"support": [100.50], "resistance": [110.0]}
    pattern_clarity_score: float = Field(0.5, ge=0.0, le=1.0)  # Default to 0.5 if not provided
    suggested_action_based_on_visuals: Optional[Literal["CONSIDER_BUY", "CONSIDER_SELL", "WAIT_CONFIRMATION", "AVOID_TRADE"]] = Field(default="AVOID_TRADE")
    reasoning: str = Field(..., min_length=10)
    chart_timeframe_analyzed: Optional[str] = Field(None) # To be filled by the agent
    main_elements_focused_on: Optional[List[str]] = Field(default_factory=list)


class EnhancedVisualAnalystAgent(EnhancedBaseLLMAgent):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="ignore", validate_assignment=False)

    name: ClassVar[str] = "EnhancedVisualAnalystAgent"
    role: ClassVar[str] = "Expert Multimodal Chart Pattern Interpreter and Visual Trend Analyst"
    goal: ClassVar[str] = (
        "Provide a comprehensive, structured visual analysis of trading charts by interpreting "
        "candlestick patterns, chart formations, trends, indicators, and volume. "
        "Output findings in a consistent JSON format."
    )
    backstory: ClassVar[str] = (
        "An advanced AI agent with exceptional visual acuity, trained on thousands of financial charts. "
        "It meticulously examines price action and related visual cues to deliver objective chart interpretations, "
        "avoiding emotional bias and focusing solely on the visual evidence presented in the chart image."
    )
    
    def __init__(self, save_charts_to_disk: Optional[bool] = None, **kwargs):
        init_data = kwargs.copy()
        init_data.setdefault('agent_type', 'visual')
        super().__init__(**init_data)
        # Usar atributo estándar, no PrivateAttr
        self._analysis_history = deque(maxlen=20)
        if save_charts_to_disk is not None:
            self._save_charts_to_disk = save_charts_to_disk
        else:
            env_save_flag = os.getenv("SAVE_GENERATED_CHARTS", "true").lower()
            self._save_charts_to_disk = True if env_save_flag == "true" else False

        if not self._supports_vision:
             logger.critical(
                f"[{self.agent_type}] CRITICAL WARNING: Configured model '{self._llm_model_name}' "
                f"is NOT marked as supporting vision (self._supports_vision is False). Visual analysis WILL LIKELY FAIL. "
                f"Please use a multimodal model and ensure 'supports_vision=True' in its ModelConfig in modern_models.py."
            )
        elif self._llm_model_name and not any(tag in self._llm_model_name.lower() for tag in ['vision', 'llava', 'moondream', 'bakllava', 'gemma3', 'qwen2.5vl']): # Added gemma3 and qwen2.5vl common tags
            logger.warning(
                f"[{self.agent_type}] VisualAnalystAgent is using model '{self._llm_model_name}'. "
                f"While _supports_vision is True, the model name doesn't common vision tags. "
                f"Ensure this model truly supports multimodal inputs for visual analysis."
            )

    def run(
        self,
        symbol: str,
        timeframe_str: str, # This is crucial for the prompt
        close_buf_deque: TypingDeque[float],
        high_buf_deque: TypingDeque[float],
        low_buf_deque: TypingDeque[float],
        vol_buf_deque: TypingDeque[float],
        tech_metrics: Optional[Dict[str, Any]] = None
    ) -> EnhancedVisualChartAnalysisOutput:
        
        last_decision, last_ts = get_last_agent_decision('visual', APP_CONFIG.trading.symbol)
        if last_decision and last_ts:
            logger.info(f"[EnhancedVisualAnalystAgent] Última decisión reciente: {last_decision} a las {last_ts}")
        
        chart_b64_string: Optional[str] = None
        chart_filepath: str = ""
        default_reasoning_llm_error = "LLM query failed or client not available for visual analysis."

        if not self._model_config_internal or not self._supports_vision:
            logger.error(f"[{self.name}] Cannot perform visual analysis. Model: '{self._llm_model_name}', Supports Vision: {self._supports_vision}.")
            return self._create_default_visual_output(
                f"Model '{self._llm_model_name}' does not support vision or agent not properly configured.",
                timeframe_str
            )

        try:
            logger.info(f"[{self.name}] Generating chart for {symbol} ({timeframe_str}). Save flag: {self._save_charts_to_disk}")
            
            # VERIFICAR QUÉ FUNCIÓN SE ESTÁ USANDO
            func_name = getattr(generate_chart_for_visual_agent, '__name__', 'unknown')
            func_module = getattr(generate_chart_for_visual_agent, '__module__', 'unknown')
            logger.info(f"[{self.name}] 🔍 USANDO FUNCIÓN: {func_name} de módulo: {func_module}")
            
            logger.info(f"[{self.name}] 📞 LLAMANDO a generate_chart_for_visual_agent...")
            chart_b64_string, chart_filepath = generate_chart_for_visual_agent(
                symbol=symbol, timeframe=timeframe_str,
                close_buf=close_buf_deque, high_buf=high_buf_deque,
                low_buf=low_buf_deque, vol_buf=vol_buf_deque,
                tech_metrics=tech_metrics or {},
                lookback_periods=100, # Or make this configurable
                save_chart=self._save_charts_to_disk
            )
            
            logger.info(f"[{self.name}] 📥 RESPUESTA RECIBIDA:")
            logger.info(f"[{self.name}] - chart_b64_string: {'✅ Recibido' if chart_b64_string else '❌ Vacío'} ({len(chart_b64_string) if chart_b64_string else 0} chars)")
            logger.info(f"[{self.name}] - chart_filepath: {chart_filepath if chart_filepath else '❌ Vacío'}")
            
            if chart_filepath:
                logger.info(f"[{self.name}] 📸 Gráfico capturado: {chart_filepath}")
            
            if chart_b64_string:
                logger.info(f"[{self.name}] 🔍 Primeros 50 chars del base64: {chart_b64_string[:50]}...")
                logger.info(f"[{self.name}] 🔍 Últimos 50 chars del base64: ...{chart_b64_string[-50:]}")
            
        except Exception as e:
            logger.error(f"[{self.name}] Error capturando gráfico: {e}", exc_info=True)
            return self._create_default_visual_output(f"Chart generation failed: {str(e)[:100]}", timeframe_str)

        if not chart_b64_string or chart_b64_string.startswith("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="): # Placeholder for error image
            logger.warning(f"[{self.name}] Chart image string is empty or error placeholder for {symbol} after generation attempt. Length: {len(chart_b64_string) if chart_b64_string else 0}")
            if chart_b64_string and len(chart_b64_string) > 200: # Log if it's long but still might be an error image
                logger.debug(f"Chart b64 start: {chart_b64_string[:100]}, end: {chart_b64_string[-100:]}")
            return self._create_default_visual_output("Chart generation resulted in an empty or error image.", timeframe_str)

        schema_json = json.dumps(EnhancedVisualChartAnalysisOutput.model_json_schema(), indent=2)
        # Generate a more complete example for the LLM
        example_output_dict = {
            "overall_visual_assessment": "BULLISH",
            "key_candlestick_patterns": ["Bullish Engulfing", "Morning Star (partial)"],
            "chart_patterns": ["Ascending Triangle (forming)"],
            "trend_analysis": {"primary": "uptrend", "strength": "moderate", "channel_observed": "upward channel visible"},
            "indicator_interpretation": {"MA_50": "acting_as_dynamic_support", "RSI_chart": "trending_up_from_50"},
            "volume_analysis": {"trend": "increasing_on_upmoves", "confirmation": "confirms_bullish_momentum"},
            "support_resistance_levels": {"support": [150.0, 148.5], "resistance": [155.0, 157.5]},
            "pattern_clarity_score": 0.75,
            "suggested_action_based_on_visuals": "CONSIDER_BUY",
            "reasoning": "The chart shows a clear short-term uptrend with price consistently making higher lows. A bullish engulfing pattern was observed near the MA50, which is acting as dynamic support. Volume increased on the last up-move. An ascending triangle appears to be forming, suggesting potential for further upside if the 155.0 resistance is broken.",
            "chart_timeframe_analyzed": timeframe_str, # Use the actual timeframe
            "main_elements_focused_on": ["Price Action", "MA50 Support", "Volume Confirmation", "Ascending Triangle"]
        }
        example_json = json.dumps(example_output_dict, indent=2)


        prompt = f"""You are an expert multimodal technical chart analyst. Analyze the provided financial chart image meticulously.
The chart timeframe is: {timeframe_str}. Ensure your analysis is relevant to this timeframe.
Based SOLELY on the visual information in the chart, provide your analysis.

CRITICAL INSTRUCTIONS FOR RESPONSE FORMATTING:
1. Your response MUST BE ONLY a single, valid JSON object that strictly adheres to the provided Pydantic schema.
2. Do NOT include any text, explanations, apologies, or markdown formatting (like ```json ... ```) outside of the JSON object itself.
3. The entire response must be the JSON object.
4. For list fields like `key_candlestick_patterns` or `chart_patterns`, if none are clearly observed, return an empty list `[]`.
5. For dictionary fields like `trend_analysis` or `indicator_interpretation`, if no clear interpretation can be made, return an empty dictionary `{{}}`.
6. The `chart_timeframe_analyzed` field in your JSON response MUST be exactly: "{timeframe_str}".

Pydantic Schema to follow:
{schema_json}

Example of the exact JSON output format required:
{example_json}

Detailed analysis points to include in the JSON:
- `overall_visual_assessment`: Your primary visual read (must be one of: "BULLISH", "BEARISH", "NEUTRAL", "UNCLEAR").
- `key_candlestick_patterns`: List any significant candlestick patterns observed (e.g., ["Doji", "Bullish Engulfing"]).
- `chart_patterns`: List any broader chart formations (e.g., ["Head and Shoulders", "Ascending Triangle"]).
- `trend_analysis`: Describe the primary trend, its strength, and any observed channels (e.g., {{"primary": "uptrend", "strength": "moderate", "channel_observed": "upward channel visible"}}).
- `indicator_interpretation`: If indicators like MAs or RSI are visibly plotted and clear, interpret them (e.g., {{"RSI_on_chart": "approaching_oversold", "MA_Cross_visible": "golden_cross_occurred"}}).
- `volume_analysis`: Analyze volume bars if present. Note trends or confirmations (e.g., {{"trend": "increasing_on_upmoves", "confirmation": "confirms_bullish_breakout"}}).
- `support_resistance_levels`: Identify key visual S/R levels (e.g., {{"support": [100.50, 98.0], "resistance": [110.0]}}).
- `pattern_clarity_score`: Your confidence in the clarity of observed patterns (float from 0.0 to 1.0).
- `suggested_action_based_on_visuals`: A cautious trading suggestion based purely on visuals (must be one of: "CONSIDER_BUY", "CONSIDER_SELL", "WAIT_CONFIRMATION", "AVOID_TRADE").
- `reasoning`: Your detailed thought process explaining the overall assessment and other findings, linking them to visual evidence. Must be a non-empty string.
- `chart_timeframe_analyzed`: Must be "{timeframe_str}".
- `main_elements_focused_on`: List the top 3-5 visual elements you primarily focused on (e.g., ["Price Action", "Volume Profile", "MA Crossover"]).

Now, provide ONLY the JSON object based on your analysis of the chart image:"""
        
        logger.debug(f"[{self.name}] Sending chart to LLM. Base64 length: {len(chart_b64_string) if chart_b64_string else 0}. Prompt includes timeframe: {timeframe_str}")
        if chart_b64_string and len(chart_b64_string) < 500: # Log short base64 strings entirely if they might be error placeholders
            logger.debug(f"[{self.name}] Chart base64 content (short): {chart_b64_string}")
        elif chart_b64_string:
            logger.debug(f"[{self.name}] Chart base64 start: {chart_b64_string[:100]}, end: {chart_b64_string[-100:]}")


        response = self._safe_visual_query(
            prompt=prompt,
            chart_b64_string=chart_b64_string,
            timeframe_str=timeframe_str
        )
        
        # Log the response details for debugging
        if response:
            logger.info(f"[{self.name}] LLM Response received - Assessment: {response.overall_visual_assessment}, Clarity Score: {response.pattern_clarity_score}")
            if response.pattern_clarity_score is None or response.pattern_clarity_score == 0.0:
                logger.warning(f"[{self.name}] Pattern clarity score is {response.pattern_clarity_score}! Response type: {type(response)}")
        else:
            logger.warning(f"[{self.name}] No response received from LLM")
        
        # If vision model is not working, try fallback analysis
        if not response or not isinstance(response, EnhancedVisualChartAnalysisOutput):
            logger.warning(f"[{self.name}] Vision model failed, attempting technical analysis fallback")
            response = self._fallback_technical_analysis(tech_metrics or {}, timeframe_str)
        
        if not response or not isinstance(response, EnhancedVisualChartAnalysisOutput):
            logger.error(f"[{self.name}] Failed to get a valid structured response from visual LLM. Response type: {type(response)}. Response: {str(response)[:200]}")
            reason_from_failed_response = getattr(response, 'reasoning', None) if isinstance(response, BaseModel) else None
            final_reason = reason_from_failed_response or default_reasoning_llm_error
            response = self._create_default_visual_output(final_reason, timeframe_str)
        
        # Ensure timeframe is correctly set, even if LLM missed it (though prompt now enforces it)
        if response.chart_timeframe_analyzed != timeframe_str:
            logger.warning(f"[{self.name}] LLM returned timeframe '{response.chart_timeframe_analyzed}', but expected '{timeframe_str}'. Overriding.")
            response.chart_timeframe_analyzed = timeframe_str
        
        # Guardar en memoria estándar, no PrivateAttr
        history_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "assessment": response.overall_visual_assessment,
            "clarity_score": response.pattern_clarity_score,
            "model_used": self._llm_model_name,
            "chart_image_path": chart_filepath if chart_filepath else "Not Saved or Path Not Returned",
            "timeframe_analyzed_by_llm": response.chart_timeframe_analyzed
        }
        if not hasattr(self, '_analysis_history') or not isinstance(self._analysis_history, deque):
            self._analysis_history = deque(maxlen=20)
        self._analysis_history.append(history_entry)
        
        clarity_score_display = f"{response.pattern_clarity_score:.2f}"
        logger.info(f"[{self.name}] LLM Visual Assessment for {symbol} ({response.chart_timeframe_analyzed}) using {self._llm_model_name}: {response.overall_visual_assessment} (Clarity: {clarity_score_display})")
        return response

    def _safe_visual_query(self, prompt: str, chart_b64_string: str, timeframe_str: str) -> Optional[EnhancedVisualChartAnalysisOutput]:
        """
        Safely query the visual LLM with Ollama error handling
        """
        try:
            response = self._query_llm_with_validation(
                prompt=prompt,
                response_model=EnhancedVisualChartAnalysisOutput,
                schema_type="visual",
                use_constitutional=True,
                images=[chart_b64_string] if chart_b64_string else []
            )
            return response
        except Exception as e:
            error_str = str(e).lower()
            
            # Specific handling for Ollama 500 errors
            if "500" in error_str and ("internal server error" in error_str or "eof" in error_str):
                logger.error(f"[{self.name}] Ollama server error (500) detected. This usually indicates:")
                logger.error(f"[{self.name}]   - Insufficient memory for vision model")
                logger.error(f"[{self.name}]   - Model '{self._llm_model_name}' may be corrupted")
                logger.error(f"[{self.name}]   - Image too large for processing")
                return None
            
            # Handle other API errors
            elif "api_error" in error_str or "timeout" in error_str:
                logger.error(f"[{self.name}] LLM API error: {str(e)[:200]}")
                return None
            
            else:
                # Re-raise unexpected errors
                raise e

    def _create_default_visual_output(self, reason: str, timeframe: str) -> EnhancedVisualChartAnalysisOutput:
        logger.warning(f"[{self.name}] Creating default visual output. Reason: {reason}")
        try:
            return EnhancedVisualChartAnalysisOutput(
                overall_visual_assessment="UNCLEAR",
                reasoning=reason[:500] if reason else "Visual analysis failed due to technical issues",
                key_candlestick_patterns=["No patterns detected"],
                chart_patterns=["Analysis unavailable"],
                trend_analysis={"trend": "undefined", "confidence": "low"},
                indicator_interpretation={"status": "no_data"},
                volume_analysis={"volume_trend": "undefined"},
                support_resistance_levels={"support": [], "resistance": []},
                pattern_clarity_score=0.0,
                chart_timeframe_analyzed=timeframe,
                main_elements_focused_on=["Technical error prevented analysis"],
                suggested_action_based_on_visuals="AVOID_TRADE"
            )
        except ValidationError as e:
            logger.critical(f"[{self.name}] Pydantic validation error even when creating default visual output: {e}")
            # Emergency fallback with all required fields
            return EnhancedVisualChartAnalysisOutput(
                overall_visual_assessment="UNCLEAR",
                reasoning="Critical error creating default output after primary failure",
                key_candlestick_patterns=["Error"],
                chart_patterns=["Error"],
                trend_analysis={"error": "true"},
                indicator_interpretation={"error": "true"},
                volume_analysis={"error": "true"},
                support_resistance_levels={"support": [], "resistance": []},
                pattern_clarity_score=0.0,
                chart_timeframe_analyzed=timeframe,
                main_elements_focused_on=["Critical Error"],
                suggested_action_based_on_visuals="AVOID_TRADE"
            )

    def _fallback_technical_analysis(self, tech_metrics: Dict[str, Any], timeframe: str) -> EnhancedVisualChartAnalysisOutput:
        """
        Fallback method that creates a visual analysis based on technical metrics
        when the vision model is not available.
        """
        logger.info(f"[{self.name}] Using technical analysis fallback for visual assessment")
        
        try:
            # Extract technical indicators
            rsi = tech_metrics.get('rsi', 50.0)
            sma_20 = tech_metrics.get('sma_20', 0.0)
            sma_50 = tech_metrics.get('sma_50', 0.0)
            bb_upper = tech_metrics.get('bb_upper', 0.0)
            bb_lower = tech_metrics.get('bb_lower', 0.0)
            current_price = tech_metrics.get('close', tech_metrics.get('current_price', 0.0))
            
            # Determine assessment based on technical indicators
            assessment = "NEUTRAL"
            clarity_score = 0.6  # Moderate clarity for technical analysis
            reasoning_parts = []
            
            # RSI analysis
            if rsi > 70:
                assessment = "BEARISH"
                reasoning_parts.append(f"RSI at {rsi:.1f} indicates overbought conditions")
            elif rsi < 30:
                assessment = "BULLISH"
                reasoning_parts.append(f"RSI at {rsi:.1f} indicates oversold conditions")
            else:
                reasoning_parts.append(f"RSI at {rsi:.1f} is in neutral range")
            
            # Moving average analysis
            if sma_20 > 0 and sma_50 > 0 and current_price > 0:
                if sma_20 > sma_50 and current_price > sma_20:
                    if assessment != "BEARISH":
                        assessment = "BULLISH"
                    reasoning_parts.append("Price above SMA20 which is above SMA50 suggests uptrend")
                    clarity_score += 0.2
                elif sma_20 < sma_50 and current_price < sma_20:
                    if assessment != "BULLISH":
                        assessment = "BEARISH"
                    reasoning_parts.append("Price below SMA20 which is below SMA50 suggests downtrend")
                    clarity_score += 0.2
                else:
                    reasoning_parts.append("Mixed signals from moving averages")
            
            # Bollinger Bands analysis
            if bb_upper > 0 and bb_lower > 0 and current_price > 0:
                if current_price > bb_upper:
                    reasoning_parts.append("Price above upper Bollinger Band suggests potential reversal")
                elif current_price < bb_lower:
                    reasoning_parts.append("Price below lower Bollinger Band suggests potential reversal")
                else:
                    reasoning_parts.append("Price within Bollinger Bands range")
            
            # Determine suggested action
            suggested_action = "WAIT_CONFIRMATION"
            if assessment == "BULLISH" and clarity_score > 0.7:
                suggested_action = "CONSIDER_BUY"
            elif assessment == "BEARISH" and clarity_score > 0.7:
                suggested_action = "CONSIDER_SELL"
            elif clarity_score < 0.5:
                suggested_action = "AVOID_TRADE"
            
            # Cap clarity score
            clarity_score = min(0.8, clarity_score)  # Max 0.8 for technical fallback
            
            reasoning = f"Technical analysis fallback (vision model unavailable): {'. '.join(reasoning_parts)}. This analysis is based on numerical indicators only, without visual chart pattern confirmation."
            
            return EnhancedVisualChartAnalysisOutput(
                overall_visual_assessment=assessment,
                key_candlestick_patterns=[],  # Cannot detect without visual analysis
                chart_patterns=[],  # Cannot detect without visual analysis
                trend_analysis={"primary": "unknown", "strength": "unknown_technical_only"},
                indicator_interpretation={
                    "RSI": f"at_{rsi:.1f}",
                    "SMA_relationship": f"sma20_{sma_20:.2f}_vs_sma50_{sma_50:.2f}" if sma_20 > 0 and sma_50 > 0 else "unavailable"
                },
                volume_analysis={"trend": "unknown", "confirmation": "not_available_in_fallback"},
                support_resistance_levels={
                    "support": [bb_lower] if bb_lower > 0 else [],
                    "resistance": [bb_upper] if bb_upper > 0 else []
                },
                pattern_clarity_score=clarity_score,
                suggested_action_based_on_visuals=suggested_action,
                reasoning=reasoning,
                chart_timeframe_analyzed=timeframe,
                main_elements_focused_on=["RSI", "Moving Averages", "Bollinger Bands"]
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] Error in technical analysis fallback: {e}", exc_info=True)
            return self._create_default_visual_output(
                f"Technical analysis fallback failed: {str(e)[:100]}", 
                timeframe
            )

    def get_analysis_summary(self) -> Dict[str, Any]:
        if not self._analysis_history:
            return {"status": "No visual analysis history available."}
        
        try:
            assessments = [h.get('assessment', 'UNCLEAR') for h in self._analysis_history]
            clarity_scores_raw = [h.get('clarity_score') for h in self._analysis_history]
            valid_clarity_scores = [cs for cs in clarity_scores_raw if isinstance(cs, (int, float)) and cs is not None]
            
            avg_clarity = round(statistics.mean(valid_clarity_scores), 2) if valid_clarity_scores else 0.0
            last_analysis_entry = self._analysis_history[-1]
            last_timestamp_iso = last_analysis_entry.get('timestamp', "N/A")
            
            return {
                'total_analyses_in_history': len(self._analysis_history),
                'assessment_distribution': dict(Counter(assessments)),
                'average_clarity_score': avg_clarity,
                'last_analysis_timestamp': last_timestamp_iso,
                'last_analysis_model': last_analysis_entry.get('model_used', 'N/A'),
                'last_timeframe_analyzed': last_analysis_entry.get('timeframe_analyzed_by_llm', 'N/A')
            }
        except Exception as e:
            logger.error(f"[{self.name}] Error generating visual analysis summary: {e}", exc_info=True)
            return {"status": "Error in summary generation."}
