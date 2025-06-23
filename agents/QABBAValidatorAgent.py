# agents/QABBAValidatorAgent.py
import logging
import numpy as np
import json # For schema in prompt
from typing import Dict, List, Optional, Any, Literal, ClassVar
from pydantic import BaseModel, Field, ConfigDict, ValidationError
import joblib # For loading ML model
from pathlib import Path
from utils.agent_memory import get_last_agent_decision
from config.config_loader import APP_CONFIG

from .enhanced_base_llm_agent import EnhancedBaseLLMAgent

logger = logging.getLogger(__name__)

class QABBAAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    qabba_signal: Literal["BUY_QABBA", "SELL_QABBA", "HOLD_QABBA", "NEUTRAL_QABBA"] = Field(...)
    qabba_confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_short: str = Field(..., min_length=10) # Ensure reasoning is provided
    bollinger_analysis: Dict[str, Any] = Field(default_factory=dict) # Populated by agent, not LLM
    volume_profile: Optional[str] = Field(None) # Example, can be expanded
    squeeze_detection: Optional[bool] = Field(None) # Populated by agent
    breakout_probability: Optional[float] = Field(None, ge=0.0, le=1.0) # Populated by agent

class EnhancedQABBAAgent(EnhancedBaseLLMAgent):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="ignore", validate_assignment=False)

    name: ClassVar[str] = "EnhancedQABBAAgent"
    role: ClassVar[str] = "Quantitative Bollinger Bands and Breakout Analysis Specialist"
    goal: ClassVar[str] = (
        "Analyze market conditions using advanced Bollinger Bands techniques, "
        "potential ML insights (if model available), and LLM reasoning to provide a "
        "QABBA signal (BUY_QABBA, SELL_QABBA, HOLD_QABBA, NEUTRAL_QABBA) with confidence and reasoning, "
        "formatted strictly as JSON."
    )
    backstory: ClassVar[str] = (
        "A specialized AI agent, expert in Bollinger Bands (BB) analysis, volatility squeezes, "
        "and breakout probabilities. It combines quantitative calculations with LLM-driven "
        "interpretation to validate trading signals from a unique quantitative perspective. "
        "If an ML model is provided, it further enhances its analysis with predictive insights."
    )
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        init_data = kwargs.copy()
        init_data.setdefault('agent_type', 'qabba')
        super().__init__(**init_data)
        # Usar atributos estándar, no PrivateAttr
        self._squeeze_threshold_pct_config = 2.0
        self._ml_model = None
        self._bb_period = 20
        self._bb_std = 2
        if model_path and Path(model_path).exists():
            try:
                self._ml_model = joblib.load(model_path)
                logger.info(f"[{self.name}] QABBA ML model loaded successfully from {model_path}")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to load QABBA ML model from {model_path}: {e}", exc_info=True)
                self._ml_model = None
        else:
            logger.warning(f"[{self.name}] QABBA ML model path not provided or invalid: {model_path}. ML-based analysis will be skipped.")
            self._ml_model = None
        
    def get_qabba_analysis(
        self,
        tech_metrics: Dict[str, Any], # Allow Any for flexibility from various sources
        price_data_sequence: List[float] # Typically list of close prices
    ) -> QABBAAnalysisOutput:
        
        if not tech_metrics or not isinstance(tech_metrics.get('last_price'), (int, float)):
            logger.warning(f"[{self.name}] Insufficient or invalid tech_metrics for QABBA analysis (missing 'last_price').")
            return self._get_default_output("Insufficient or invalid technical metrics for QABBA analysis.")

        # Calculate detailed QABBA metrics (these are inputs for the LLM, not from LLM)
        qabba_calc_metrics = self._calculate_detailed_qabba_metrics(tech_metrics, price_data_sequence)

        # Attempt ML-based analysis if model is available
        ml_analysis_result = self._ml_based_analysis(tech_metrics, price_data_sequence)
        
        # Perform LLM-based analysis
        # The LLM will provide qabba_signal, qabba_confidence, and reasoning_short
        llm_analysis_result_obj = self._llm_based_analysis(qabba_calc_metrics, tech_metrics, ml_analysis_result)
        
        if not isinstance(llm_analysis_result_obj, QABBAAnalysisOutput):
            logger.error(f"[{self.name}] LLM analysis did not return QABBAAnalysisOutput. Got: {type(llm_analysis_result_obj)}. Using default.")
            reason = getattr(llm_analysis_result_obj, 'reasoning_short', "LLM analysis failed to produce valid QABBA output.") if llm_analysis_result_obj else "LLM analysis returned None."
            final_output = self._get_default_output(reason)
        else:
            final_output = llm_analysis_result_obj

        # Populate fields that are calculated by this agent, not the LLM directly (or use LLM's if it somehow provided them)
        final_output.bollinger_analysis = qabba_calc_metrics # Always use our calculated metrics
        final_output.squeeze_detection = qabba_calc_metrics.get('is_squeeze', False)
        final_output.breakout_probability = self._calculate_breakout_probability(qabba_calc_metrics)
        
        # Combine ML insights with LLM output if ML was used
        # This logic is now moved inside _llm_based_analysis or handled by the prompt
        # For simplicity, the prompt now includes ML results if available.

        logger.info(f"[{self.name}] QABBA Final Analysis: Signal={final_output.qabba_signal}, Confidence={final_output.qabba_confidence:.2f}")
        return final_output

    def run(self, *args, **kwargs):
        last_decision, last_ts = get_last_agent_decision('qabba', APP_CONFIG.trading.symbol)
        if last_decision and last_ts:
            logger.info(f"[EnhancedQABBAAgent] Última decisión reciente: {last_decision} a las {last_ts}")
        super().run(*args, **kwargs)  # Call the parent class's run method

    def _llm_based_analysis(
        self,
        qabba_calc_metrics: Dict[str, Any],
        tech_metrics: Dict[str, Any],
        ml_analysis_result: Optional[Dict[str, Any]] # Pass ML results to include in prompt
    ) -> QABBAAnalysisOutput:
        
        prompt = self._create_qabba_prompt(qabba_calc_metrics, tech_metrics, ml_analysis_result)

        # _query_llm is inherited from EnhancedBaseLLMAgent
        # Temperature and max_tokens will use model_config defaults if not specified here
        response = self._query_llm(prompt, QABBAAnalysisOutput)
        
        if not response or not isinstance(response, QABBAAnalysisOutput):
            logger.error(f"[{self.name}] LLM query for QABBA failed or returned invalid type. Response: {response}")
            reason = getattr(response, 'reasoning_short', "LLM query failed for QABBA analysis.") if response else "LLM query returned None."
            return self._get_default_output(reason)
            
        return response

    def _ml_based_analysis(self, tech_metrics: Dict[str, Any], price_sequence: List[float]) -> Optional[Dict[str, Any]]:
        if not self._ml_model:
            # logger.debug(f"[{self.name}] ML model not loaded, skipping ML-based QABBA analysis.")
            return None
        if len(price_sequence) < self._bb_period: # Ensure enough data for feature calculation
            logger.warning(f"[{self.name}] Not enough price data ({len(price_sequence)} points) for ML features requiring {self._bb_period} points.")
            return None
        
        try:
            features = self._calculate_qabba_features(tech_metrics, price_sequence)
            if features is None or features.ndim == 0 or features.size == 0:
                logger.warning(f"[{self.name}] Could not calculate features for ML model.")
                return None
            
            prediction_proba = self._ml_model.predict_proba(features.reshape(1, -1))[0]
            signal_idx = np.argmax(prediction_proba)
            confidence = float(prediction_proba[signal_idx])
            
            # Assuming model outputs probabilities for [SELL, HOLD, BUY]
            signals_map = ["SELL_QABBA", "HOLD_QABBA", "BUY_QABBA"]
            if signal_idx >= len(signals_map):
                logger.error(f"[{self.name}] ML model signal index {signal_idx} out of bounds for signals_map.")
                return None

            ml_signal = signals_map[signal_idx]
            logger.info(f"[{self.name}] QABBA ML Analysis: Signal={ml_signal}, Confidence={confidence:.2f}")
            return {"ml_signal": ml_signal, "ml_confidence": confidence, "ml_based": True}
        
        except Exception as e:
            logger.error(f"[{self.name}] Error during ML-based QABBA analysis: {e}", exc_info=True)
            return None

    def _calculate_qabba_features(self, tech_metrics: Dict[str, Any], price_sequence: List[float]) -> Optional[np.ndarray]:
        features_list: List[float] = []
        try:
            current_price = float(tech_metrics.get("last_price", 0.0))
            upper_band = float(tech_metrics.get("upper_band", current_price * 1.02)) # Default if not present
            lower_band = float(tech_metrics.get("lower_band", current_price * 0.98)) # Default if not present
            middle_band = float(tech_metrics.get("middle_band", current_price))    # Default if not present

            if not all(isinstance(x, (int, float)) and x > 0 for x in [current_price, upper_band, lower_band, middle_band]):
                logger.warning(f"[{self.name}] Invalid base values for QABBA features: P={current_price}, UB={upper_band}, LB={lower_band}, MB={middle_band}")
                return None

            percent_b = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 1e-9 else 0.5
            features_list.append(percent_b)
            
            bandwidth = (upper_band - lower_band) / middle_band if middle_band > 1e-9 else 0.0
            features_list.append(bandwidth)
            
            features_list.append(float(tech_metrics.get("rsi", 50.0)) / 100.0) # Normalize RSI
            
            avg_vol_20 = float(tech_metrics.get("avg_vol_20", 1.0))
            volume_ratio = float(tech_metrics.get("curr_vol", 1.0)) / (avg_vol_20 + 1e-9) # Avoid zero division
            features_list.append(min(volume_ratio, 3.0)) # Cap volume ratio
            
            momentum = 0.0
            momentum_period = min(10, len(price_sequence))
            if len(price_sequence) >= momentum_period and price_sequence[-momentum_period] > 1e-9:
                momentum = (price_sequence[-1] - price_sequence[-momentum_period]) / price_sequence[-momentum_period]
            features_list.append(momentum)
            
            atr_ratio = float(tech_metrics.get("atr", 0.0)) / current_price if current_price > 1e-9 else 0.0
            features_list.append(atr_ratio)
            
            return np.array(features_list, dtype=np.float64)
        except Exception as e:
            logger.error(f"[{self.name}] Error calculating QABBA ML features: {e}", exc_info=True)
            return None

    def _calculate_detailed_qabba_metrics(self, tech_metrics: Dict[str, Any], price_sequence: List[float]) -> Dict[str, Any]:
        # This function calculates metrics based on inputs, it does not call LLM
        current_price = float(tech_metrics.get("last_price", 0.0))
        # Provide defaults if bands are not in tech_metrics, though they should be
        upper_band = float(tech_metrics.get("upper_band", current_price * 1.02))
        lower_band = float(tech_metrics.get("lower_band", current_price * 0.98))
        middle_band = float(tech_metrics.get("middle_band", current_price))

        if not all(isinstance(x, (int, float)) for x in [current_price, upper_band, lower_band, middle_band]):
             logger.warning(f"[{self.name}] Invalid numeric inputs for detailed QABBA metrics calculation.")
             return {"error": "Invalid input data for QABBA metrics"}


        percent_b = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 1e-9 else 0.5
        bandwidth = (upper_band - lower_band) / middle_band if middle_band > 1e-9 else 0.0
        bandwidth_pct = bandwidth * 100
        squeeze_threshold = getattr(self, '_squeeze_threshold_pct_config', 2.0)
        is_squeeze = bandwidth_pct < squeeze_threshold
        
        band_position = "middle"
        if percent_b > 0.85:
            band_position = "touching_upper"
        elif percent_b > 0.65:
            band_position = "upper_half"
        elif percent_b < 0.15:
            band_position = "touching_lower"
        elif percent_b < 0.35:
            band_position = "lower_half"
        
        return {
            "current_price": round(current_price, 4),
            "upper_band": round(upper_band, 4),
            "middle_band": round(middle_band, 4),
            "lower_band": round(lower_band, 4),
            "percent_b": round(percent_b, 4),
            "bandwidth_abs": round(bandwidth, 6),
            "bandwidth_pct": round(bandwidth_pct, 2),
            "is_squeeze": is_squeeze,
            "band_position_relative_to_price": band_position,
            "bb_settings_used": f"Period={self._bb_period}, StdDev={self._bb_std}",
            "squeeze_threshold_pct_config": self._squeeze_threshold_pct_config
        }

    def _create_qabba_prompt(
        self,
        qabba_metrics: Dict[str, Any],
        tech_metrics: Dict[str, Any],
        ml_analysis: Optional[Dict[str, Any]] # Add ML analysis to prompt
    ) -> str:
        # Format calculated QABBA metrics for the prompt
        qabba_metrics_str = json.dumps({k: (f"{v:.4f}" if isinstance(v, float) else v) for k,v in qabba_metrics.items()}, indent=2)
        
        # Select key general technical metrics for context
        macd_hist_val = None
        if isinstance(tech_metrics.get("macd_line"), (int,float)) and isinstance(tech_metrics.get("signal_line"), (int,float)):
            macd_hist_val = tech_metrics["macd_line"] - tech_metrics["signal_line"]

        vol_ratio_val = None
        if isinstance(tech_metrics.get("curr_vol"), (int,float)) and isinstance(tech_metrics.get("avg_vol_20"), (int,float)) and tech_metrics["avg_vol_20"] > 1e-9:
            vol_ratio_val = tech_metrics["curr_vol"] / tech_metrics["avg_vol_20"]

        key_general_metrics = {
            "rsi": tech_metrics.get("rsi"),
            "adx": tech_metrics.get("adx"),
            "macd_histogram": macd_hist_val,
            "volume_ratio_vs_avg": vol_ratio_val
        }
        general_metrics_str = json.dumps({k: (f"{v:.2f}" if isinstance(v, float) else v) for k,v in key_general_metrics.items() if v is not None}, indent=2)

        ml_analysis_str = "No ML model analysis available."
        if ml_analysis:
            ml_signal = ml_analysis.get("ml_signal", "N/A")
            ml_conf = ml_analysis.get("ml_confidence", 0.0)
            ml_analysis_str = f"ML Model Prediction: Signal={ml_signal}, Confidence={ml_conf:.2f}"

        # Example for the LLM
        example_output = {
            "qabba_signal": "BUY_QABBA",
            "qabba_confidence": 0.75,
            "reasoning_short": "Price bouncing off lower Bollinger Band during a squeeze, with RSI oversold. ML model confirms potential upward movement.",
            # bollinger_analysis, squeeze_detection, breakout_probability are filled by agent, not LLM
        }
        example_json = json.dumps(example_output, indent=2)

        return (
            f"Eres un especialista en Análisis Cuantitativo de Bandas de Bollinger (QABBA) y Detección de Breakouts.\n"
            f"Tu tarea es analizar las siguientes métricas QABBA detalladas, algunas métricas técnicas generales, y una predicción de un modelo ML (si está disponible) para el activo.\n\n"
            f"**Métricas Detalladas de Bandas de Bollinger (QABBA Calculadas):**\n{qabba_metrics_str}\n\n"
            f"**Métricas Técnicas Generales Adicionales:**\n{general_metrics_str}\n\n"
            f"**Análisis de Modelo ML (si disponible):**\n{ml_analysis_str}\n\n"
            f"**Instrucciones para tu Análisis y Respuesta JSON:**\n"
            f"1.  Evalúa la situación actual basándote en las Bandas de Bollinger (posición del precio %B, ancho de banda, compresión 'is_squeeze').\n"
            f"2.  Considera cómo las métricas generales (RSI, ADX, MACD, volumen) y el análisis del modelo ML (si existe y es confiable) confirman o contradicen la lectura de las Bandas de Bollinger.\n"
            f"3.  Determina una señal QABBA final: 'BUY_QABBA', 'SELL_QABBA', 'HOLD_QABBA', o 'NEUTRAL_QABBA'.\n"
            f"4.  Estima una confianza para esta señal QABBA (`qabba_confidence`) entre 0.0 y 1.0.\n"
            f"5.  Proporciona un razonamiento corto y conciso (`reasoning_short`, mín. 10 caracteres) que justifique tu señal y confianza, integrando todas las fuentes de información.\n\n"
            f"CRITICAL RESPONSE FORMATTING:\n"
            f"1. Tu respuesta DEBE SER ÚNICAMENTE un objeto JSON válido que se adhiera estrictamente al esquema Pydantic proporcionado.\n"
            f"2. No incluyas ningún texto, explicación, o markdown (como ```json ... ```) fuera del objeto JSON.\n"
            f"3. El objeto JSON solo debe contener los campos: `qabba_signal`, `qabba_confidence`, `reasoning_short`.\n"
            f"   Los campos `bollinger_analysis`, `squeeze_detection`, `breakout_probability`, `volume_profile` NO deben ser generados por ti; serán completados por el sistema.\n\n"
            f"Pydantic Schema para los campos que DEBES generar:\n"
            f"{{\n  \"qabba_signal\": \"BUY_QABBA\" | \"SELL_QABBA\" | \"HOLD_QABBA\" | \"NEUTRAL_QABBA\",\n  \"qabba_confidence\": float (0.0-1.0),\n  \"reasoning_short\": \"string (min 10 chars)\"\n}}\n\n"
            f"Ejemplo del formato JSON exacto requerido (solo estos campos):\n{example_json}\n\n"
            f"Ahora, proporciona ÚNICAMENTE el objeto JSON con los campos `qabba_signal`, `qabba_confidence`, y `reasoning_short`:"
        )

    def _calculate_breakout_probability(self, qabba_metrics: Dict[str, Any]) -> float:
        # Simple heuristic for breakout probability, can be made more complex
        prob = 0.5 # Base probability
        
        if qabba_metrics.get('is_squeeze', False):
            prob += 0.25 # Squeezes often precede breakouts
        
        bandwidth_pct = float(qabba_metrics.get('bandwidth_pct', 100.0))
        if bandwidth_pct < self._squeeze_threshold_pct_config * 0.75: # Tighter squeeze
            prob += 0.10
        if bandwidth_pct < self._squeeze_threshold_pct_config * 0.5: # Very tight squeeze
            prob += 0.10
            
        percent_b = float(qabba_metrics.get('percent_b', 0.5))
        # If in a squeeze and price is pushing against a band, higher chance of breakout in that direction
        if qabba_metrics.get('is_squeeze', False):
            if percent_b > 0.90 or percent_b < 0.10: # Close to upper or lower band
                prob += 0.05
                
        return round(min(max(0.1, prob), 0.9), 2) # Clamp between 0.1 and 0.9

    def _get_default_output(self, reason: str = "QABBA analysis failed or critical error.") -> QABBAAnalysisOutput:
        logger.warning(f"[{self.name}] Generating default QABBA output. Reason: {reason}")
        try:
            return QABBAAnalysisOutput(
                qabba_signal="NEUTRAL_QABBA",
                qabba_confidence=0.0,
                reasoning_short=reason[:250], # Truncate if too long
                bollinger_analysis={}, # Default empty
                volume_profile=None,
                squeeze_detection=None,
                breakout_probability=0.5 # Neutral default
            )
        except ValidationError as e:
            logger.critical(f"[{self.name}] Pydantic validation error even when creating default QABBA output: {e}")
            # This indicates a fundamental issue with the Pydantic model or default values
            return QABBAAnalysisOutput(
                qabba_signal="NEUTRAL_QABBA",
                qabba_confidence=0.0,
                reasoning_short="Critical error creating default QABBA output.",
                bollinger_analysis={},
                volume_profile=None,
                squeeze_detection=False,
                breakout_probability=0.5
            )

