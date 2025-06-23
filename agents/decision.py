# agents/decision.py
"""
Sistema de Ponderación Dinámica y Agente de Decisión.
Asigna pesos adaptativos a cada agente basándose en su rendimiento histórico y condiciones del mercado,
y luego toma una decisión final de trading.
"""
from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Literal, ClassVar
from datetime import datetime
from collections import deque, Counter

import numpy as np
from crewai import Agent
# Asegúrate de importar PrivateAttr
from pydantic import BaseModel, PrivateAttr

from models.outputs import (
    SentimentOutput, TechnicalAnalysisOutput,
    FinalDecisionOutput, RiskAssessment
)
# Asumiendo que los outputs de visual y qabba se manejan como 'Any' o se importan específicamente
# from agents.visual_analyst_enhanced import EnhancedVisualChartAnalysisOutput
# from agents.QABBAValidatorAgent import QABBAAnalysisOutput


logger = logging.getLogger(__name__)

class DynamicWeightingSystem:
    """Sistema que ajusta dinámicamente los pesos de cada agente según su performance."""

    def __init__(self, performance_log_path: str = "logs/agent_performance.jsonl"):
        self.performance_log_path = Path(performance_log_path)
        self.performance_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.weights = {
            'sentiment': 0.20,
            'technical': 0.30,
            'visual': 0.30,
            'qabba': 0.20
        }
        self.prediction_history = {
            'sentiment': deque(maxlen=100),
            'technical': deque(maxlen=100),
            'visual': deque(maxlen=100),
            'qabba': deque(maxlen=100)
        }
        self.performance_metrics = {
            'sentiment': {'correct': 0, 'total': 0, 'profit_contribution': 0.0},
            'technical': {'correct': 0, 'total': 0, 'profit_contribution': 0.0},
            'visual': {'correct': 0, 'total': 0, 'profit_contribution': 0.0},
            'qabba': {'correct': 0, 'total': 0, 'profit_contribution': 0.0}
        }
        self.market_condition_adjustments = {
            'high_volatility': {'sentiment': 0.8, 'technical': 1.2, 'visual': 1.1, 'qabba': 1.0},
            'trending': {'sentiment': 0.9, 'technical': 1.3, 'visual': 1.2, 'qabba': 1.1},
            'ranging': {'sentiment': 1.1, 'technical': 0.9, 'visual': 1.0, 'qabba': 1.2}
        }
        self._load_performance_history()

    def calculate_weighted_decision(
        self,
        sentiment_analysis: SentimentOutput,
        technical_analysis: TechnicalAnalysisOutput,
        visual_analysis: Any, # Reemplazar con tipo específico si es necesario
        qabba_analysis: Any,  # Reemplazar con tipo específico si es necesario
        market_conditions: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, Any]]:
        scores = self._convert_to_scores(
            sentiment_analysis, technical_analysis,
            visual_analysis, qabba_analysis
        )
        # Nuevo: calcular confianza individual de cada agente
        confidences = {}
        if hasattr(sentiment_analysis, 'confidence_score'):
            confidences['sentiment'] = getattr(sentiment_analysis, 'confidence_score', 0.5)
        if hasattr(technical_analysis, 'confidence_level'):
            conf_map = {'LOW': 0.5, 'MEDIUM': 0.75, 'HIGH': 1.0}
            confidences['technical'] = conf_map.get(getattr(technical_analysis, 'confidence_level', 'MEDIUM'), 0.75)
        if hasattr(qabba_analysis, 'qabba_confidence'):
            confidences['qabba'] = getattr(qabba_analysis, 'qabba_confidence', 0.5)
        # Visual: suponer 0.75 por defecto
        confidences['visual'] = 0.75

        market_condition_label = self._detect_market_condition(market_conditions)
        adjusted_weights = self._adjust_weights_for_market(market_condition_label)

        weighted_score = 0.0
        weight_sum = 0.0
        contributions = {}

        for agent_name, score_value in scores.items():
            weight = adjusted_weights.get(agent_name, 0.25)
            contribution = score_value * weight
            weighted_score += contribution
            weight_sum += weight
            contributions[agent_name] = {
                'score': score_value, 'weight': weight, 'contribution': contribution
            }

        if weight_sum > 0:
            weighted_score /= weight_sum

        # Usar la nueva lógica de decisión flexible
        decision = self._score_to_decision(weighted_score, scores, confidences)
        confidence = self._calculate_confidence(scores, weighted_score)
        self._log_prediction(
            decision, confidence, scores, contributions,
            market_condition_label, sentiment_analysis, technical_analysis,
            visual_analysis, qabba_analysis
        )
        details = {
            'weighted_score': weighted_score, 'contributions': contributions,
            'market_condition': market_condition_label, 'adjusted_weights': adjusted_weights,
            'raw_scores': scores, 'convergence': self._calculate_convergence(scores)
        }
        return decision, confidence, details

    def _convert_to_scores(self, sentiment, technical, visual, qabba) -> Dict[str, float]:
        scores = {}
        sentiment_map = {'POSITIVE': 1.0, 'NEUTRAL': 0.0, 'NEGATIVE': -1.0}
        scores['sentiment'] = sentiment_map.get(getattr(sentiment, 'overall_sentiment', 'NEUTRAL'), 0.0)

        technical_map = {'BUY': 1.0, 'HOLD': 0.0, 'SELL': -1.0}
        scores['technical'] = technical_map.get(getattr(technical, 'signal', 'HOLD'), 0.0)

        visual_map = {'BULLISH': 1.0, 'NEUTRAL': 0.0, 'BEARISH': -1.0, 'UNCLEAR': 0.0}
        # Asumiendo que visual_analysis puede no tener 'overall_visual_assessment' si falla
        visual_assessment = getattr(visual, 'overall_visual_assessment', 'UNCLEAR')
        if visual_assessment is None:
            visual_assessment = 'UNCLEAR' # Manejar None explícitamente
        scores['visual'] = visual_map.get(visual_assessment, 0.0)


        qabba_map = {'BUY_QABBA': 1.0, 'HOLD_QABBA': 0.0, 'SELL_QABBA': -1.0, 'NEUTRAL_QABBA': 0.0}
        qabba_signal_val = getattr(qabba, 'qabba_signal', 'NEUTRAL_QABBA') # Renombrado
        if qabba_signal_val is None:
            qabba_signal_val = 'NEUTRAL_QABBA' # Manejar None
        scores['qabba'] = qabba_map.get(qabba_signal_val, 0.0)


        if hasattr(sentiment, 'confidence_score') and getattr(sentiment, 'confidence_score') is not None:
            scores['sentiment'] *= getattr(sentiment, 'confidence_score', 0.5)
        if hasattr(technical, 'confidence_level') and getattr(technical, 'confidence_level') is not None:
            conf_map = {'LOW': 0.5, 'MEDIUM': 0.75, 'HIGH': 1.0}
            scores['technical'] *= conf_map.get(getattr(technical, 'confidence_level', 'MEDIUM'), 0.75)
        if hasattr(qabba, 'qabba_confidence') and getattr(qabba, 'qabba_confidence') is not None:
             scores['qabba'] *= getattr(qabba, 'qabba_confidence', 0.5)
        return scores

    def _detect_market_condition(self, metrics: Dict[str, float]) -> str:
        atr_val = metrics.get('atr', 0.0)
        last_price_val = metrics.get('last_price', 1.0)
        atr_pct = (atr_val / last_price_val) * 100 if last_price_val > 0 else 0
        adx = metrics.get('adx', 0)
        ma50 = metrics.get('ma50', 0.0)
        ma_diff = 0.0
        if ma50 > 0 and last_price_val > 0 :
            ma_diff = ((last_price_val - ma50) / ma50) * 100

        if atr_pct > 2.0:
            return 'high_volatility'
        elif adx > 25 and abs(ma_diff) > 1.0:
            return 'trending'
        else:
            return 'ranging'

    def _adjust_weights_for_market(self, market_condition_label: str) -> Dict[str, float]:
        base_weights = self.weights.copy()
        adjustments = self.market_condition_adjustments.get(
            market_condition_label,
            {'sentiment': 1.0, 'technical': 1.0, 'visual': 1.0, 'qabba': 1.0}
        )
        adjusted = {agent_name: base_weight * adjustments.get(agent_name, 1.0)
                    for agent_name, base_weight in base_weights.items()}
        total = sum(adjusted.values())
        if total > 0:
            return {agent_name: weight / total for agent_name, weight in adjusted.items()}
        return adjusted

    def _score_to_decision(self, score: float, scores: Dict[str, float], confidences: Dict[str, float]) -> str:
        # Nuevo: lógica de convergencia flexible
        # Si al menos dos agentes tienen score > 0.7 y están alineados (ambos positivos o ambos negativos), operar aunque la convergencia global sea baja
        bullish = [k for k, v in scores.items() if v > 0.7 and confidences.get(k, 0.0) > 0.8]
        bearish = [k for k, v in scores.items() if v < -0.7 and confidences.get(k, 0.0) > 0.8]
        if len(bullish) >= 2:
            return "BUY"
        if len(bearish) >= 2:
            return "SELL"
        # Sistema de desempate: priorizar agente con mejor performance histórica
        if abs(score) <= 0.3:
            best_agent = max(self.performance_metrics, key=lambda k: self.performance_metrics[k]['correct'] / (self.performance_metrics[k]['total']+1e-6))
            if scores.get(best_agent, 0.0) > 0.5:
                return "BUY"
            elif scores.get(best_agent, 0.0) < -0.5:
                return "SELL"
        # Lógica original
        if score > 0.3:
            return "BUY"
        elif score < -0.3:
            return "SELL"
        else:
            return "HOLD"

    def _calculate_confidence(self, scores: Dict[str, float], weighted_score: float) -> float:
        convergence = self._calculate_convergence(scores)
        signal_strength = abs(weighted_score)
        config_performance = self._get_configuration_performance(scores)
        confidence = (convergence * 0.4 + signal_strength * 0.4 + config_performance * 0.2)
        return min(max(confidence, 0.0), 1.0)

    def _calculate_convergence(self, scores: Dict[str, float]) -> float:
        values = list(scores.values())
        if not values:
            return 0.0
        # Manejar None en scores
        valid_values = [v for v in values if isinstance(v, (int, float))]
        if not valid_values:
            return 0.0

        signs = [1 if v > 0.1 else -1 if v < -0.1 else 0 for v in valid_values]
        if all(s == signs[0] for s in signs) and signs[0] != 0:
            return 1.0
        
        std_dev = np.std(valid_values) if valid_values else 0.0
        return 1.0 - min(std_dev, 1.0)

    def _get_configuration_performance(self, scores: Dict[str, float]) -> float:
        similar_configs = 0
        successful_configs = 0
        history_to_check = self.prediction_history.get('sentiment', [])
        for hist_entry in history_to_check:
            if isinstance(hist_entry, dict) and 'scores' in hist_entry and isinstance(hist_entry['scores'], dict):
                similarity = self._calculate_similarity(scores, hist_entry['scores'])
                if similarity > 0.8:
                    similar_configs += 1
                    if hist_entry.get('profitable', False):
                        successful_configs += 1
        if similar_configs > 5:
            return successful_configs / similar_configs
        return 0.5

    def _calculate_similarity(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> float:
        diff_sum = 0.0
        count = 0
        for agent_name, score1_val in scores1.items():
            # Asegurarse que score1_val y scores2.get(agent_name) son numéricos antes de restar
            s1 = score1_val if isinstance(score1_val, (int, float)) else 0.0
            s2 = scores2.get(agent_name)
            s2 = s2 if isinstance(s2, (int, float)) else 0.0
            
            if agent_name in scores2:
                diff_sum += abs(s1 - s2)
                count += 1
        if count == 0:
            return 0.0
        avg_diff = diff_sum / count
        # Asumiendo que los scores normalizados están entre -1 y 1, la diferencia máxima es 2.
        # avg_diff / 2.0 normaliza la diferencia a un rango de 0 a 1.
        return 1.0 - min(avg_diff / 2.0, 1.0)


    def update_performance(self, trade_result: Dict[str, Any]):
        if 'decision_context' not in trade_result:
            return
        profitable = trade_result.get('pnl_usd', 0) > 0
        pnl = trade_result.get('pnl_usd', 0)

        for agent_history in self.prediction_history.values():
            if agent_history and isinstance(agent_history[-1], dict):
                agent_history[-1]['profitable'] = profitable
                agent_history[-1]['pnl'] = pnl
        
        context = trade_result['decision_context']
        actual_direction = "BUY" if pnl > 0 else ("SELL" if pnl < 0 else "HOLD")

        for agent_name, metrics in self.performance_metrics.items():
            agent_data_key = f"{agent_name}_analysis"
            agent_data = context.get(agent_data_key, {})
            agent_signal = self._extract_agent_signal(agent_name, agent_data)
            metrics['total'] += 1
            if self._agent_was_correct(agent_signal, actual_direction):
                metrics['correct'] += 1
            metrics['profit_contribution'] += pnl * self.weights.get(agent_name, 0.25)

        self._update_weights()
        self._save_performance_history()

    def _update_weights(self):
        new_weights = {}
        num_agents = len(self.weights)
        if num_agents == 0:
            return # Evitar división por cero

        for agent_name, current_weight in self.weights.items():
            metrics = self.performance_metrics.get(agent_name, {'total': 0, 'correct': 0, 'profit_contribution': 0.0})
            if metrics['total'] > 10:
                accuracy = metrics['correct'] / metrics['total']
                avg_profit = metrics['profit_contribution'] / metrics['total']
                profit_factor = np.tanh(avg_profit / 10) + 1
                base_weight = 1.0 / num_agents
                performance_multiplier = accuracy * profit_factor
                new_weights[agent_name] = base_weight * performance_multiplier
            else:
                new_weights[agent_name] = current_weight
        
        total_new_weight = sum(new_weights.values())
        if total_new_weight > 0:
            for agent_name_key in new_weights: # Renombrar para evitar conflicto
                new_weights[agent_name_key] /= total_new_weight
        
        for agent_name_key_update in self.weights: # Renombrar para evitar conflicto
            self.weights[agent_name_key_update] = 0.8 * self.weights[agent_name_key_update] + 0.2 * new_weights.get(agent_name_key_update, self.weights[agent_name_key_update])
        logger.info(f"Pesos actualizados: {self.weights}")

    def _extract_agent_signal(self, agent_name: str, agent_data: Dict) -> str:
        if agent_name == 'sentiment':
            return agent_data.get('overall_sentiment', 'NEUTRAL')
        elif agent_name == 'technical':
            return agent_data.get('signal', 'HOLD')
        elif agent_name == 'visual':
            visual_assessment = agent_data.get('overall_visual_assessment', 'UNCLEAR')
            return visual_assessment if visual_assessment is not None else 'UNCLEAR'
        elif agent_name == 'qabba':
            qabba_signal = agent_data.get('qabba_signal', 'NEUTRAL_QABBA')
            return qabba_signal if qabba_signal is not None else 'NEUTRAL_QABBA'
        return 'NEUTRAL'

    def _agent_was_correct(self, agent_signal: str, actual_direction: str) -> bool:
        bullish_signals = ['BUY', 'POSITIVE', 'BULLISH', 'BUY_QABBA']
        bearish_signals = ['SELL', 'NEGATIVE', 'BEARISH', 'SELL_QABBA']
        neutral_signals = ['HOLD', 'NEUTRAL', 'UNCLEAR', 'NEUTRAL_QABBA']

        if actual_direction == "BUY" and agent_signal in bullish_signals:
            return True
        elif actual_direction == "SELL" and agent_signal in bearish_signals:
            return True
        elif actual_direction == "HOLD" and agent_signal in neutral_signals:
            return True
        return False

    def _log_prediction(self, decision, confidence, scores, contributions,
                       market_condition_label, sentiment, technical, visual, qabba):
        prediction = {
            'timestamp': datetime.now().isoformat(), 'decision': decision, 'confidence': confidence,
            'scores': scores, 'contributions': contributions, 'market_condition': market_condition_label,
            'weights': self.weights.copy()
        }
        for agent_history in self.prediction_history.values():
             agent_history.append(prediction.copy())

    def _save_performance_history(self):
        data = {
            'weights': self.weights, 'performance_metrics': self.performance_metrics,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(self.performance_log_path, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Error guardando historial de performance: {e}")

    def _load_performance_history(self):
        if self.performance_log_path.exists():
            try:
                with open(self.performance_log_path, 'r') as f: lines = f.readlines()
                if lines:
                    last_data = json.loads(lines[-1])
                    self.weights = last_data.get('weights', self.weights)
                    self.performance_metrics = last_data.get('performance_metrics', self.performance_metrics)
                    logger.info(f"Historial de performance cargado. Pesos: {self.weights}")
            except Exception as e:
                logger.error(f"Error cargando historial de performance: {e}")

    def get_weight_distribution_chart(self) -> Dict[str, Any]:
        return {
            'weights': self.weights,
            'performance': {
                agent_name: {
                    'accuracy': (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0,
                    'total_trades': metrics['total'],
                    'profit_contribution': metrics['profit_contribution']
                } for agent_name, metrics in self.performance_metrics.items()
            }
        }

class EnhancedDecisionAgent(Agent):
    name: ClassVar[str] = "EnhancedDecisionAgent"
    role: ClassVar[str] = "Agente de Decisión Estratégica de Trading"
    goal: ClassVar[str] = "Tomar decisiones finales de trading (COMPRAR, VENDER, MANTENER) basadas en el análisis ponderado de múltiples agentes especializados."
    backstory: ClassVar[str] = (
        "Un sofisticado agente de IA diseñado para sintetizar información de analistas de sentimiento, "
        "técnicos, visuales y cuantitativos. Utiliza un sistema de ponderación dinámica para adaptar su "
        "proceso de toma de decisiones a las condiciones cambiantes del mercado y al rendimiento histórico "
        "de cada agente, buscando optimizar la rentabilidad y gestionar el riesgo."
    )
    # tools: ClassVar[List[Any]] = []

    # Definir _weighting_system como PrivateAttr
    _weighting_system: DynamicWeightingSystem = PrivateAttr()

    def __init__(self, **kwargs):
        current_name = kwargs.pop('name', EnhancedDecisionAgent.name)
        current_role = kwargs.pop('role', EnhancedDecisionAgent.role)
        current_goal = kwargs.pop('goal', EnhancedDecisionAgent.goal)
        current_backstory = kwargs.pop('backstory', EnhancedDecisionAgent.backstory)
        
        super().__init__(
            name=current_name,
            role=current_role,
            goal=current_goal,
            backstory=current_backstory,
            **kwargs
        )
        # Inicializar el atributo privado después de llamar a super().__init__
        self._weighting_system = DynamicWeightingSystem()

    def run(
        self,
        sentiment_analysis: SentimentOutput,
        numerical_technical_analysis: TechnicalAnalysisOutput,
        visual_technical_analysis: Any, # Reemplazar con EnhancedVisualChartAnalysisOutput
        qabba_validation_analysis: Any, # Reemplazar con QABBAAnalysisOutput
        current_tech_metrics: Optional[Dict[str, float]] = None
    ) -> FinalDecisionOutput:

        # Usar self._weighting_system
        decision, confidence, weight_details = self._weighting_system.calculate_weighted_decision(
            sentiment_analysis,
            numerical_technical_analysis,
            visual_technical_analysis,
            qabba_validation_analysis,
            current_tech_metrics or {}
        )

        combined_reasoning = self._generate_weighted_reasoning(
            decision, confidence, weight_details,
            sentiment_analysis, numerical_technical_analysis,
            visual_technical_analysis, qabba_validation_analysis
        )

        confidence_level: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
        if confidence > 0.8:
            confidence_level = "HIGH"
        elif confidence > 0.6:
            confidence_level = "MEDIUM"
        
        conflicting_signals = self._identify_conflicts(weight_details.get('raw_scores', {}))

        return FinalDecisionOutput(
            final_decision=decision,
            combined_reasoning=combined_reasoning,
            confidence_in_decision=confidence_level,
            key_conflicting_signals=conflicting_signals
        )

    def _generate_weighted_reasoning(
        self, decision: str, confidence: float, weight_details: Dict[str, Any],
        sentiment: Any, technical: Any, visual: Any, qabba: Any
    ) -> str:
        reasoning = f"DECISIÓN FINAL: {decision} (Confianza: {confidence:.2%})\n\n"
        reasoning += f"ANÁLISIS PONDERADO:\n"
        reasoning += f"- Condición de Mercado: {weight_details.get('market_condition', 'N/A')}\n"
        reasoning += f"- Convergencia de Señales: {weight_details.get('convergence', 0.0):.2%}\n"
        reasoning += f"\nCONTRIBUCIONES POR AGENTE:\n"

        contributions = weight_details.get('contributions', {})
        for agent_name, contrib_data in contributions.items():
            reasoning += f"\n{agent_name.upper()}:\n"
            reasoning += f"  - Señal: {self._get_agent_signal_text(agent_name, contrib_data.get('score',0.0))}\n"
            reasoning += f"  - Peso: {contrib_data.get('weight',0.0):.2%}\n"
            reasoning += f"  - Contribución: {contrib_data.get('contribution',0.0):.3f}\n"

        reasoning += f"\nSCORE PONDERADO FINAL: {weight_details.get('weighted_score',0.0):.3f}\n"

        if abs(weight_details.get('weighted_score',0.0)) > 0.5:
            reasoning += "\nSeñal fuerte detectada con alta convergencia entre agentes."
        elif weight_details.get('convergence', 1.0) < 0.5:
            reasoning += "\nSeñales mixtas detectadas. Decisión basada en agentes con mejor performance histórico."
        return reasoning.strip()

    def _get_agent_signal_text(self, agent_name: str, score: float) -> str:
        # agent_name no se usa aquí, pero se mantiene por consistencia
        if not isinstance(score, (int, float)): score = 0.0 # Manejar None o tipos incorrectos
        if score > 0.5: return "Fuertemente Alcista"
        elif score > 0: return "Alcista"
        elif score < -0.5: return "Fuertemente Bajista"
        elif score < 0: return "Bajista"
        else: return "Neutral"

    def _identify_conflicts(self, scores: Dict[str, float]) -> List[str]:
        conflicts = []
        if not scores: return conflicts

        # Filtrar scores que no son numéricos o son None
        valid_scores = {k: v for k, v in scores.items() if isinstance(v, (int, float))}

        bullish = [agent_name for agent_name, score_val in valid_scores.items() if score_val > 0.1]
        bearish = [agent_name for agent_name, score_val in valid_scores.items() if score_val < -0.1]

        if bullish and bearish:
            conflicts.append(f"Divergencia: {', '.join(bullish)} alcistas vs {', '.join(bearish)} bajistas")

        score_values = list(valid_scores.values())
        if len(score_values) >= 2 and (max(score_values) - min(score_values) > 1.5):
            conflicts.append("Alta divergencia en la fuerza de las señales")
        return conflicts
