# agents/json_validator.py
"""
Sistema robusto de validación y reparación JSON basado en la investigación.
Implementa las técnicas anti-alucinación específicas para trading financiero.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, TypeVar, List
from jsonschema import validate, ValidationError as JSONSchemaValidationError
from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

class TradingSignalValidator:
    """
    Validador principal de señales de trading con esquemas específicos
    y reparación automática de JSON malformado.
    """
    
    def __init__(self):
        # Esquemas base para diferentes tipos de señales
        self.base_trading_schema = {
            "type": "object",
            "required": ["signal", "reasoning"],
            "properties": {
                "signal": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "reasoning": {"type": "string", "minLength": 10},
                "confidence_level": {"type": ["string", "null"], "enum": ["LOW", "MEDIUM", "HIGH", None]},
                "key_patterns_observed": {"type": ["array", "null"]},
                "temporal_analysis": {"type": ["string", "null"]},
                "price_target": {"type": ["number", "null"]},
                "stop_loss_suggestion": {"type": ["number", "null"]},
                "market_phase": {"type": ["string", "null"], "enum": ["ACCUMULATION", "MARKUP", "DISTRIBUTION", "MARKDOWN", "UNCERTAIN", None]}
            }
        }
        
        self.qabba_schema = {
            "type": "object",
            "required": ["qabba_signal", "qabba_confidence", "reasoning_short"],
            "properties": {
                "qabba_signal": {
                    "type": "string", 
                    "enum": ["BUY_QABBA", "SELL_QABBA", "HOLD_QABBA", "NEUTRAL_QABBA"]
                },
                "qabba_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning_short": {"type": "string", "minLength": 10},
                "squeeze_detection": {"type": ["boolean", "null"]},
                "breakout_probability": {"type": ["number", "null"], "minimum": 0, "maximum": 1}
            }
        }
        
        self.visual_schema = {
            "type": "object",
            "required": ["overall_visual_assessment", "suggested_action_based_on_visuals", "reasoning"],
            "properties": {
                "overall_visual_assessment": {
                    "type": "string",
                    "enum": ["BULLISH", "BEARISH", "NEUTRAL", "UNCLEAR"]
                },
                "suggested_action_based_on_visuals": {
                    "type": "string",
                    "enum": ["CONSIDER_BUY", "CONSIDER_SELL", "WAIT_CONFIRMATION", "AVOID_TRADE"]
                },
                "reasoning": {"type": "string", "minLength": 20},
                "pattern_clarity_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "chart_timeframe_analyzed": {"type": "string"}
            }
        }
        
        # Add sentiment schema for sentiment analysis responses
        self.sentiment_schema = {
            "type": "object",
            "required": ["overall_sentiment", "reasoning"],
            "properties": {
                "overall_sentiment": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
                "positive_texts_count": {"type": "integer", "minimum": 0},
                "negative_texts_count": {"type": "integer", "minimum": 0},
                "neutral_texts_count": {"type": "integer", "minimum": 0},
                "reasoning": {"type": "string", "minLength": 10}
            }
        }

    def validate_and_repair(self, response_text: str, schema_type: str = "trading") -> Optional[Dict[str, Any]]:
        """
        Valida y repara JSON malformado usando múltiples estrategias.
        
        Args:
            response_text: Texto de respuesta del LLM
            schema_type: Tipo de esquema ("trading", "qabba", "visual")
            
        Returns:
            Dict validado o None si no se puede reparar
        """
        if not response_text or not response_text.strip():
            logger.warning("Response text is empty or None")
            return None
            
        schema = self._get_schema(schema_type)
        
        # Extraer JSON del texto
        json_content = self._extract_json_from_text(response_text)
        if not json_content:
            logger.warning("No JSON content found in response")
            return None
            
        # Intentar parsear directamente
        try:
            data = json.loads(json_content)
            validate(instance=data, schema=schema)
            logger.info(f"JSON válido en primer intento para schema {schema_type}")
            return data
        except (json.JSONDecodeError, JSONSchemaValidationError) as e:
            logger.debug(f"Direct parsing failed: {e}")
            
        # Intentar reparación de JSON malformado
        repaired_data = self._repair_json_malformed(json_content)
        if repaired_data:
            try:
                validate(instance=repaired_data, schema=schema)
                logger.info(f"JSON reparado exitosamente para schema {schema_type}")
                return repaired_data
            except JSONSchemaValidationError as e:
                logger.warning(f"Repaired JSON doesn't match schema: {e}")
                
        # Último intento: crear estructura mínima válida
        logger.warning(f"Creating minimal valid structure for {schema_type}")
        return self._create_minimal_valid_structure(schema_type, json_content)

    def _get_schema(self, schema_type: str) -> Dict[str, Any]:
        """Obtiene el esquema correspondiente al tipo."""
        schemas = {
            "trading": self.base_trading_schema,
            "qabba": self.qabba_schema,
            "visual": self.visual_schema,
            "sentiment": self.sentiment_schema
        }
        return schemas.get(schema_type, self.base_trading_schema)

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extrae contenido JSON del texto usando múltiples patrones."""
        text = text.strip()
        
        # Remover tags de thinking
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.MULTILINE)
        text = text.strip()
        
        # Patrones para extraer JSON
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # JSON balanceado
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                return match.group(1).strip()
                
        # Si no hay match con patrones, buscar primer { hasta último }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace + 1]
            
        return None

    def _repair_json_malformed(self, json_content: str) -> Optional[Dict[str, Any]]:
        """Repara JSON malformado usando técnicas específicas."""
        try:
            # Técnica 1: Reparar comillas faltantes en claves
            repaired = re.sub(r'(\w+):', r'"\1":', json_content)
            
            # Técnica 2: Reparar valores sin comillas
            repaired = re.sub(r': ([A-Z_]+)([,}])', r': "\1"\2', repaired)
            
            # Técnica 3: Reparar comas faltantes
            repaired = re.sub(r'"\s*\n\s*"', '",\n"', repaired)
            
            # Técnica 4: Remover comas finales
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
            
            # Técnica 5: Balancear llaves
            open_braces = repaired.count('{')
            close_braces = repaired.count('}')
            if open_braces > close_braces:
                repaired += '}' * (open_braces - close_braces)
            
            return json.loads(repaired)
            
        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
            return None

    def _create_minimal_valid_structure(self, schema_type: str, original_content: str) -> Dict[str, Any]:
        """Crea estructura mínima válida como último recurso."""
        error_reason = f"JSON malformado detectado. Contenido original: {original_content[:100]}..."
        
        if schema_type == "trading":
            return {
                "signal": "HOLD",
                "reasoning": error_reason,
                "confidence_level": "LOW",
                "key_patterns_observed": [],
                "temporal_analysis": "Analysis unavailable due to error",
                "price_target": None,
                "stop_loss_suggestion": None,
                "market_phase": "UNCERTAIN"
            }
        elif schema_type == "qabba":
            return {
                "qabba_signal": "NEUTRAL_QABBA",
                "qabba_confidence": 0.1,
                "reasoning_short": error_reason[:100],
                "squeeze_detection": None,
                "breakout_probability": None
            }
        elif schema_type == "visual":
            return {
                "overall_visual_assessment": "UNCLEAR",
                "suggested_action_based_on_visuals": "AVOID_TRADE",
                "reasoning": error_reason,
                "pattern_clarity_score": 0.0,
                "chart_timeframe_analyzed": "unknown"
            }
        elif schema_type == "sentiment":
            return {
                "overall_sentiment": "NEUTRAL",
                "positive_texts_count": 0,
                "negative_texts_count": 0,
                "neutral_texts_count": 0,
                "reasoning": error_reason
            }
        else:
            return {"error": "Unknown schema type", "reasoning": error_reason}


class ConstitutionalFinancialPrompt:
    """
    Implementa prompting constitucional específico para análisis financiero.
    Basado en las técnicas anti-alucinación de la investigación.
    """
    
    CONSTITUTIONAL_PRINCIPLES = [
        "NUNCA especules sobre precios sin disclaimer claro",
        "SIEMPRE distingue entre hechos históricos y predicciones",
        "RECONOCE incertidumbre cuando los datos son insuficientes",
        "PRIORIZA precisión sobre completitud",
        "CITA fuentes específicas para todas las afirmaciones",
        "ADMITE 'no_visible' cuando no puedes determinar visualmente"
    ]
    
    @classmethod
    def create_constitutional_prompt(cls, base_prompt: str, analysis_type: str) -> str:
        """
        Envuelve un prompt base con principios constitucionales.
        
        Args:
            base_prompt: Prompt original del agente
            analysis_type: Tipo de análisis ("sentiment", "technical", "visual", etc.)
            
        Returns:
            Prompt constitucional completo
        """
        constitutional_header = f"""
Eres un asistente de análisis financiero especializado en {analysis_type}. 
Antes de proporcionar cualquier análisis, debes seguir estos principios constitucionales:

CRÍTICA: Analiza tu respuesta inicial por inexactitudes potenciales
VERIFICA: Contrasta con los datos proporcionados
REVISA: Corrige cualquier afirmación especulativa
CITA: Proporciona fuentes específicas para todas las afirmaciones

Principios constitucionales obligatorios:
{chr(10).join(f"- {principle}" for principle in cls.CONSTITUTIONAL_PRINCIPLES)}

PROCESO REQUERIDO:
1. Realiza el análisis solicitado
2. Critica tu propia respuesta
3. Verifica contra los datos disponibles
4. Corrige cualquier especulación
5. Proporciona respuesta final en JSON

"""
        
        constitutional_footer = """

IMPORTANTE: Si no puedes determinar algo con certeza basándote en los datos proporcionados, 
indica claramente "no_determinable" o "insuficientes_datos" en lugar de especular.

Proporciona tu respuesta ÚNICAMENTE en formato JSON válido."""

        return constitutional_header + base_prompt + constitutional_footer


class MultiModelConsensus:
    """
    Sistema de consenso entre múltiples modelos para decisiones críticas.
    Implementa votación ponderada y umbral de confianza.
    """
    
    def __init__(self, consensus_threshold: float = 0.7, min_models: int = 2):
        self.consensus_threshold = consensus_threshold
        self.min_models = min_models
        self.validator = TradingSignalValidator()
        
    def get_consensus_decision(self, model_responses: List[Dict[str, Any]], 
                             weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Obtiene decisión de consenso entre múltiples respuestas de modelos.
        
        Args:
            model_responses: Lista de respuestas validadas de diferentes modelos
            weights: Pesos opcionales por modelo
            
        Returns:
            Decisión de consenso o señal de "HOLD" si no hay consenso
        """
        if len(model_responses) < self.min_models:
            return self._create_no_consensus_response("Insuficientes modelos para consenso")
            
        # Extraer acciones y confidencias
        actions = []
        confidences = []
        
        for i, response in enumerate(model_responses):
            if not response or 'action' not in response:
                continue
                
            action = response.get('action', 'HOLD')
            confidence = response.get('confidence', 0.0)
            
            # Aplicar peso si está disponible
            model_weight = weights.get(f"model_{i}", 1.0) if weights else 1.0
            weighted_confidence = confidence * model_weight
            
            actions.append(action)
            confidences.append(weighted_confidence)
        
        if not actions:
            return self._create_no_consensus_response("No hay acciones válidas en las respuestas")
            
        # Verificar consenso
        unique_actions = list(set(actions))
        
        if len(unique_actions) == 1:
            # Consenso completo
            consensus_action = unique_actions[0]
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_confidence >= self.consensus_threshold:
                return {
                    "action": consensus_action,
                    "confidence": avg_confidence,
                    "reasoning": f"Consenso completo entre {len(actions)} modelos",
                    "consensus_type": "FULL_CONSENSUS",
                    "participating_models": len(actions)
                }
        
        # Consenso mayoritario
        action_counts = {action: actions.count(action) for action in unique_actions}
        majority_action = max(action_counts.keys(), key=lambda x: action_counts[x])
        majority_ratio = action_counts[majority_action] / len(actions)
        
        if majority_ratio >= 0.6:  # 60% mayoría
            majority_indices = [i for i, action in enumerate(actions) if action == majority_action]
            majority_confidences = [confidences[i] for i in majority_indices]
            avg_confidence = sum(majority_confidences) / len(majority_confidences)
            
            if avg_confidence >= self.consensus_threshold:
                return {
                    "action": majority_action,
                    "confidence": avg_confidence,
                    "reasoning": f"Consenso mayoritario: {majority_ratio:.1%} de {len(actions)} modelos",
                    "consensus_type": "MAJORITY_CONSENSUS",
                    "participating_models": len(actions)
                }
        
        # Sin consenso suficiente
        return self._create_no_consensus_response(
            f"Sin consenso: {len(unique_actions)} acciones diferentes, "
            f"mayor consenso: {max(action_counts.values())}/{len(actions)}"
        )
    
    def _create_no_consensus_response(self, reason: str) -> Dict[str, Any]:
        """Crea respuesta estándar cuando no hay consenso."""
        return {
            "action": "HOLD",
            "confidence": 0.1,
            "reasoning": f"Sin consenso entre modelos: {reason}",
            "consensus_type": "NO_CONSENSUS",
            "participating_models": 0
        }
