# agents/multi_model_consensus.py
"""
Multi-Model Consensus System for FenixTradingBot
Implementa consenso entre múltiples modelos LLM para decisiones críticas de trading
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import Counter
import statistics
import json

from pydantic import BaseModel, Field
from config.modern_models import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Respuesta de un modelo individual"""
    model_name: str
    agent_type: str
    response: Dict[str, Any]
    confidence: float
    latency_ms: float
    success: bool
    error_msg: Optional[str] = None

class ConsensusTradingSignal(BaseModel):
    """Señal de trading consensuada entre múltiples modelos"""
    action: Literal["buy", "sell", "hold", "watch"]
    symbol: str
    consensus_confidence: float = Field(ge=0.0, le=1.0)
    individual_responses: List[Dict[str, Any]] = Field(default_factory=list)
    voting_results: Dict[str, int] = Field(default_factory=dict)
    reasoning: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    models_participated: List[str] = Field(default_factory=list)
    agreement_score: float = Field(ge=0.0, le=1.0)
    risk_assessment: Literal["low", "medium", "high"] = "medium"

class MultiModelConsensus:
    """
    Sistema de consenso entre múltiples modelos LLM para trading
    """
    
    def __init__(self, 
                 min_models_required: int = 2,
                 consensus_threshold: float = 0.6,
                 max_response_time_ms: float = 30000):
        """
        Args:
            min_models_required: Mínimo de modelos que deben responder
            consensus_threshold: Umbral de consenso requerido (0.6 = 60%)
            max_response_time_ms: Timeout máximo por modelo
        """
        self.min_models_required = min_models_required
        self.consensus_threshold = consensus_threshold
        self.max_response_time_ms = max_response_time_ms
        
        self.model_manager = ModelManager()
        self.consensus_history: List[ConsensusTradingSignal] = []
        
        # Configuración de modelos por agente
        self.agent_models = {
            'sentiment': ['qwen2.5:7b-instruct-q5_k_m'],
            'technical': ['deepseek-r1:7b-qwen-distill-q4_K_M'],
            'qabba': ['adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M'],
            'decision': ['qwen2.5:7b-instruct-q5_k_m']
        }
        
        # Pesos por tipo de agente para el consenso
        self.agent_weights = {
            'sentiment': 0.25,
            'technical': 0.35,
            'qabba': 0.20,
            'decision': 0.20
        }
        
        logger.info(f"MultiModelConsensus initialized with {min_models_required} min models, {consensus_threshold} threshold")

    async def get_consensus_signal(self, 
                                 symbol: str, 
                                 market_data: Dict[str, Any],
                                 agents_to_consult: List[str] = None) -> ConsensusTradingSignal:
        """
        Obtiene una señal de trading consensuada entre múltiples agentes/modelos
        
        Args:
            symbol: Símbolo crypto (ej: BTC, ETH)
            market_data: Datos de mercado para análisis
            agents_to_consult: Lista de agentes a consultar (default: todos)
            
        Returns:
            ConsensusTradingSignal con la decisión consensuada
        """
        
        if agents_to_consult is None:
            agents_to_consult = ['sentiment', 'technical', 'qabba', 'decision']
        
        logger.info(f"🗳️ Getting consensus for {symbol} from {len(agents_to_consult)} agents")
        
        # Consultar múltiples agentes en paralelo
        agent_responses = await self._query_multiple_agents(
            symbol, market_data, agents_to_consult
        )
        
        # Filtrar respuestas exitosas
        successful_responses = [r for r in agent_responses if r.success]
        
        if len(successful_responses) < self.min_models_required:
            logger.warning(f"⚠️ Only {len(successful_responses)} models responded successfully")
            return self._create_fallback_signal(symbol, agent_responses)
        
        # Calcular consenso
        consensus_signal = self._calculate_consensus(symbol, successful_responses)
        
        # Guardar en historial
        self.consensus_history.append(consensus_signal)
        if len(self.consensus_history) > 100:  # Mantener solo últimos 100
            self.consensus_history.pop(0)
        
        logger.info(f"✅ Consensus reached for {symbol}: {consensus_signal.action} "
                   f"(confidence: {consensus_signal.consensus_confidence:.2f})")
        
        return consensus_signal

    async def _query_multiple_agents(self, 
                                   symbol: str, 
                                   market_data: Dict[str, Any],
                                   agents: List[str]) -> List[ModelResponse]:
        """Consulta múltiples agentes en paralelo"""
        
        tasks = []
        for agent_type in agents:
            if agent_type in self.agent_models:
                for model_name in self.agent_models[agent_type]:
                    task = self._query_single_agent(agent_type, model_name, symbol, market_data)
                    tasks.append(task)
        
        # Ejecutar todas las consultas en paralelo con timeout
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.max_response_time_ms / 1000
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Consensus query timed out after {self.max_response_time_ms}ms")
            responses = [None] * len(tasks)
        
        # Procesar respuestas
        model_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, ModelResponse):
                model_responses.append(response)
            elif isinstance(response, Exception):
                logger.error(f"❌ Agent query failed: {response}")
        
        return model_responses

    async def _query_single_agent(self, 
                                agent_type: str, 
                                model_name: str,
                                symbol: str, 
                                market_data: Dict[str, Any]) -> ModelResponse:
        """Consulta un agente individual"""
        
        start_time = datetime.now()
        
        try:
            # Crear prompt específico para consenso
            prompt = self._create_consensus_prompt(agent_type, symbol, market_data)
            
            # Consultar modelo
            response = await self._call_model(model_name, prompt, agent_type)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            # Validar respuesta
            if self._validate_model_response(response, agent_type):
                return ModelResponse(
                    model_name=model_name,
                    agent_type=agent_type,
                    response=response,
                    confidence=response.get('confidence', 0.5),
                    latency_ms=latency,
                    success=True
                )
            else:
                return ModelResponse(
                    model_name=model_name,
                    agent_type=agent_type,
                    response={},
                    confidence=0.0,
                    latency_ms=latency,
                    success=False,
                    error_msg="Invalid response format"
                )
                
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"❌ Error querying {model_name} ({agent_type}): {e}")
            
            return ModelResponse(
                model_name=model_name,
                agent_type=agent_type,
                response={},
                confidence=0.0,
                latency_ms=latency,
                success=False,
                error_msg=str(e)
            )

    async def _call_model(self, model_name: str, prompt: str, agent_type: str) -> Dict[str, Any]:
        """Llama a un modelo específico usando ollama"""
        
        try:
            import ollama
            
            # Configuración específica por agente
            options = self._get_model_options(agent_type)
            
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options=options
            )
            
            # Intentar parsear JSON
            response_text = response.get('response', '')
            
            # Extraer JSON del response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                parsed_response = json.loads(json_match.group())
                return parsed_response
            else:
                # Fallback: crear respuesta estruturada básica
                return {
                    "action": "hold",
                    "confidence": 0.3,
                    "reasoning": f"Model {model_name} response could not be parsed as JSON"
                }
                
        except Exception as e:
            logger.error(f"❌ Error calling model {model_name}: {e}")
            raise

    def _create_consensus_prompt(self, agent_type: str, symbol: str, market_data: Dict[str, Any]) -> str:
        """Crea prompts específicos para consenso por tipo de agente"""
        
        base_context = f"""
Symbol: {symbol}
Current Price: {market_data.get('current_price', 'N/A')}
24h Change: {market_data.get('price_change_24h', 'N/A')}%
Volume: {market_data.get('volume_24h', 'N/A')}
Market Cap: {market_data.get('market_cap', 'N/A')}

You are participating in a multi-model consensus system for trading decisions.
Provide your analysis and recommendation in JSON format.

Required JSON structure:
{{
  "action": "buy|sell|hold|watch",
  "confidence": 0.XX,
  "reasoning": "Detailed explanation of your recommendation",
  "key_factors": ["factor1", "factor2", "factor3"],
  "risk_level": "low|medium|high"
}}
"""
        
        if agent_type == 'sentiment':
            return f"""
{base_context}

ROLE: Sentiment Analysis Specialist
Analyze market sentiment for {symbol} based on:
- Social media trends
- News sentiment
- Community discussions
- Fear & Greed indicators

Focus on sentiment-driven price movements and public perception.
"""
            
        elif agent_type == 'technical':
            return f"""
{base_context}

ROLE: Technical Analysis Specialist  
Analyze {symbol} based on:
- Price action and trends
- Technical indicators (RSI, MACD, Bollinger Bands)
- Support/resistance levels
- Volume analysis

Focus on chart patterns and technical signals.
"""
            
        elif agent_type == 'qabba':
            return f"""
{base_context}

ROLE: QABBA (Bollinger Bands) Specialist
Analyze {symbol} based on:
- Bollinger Bands position
- Volatility squeeze/expansion
- Band breakouts and reversals
- Statistical probabilities

Focus specifically on Bollinger Band signals and volatility patterns.
"""
            
        elif agent_type == 'decision':
            return f"""
{base_context}

ROLE: Final Decision Synthesizer
Provide a balanced trading recommendation for {symbol} considering:
- Overall market conditions
- Risk management principles
- Trade timing and execution
- Portfolio implications

Focus on practical trading decisions with risk awareness.
"""
        
        else:
            return base_context

    def _get_model_options(self, agent_type: str) -> Dict[str, Any]:
        """Obtiene opciones específicas por tipo de agente"""
        
        base_options = {
            'format': 'json',
            'temperature': 0.1,
            'top_p': 0.9,
            'repeat_penalty': 1.1
        }
        
        if agent_type == 'sentiment':
            base_options['temperature'] = 0.2  # Más variabilidad para sentiment
        elif agent_type == 'technical':
            base_options['temperature'] = 0.05  # Máxima precisión para análisis técnico
        elif agent_type == 'qabba':
            base_options['temperature'] = 0.05  # Precisión matemática
        elif agent_type == 'decision':
            base_options['temperature'] = 0.1   # Balanceado para decisiones
        
        return base_options

    def _validate_model_response(self, response: Dict[str, Any], agent_type: str) -> bool:
        """Valida que la respuesta del modelo tenga el formato correcto"""
        
        required_fields = ['action', 'confidence', 'reasoning']
        
        for field in required_fields:
            if field not in response:
                return False
        
        # Validar valores específicos
        if response['action'] not in ['buy', 'sell', 'hold', 'watch']:
            return False
        
        try:
            confidence = float(response['confidence'])
            if not (0.0 <= confidence <= 1.0):
                return False
        except (ValueError, TypeError):
            return False
        
        # Validar que reasoning no esté vacío
        if not response['reasoning'] or len(response['reasoning'].strip()) < 10:
            return False
        
        return True

    def _calculate_consensus(self, symbol: str, responses: List[ModelResponse]) -> ConsensusTradingSignal:
        """Calcula el consenso entre las respuestas de los modelos"""
        
        # Agrupar respuestas por acción
        action_votes = Counter()
        action_confidences = {}
        weighted_votes = {}
        
        for response in responses:
            action = response.response['action']
            confidence = response.confidence
            agent_type = response.agent_type
            weight = self.agent_weights.get(agent_type, 0.25)
            
            # Contar votos
            action_votes[action] += 1
            
            # Acumular confianzas por acción
            if action not in action_confidences:
                action_confidences[action] = []
            action_confidences[action].append(confidence)
            
            # Votos ponderados
            if action not in weighted_votes:
                weighted_votes[action] = 0
            weighted_votes[action] += confidence * weight
        
        # Determinar acción ganadora
        if weighted_votes:
            winning_action = max(weighted_votes.items(), key=lambda x: x[1])[0]
        else:
            winning_action = action_votes.most_common(1)[0][0] if action_votes else 'hold'
        
        # Calcular consenso y confianza
        total_responses = len(responses)
        winning_votes = action_votes[winning_action]
        agreement_score = winning_votes / total_responses
        
        # Confianza promedio para la acción ganadora
        avg_confidence = statistics.mean(action_confidences.get(winning_action, [0.5]))
        
        # Ajustar confianza por nivel de consenso
        consensus_confidence = avg_confidence * agreement_score
        
        # Evaluar riesgo basado en consenso
        if agreement_score >= 0.8 and avg_confidence >= 0.7:
            risk_level = "low"
        elif agreement_score >= 0.6 and avg_confidence >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Crear reasoning consensuado
        reasoning_parts = []
        for response in responses:
            agent_reasoning = response.response.get('reasoning', '')
            reasoning_parts.append(f"[{response.agent_type}]: {agent_reasoning[:100]}...")
        
        consensus_reasoning = f"Consensus decision based on {total_responses} models. " + \
                            f"Agreement: {agreement_score:.1%}. " + \
                            " | ".join(reasoning_parts)
        
        return ConsensusTradingSignal(
            action=winning_action,
            symbol=symbol,
            consensus_confidence=consensus_confidence,
            individual_responses=[r.response for r in responses],
            voting_results=dict(action_votes),
            reasoning=consensus_reasoning,
            models_participated=[r.model_name for r in responses],
            agreement_score=agreement_score,
            risk_assessment=risk_level
        )

    def _create_fallback_signal(self, symbol: str, responses: List[ModelResponse]) -> ConsensusTradingSignal:
        """Crea una señal de fallback cuando no hay suficientes respuestas"""
        
        successful_count = len([r for r in responses if r.success])
        
        return ConsensusTradingSignal(
            action="hold",
            symbol=symbol,
            consensus_confidence=0.2,
            individual_responses=[],
            voting_results={"hold": 1},
            reasoning=f"Insufficient model responses ({successful_count}/{self.min_models_required} required). "
                     f"Defaulting to HOLD for safety.",
            models_participated=[],
            agreement_score=0.0,
            risk_assessment="high"
        )

    def get_consensus_history(self, symbol: Optional[str] = None, 
                            last_n: int = 10) -> List[ConsensusTradingSignal]:
        """Obtiene el historial de consensos"""
        
        history = self.consensus_history
        
        if symbol:
            history = [signal for signal in history if signal.symbol == symbol]
        
        return history[-last_n:] if last_n > 0 else history

    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de consenso"""
        
        if not self.consensus_history:
            return {"error": "No consensus history available"}
        
        actions = [signal.action for signal in self.consensus_history]
        confidences = [signal.consensus_confidence for signal in self.consensus_history]
        agreements = [signal.agreement_score for signal in self.consensus_history]
        
        return {
            "total_signals": len(self.consensus_history),
            "action_distribution": dict(Counter(actions)),
            "avg_confidence": statistics.mean(confidences),
            "avg_agreement": statistics.mean(agreements),
            "high_confidence_signals": len([c for c in confidences if c >= 0.7]),
            "strong_consensus_signals": len([a for a in agreements if a >= 0.8])
        }

# Función helper para testing
async def test_consensus_system():
    """Test básico del sistema de consenso"""
    
    consensus = MultiModelConsensus(min_models_required=2)
    
    # Datos de prueba
    test_market_data = {
        'current_price': 45000,
        'price_change_24h': 2.5,
        'volume_24h': 1000000000,
        'market_cap': 850000000000
    }
    
    try:
        signal = await consensus.get_consensus_signal('BTC', test_market_data)
        print(f"✅ Test consensus signal: {signal.action} (confidence: {signal.consensus_confidence:.2f})")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Test del sistema
    import asyncio
    asyncio.run(test_consensus_system())
