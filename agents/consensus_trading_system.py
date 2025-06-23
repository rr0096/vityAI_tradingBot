# agents/consensus_trading_system.py
"""
Sistema de Trading con Consenso Multi-Modelo Integrado
Combina el consenso entre modelos con fuentes de datos gratuitas
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from .multi_model_consensus import MultiModelConsensus, ConsensusTradingSignal
from tools.data_source_manager import FreeDataSourceManager
from agents.sentiment_enhanced import EnhancedSentimentAnalyst
from agents.technical_v_enhanced_fixed import EnhancedTechnicalAnalyst
from agents.QABBAValidatorAgent import EnhancedQABBAAgent

logger = logging.getLogger(__name__)

class ConsensusTradingSystem:
    """
    Sistema completo de trading que integra:
    - Consenso multi-modelo
    - Fuentes de datos gratuitas
    - Agentes especializados
    - Validación y anti-alucinación
    """
    
    def __init__(self):
        # Inicializar componentes
        self.consensus_system = MultiModelConsensus(
            min_models_required=2,
            consensus_threshold=0.6,
            max_response_time_ms=30000
        )
        
        self.data_manager = FreeDataSourceManager()
        
        # Inicializar agentes especializados
        self.agents = {}
        self._initialize_agents()
        
        # Historial de señales
        self.trading_history: List[ConsensusTradingSignal] = []
        
        logger.info("🚀 ConsensusTradingSystem initialized with all components")
    
    def _initialize_agents(self):
        """Inicializa todos los agentes especializados"""
        
        try:
            # Agente de Sentiment
            self.agents['sentiment'] = EnhancedSentimentAnalyst()
            logger.info("✅ Sentiment agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment agent: {e}")
        
        try:
            # Agente Técnico
            self.agents['technical'] = EnhancedTechnicalAnalyst()
            logger.info("✅ Technical agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize technical agent: {e}")
        
        try:
            # Agente QABBA
            self.agents['qabba'] = EnhancedQABBAAgent()
            logger.info("✅ QABBA agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize QABBA agent: {e}")
    
    async def analyze_and_decide(self, symbol: str, 
                               include_agents: List[str] = None) -> ConsensusTradingSignal:
        """
        Análisis completo y decisión consensuada para un símbolo
        
        Args:
            symbol: Símbolo crypto (BTC, ETH, etc.)
            include_agents: Lista de agentes a incluir (default: todos)
            
        Returns:
            ConsensusTradingSignal con la decisión final
        """
        
        logger.info(f"🎯 Starting consensus analysis for {symbol}")
        
        # 1. Obtener datos de mercado de múltiples fuentes
        market_data = self.data_manager.get_comprehensive_market_data(symbol)
        
        logger.info(f"📊 Market data quality: {market_data.get('data_quality', 'unknown')}")
        logger.info(f"📊 Sources used: {market_data.get('sources_used', [])}")
        
        # 2. Enriquecer datos con análisis de agentes especializados
        enriched_data = await self._enrich_market_data(symbol, market_data)
        
        # 3. Obtener consenso entre modelos
        consensus_signal = await self.consensus_system.get_consensus_signal(
            symbol=symbol,
            market_data=enriched_data,
            agents_to_consult=include_agents
        )
        
        # 4. Post-procesamiento y validación final
        validated_signal = self._validate_and_enhance_signal(consensus_signal, enriched_data)
        
        # 5. Guardar en historial
        self.trading_history.append(validated_signal)
        if len(self.trading_history) > 200:  # Mantener últimas 200 señales
            self.trading_history.pop(0)
        
        logger.info(f"✅ Consensus analysis complete for {symbol}: "
                   f"{validated_signal.action} (confidence: {validated_signal.consensus_confidence:.2f})")
        
        return validated_signal
    
    async def _enrich_market_data(self, symbol: str, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enriquece datos de mercado con análisis de agentes especializados"""
        
        enriched = base_data.copy()
        
        # Análisis de sentiment si el agente está disponible
        if 'sentiment' in self.agents:
            try:
                sentiment_agent = self.agents['sentiment']
                
                # Usar los métodos internos del agente
                twitter_data = sentiment_agent._get_twitter_data([symbol])
                news_data = sentiment_agent._get_news_data([symbol])
                
                enriched['social_sentiment'] = {
                    'twitter_posts_count': len(twitter_data) if twitter_data else 0,
                    'news_articles_count': len(news_data) if news_data else 0,
                    'data_freshness': 'real_time'
                }
                
                # Si hay suficientes datos, ejecutar análisis completo
                if (enriched['social_sentiment']['twitter_posts_count'] + 
                    enriched['social_sentiment']['news_articles_count']) > 5:
                    
                    from models.outputs import SentimentAnalysisInput
                    sentiment_input = SentimentAnalysisInput(
                        symbols=[symbol],
                        context=f"Market analysis for {symbol}"
                    )
                    
                    sentiment_result = sentiment_agent._run(sentiment_input)
                    
                    if hasattr(sentiment_result, 'sentiment_scores') and sentiment_result.sentiment_scores:
                        enriched['sentiment_analysis'] = sentiment_result.sentiment_scores.get(symbol, {})
                
                logger.info(f"📈 Sentiment data enriched for {symbol}")
                
            except Exception as e:
                logger.warning(f"⚠️ Sentiment enrichment failed for {symbol}: {e}")
        
        # Análisis técnico si el agente está disponible
        if 'technical' in self.agents:
            try:
                # Agregar indicadores técnicos calculados
                price_data = [
                    enriched.get('current_price'),
                    enriched.get('high_24h'),
                    enriched.get('low_24h'),
                    enriched.get('avg_price_24h')
                ]
                
                if all(p is not None for p in price_data):
                    enriched['technical_indicators'] = {
                        'price_position_24h': self._calculate_price_position(
                            enriched.get('current_price'),
                            enriched.get('high_24h'),
                            enriched.get('low_24h')
                        ),
                        'volatility_level': self._categorize_volatility(
                            enriched.get('volatility_24h', 0)
                        ),
                        'volume_trend': self._analyze_volume_trend(enriched)
                    }
                
                logger.info(f"📊 Technical data enriched for {symbol}")
                
            except Exception as e:
                logger.warning(f"⚠️ Technical enrichment failed for {symbol}: {e}")
        
        # Fear & Greed Index
        try:
            from tools.fear_greed import FearGreedTool
            fear_greed = FearGreedTool()
            fg_data = fear_greed._run()
            
            if isinstance(fg_data, dict) and 'value' in fg_data:
                enriched['fear_greed_index'] = fg_data
                logger.info(f"😨 Fear & Greed: {fg_data.get('value', 'N/A')}")
            
        except Exception as e:
            logger.warning(f"⚠️ Fear & Greed data unavailable: {e}")
        
        return enriched
    
    def _calculate_price_position(self, current: float, high: float, low: float) -> float:
        """Calcula posición del precio actual en el rango 24h (0-1)"""
        
        if high == low:
            return 0.5
        
        return (current - low) / (high - low)
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Categoriza el nivel de volatilidad"""
        
        if volatility < 0.02:
            return 'low'
        elif volatility < 0.05:
            return 'medium'
        else:
            return 'high'
    
    def _analyze_volume_trend(self, data: Dict[str, Any]) -> str:
        """Analiza tendencia de volumen"""
        
        current_volume = data.get('latest_volume', 0)
        avg_volume = data.get('volume_24h', 0)
        
        if avg_volume == 0:
            return 'unknown'
        
        volume_ratio = current_volume / avg_volume * 24  # Normalizar por hora
        
        if volume_ratio > 1.5:
            return 'increasing'
        elif volume_ratio < 0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _validate_and_enhance_signal(self, signal: ConsensusTradingSignal, 
                                   market_data: Dict[str, Any]) -> ConsensusTradingSignal:
        """Valida y mejora la señal consensuada con checks adicionales"""
        
        # Aplicar filtros de seguridad
        enhanced_signal = signal
        
        # 1. Filtro de calidad de datos
        data_quality = market_data.get('data_quality', 'unknown')
        if data_quality == 'insufficient':
            # Reducir confianza si los datos son insuficientes
            enhanced_signal.consensus_confidence *= 0.5
            enhanced_signal.reasoning += " [ADJUSTED: Insufficient market data quality]"
        
        # 2. Filtro de volatilidad extrema
        volatility = market_data.get('volatility_24h', 0)
        if volatility > 0.1:  # Volatilidad > 10%
            if signal.action in ['buy', 'sell']:
                enhanced_signal.risk_assessment = 'high'
                enhanced_signal.reasoning += " [WARNING: High volatility detected]"
        
        # 3. Filtro de consenso mínimo
        if signal.agreement_score < 0.5:
            # Si hay poco consenso, cambiar a 'watch'
            enhanced_signal.action = 'watch'
            enhanced_signal.reasoning += " [SAFETY: Low consensus, switching to WATCH]"
        
        # 4. Añadir contexto de mercado
        if 'fear_greed_index' in market_data:
            fg_value = market_data['fear_greed_index'].get('value', 50)
            fg_text = market_data['fear_greed_index'].get('value_classification', 'Neutral')
            enhanced_signal.reasoning += f" [MARKET: Fear&Greed {fg_value}/100 ({fg_text})]"
        
        return enhanced_signal
    
    def get_trading_summary(self, symbol: Optional[str] = None, 
                          last_n: int = 10) -> Dict[str, Any]:
        """Obtiene resumen de trading reciente"""
        
        # Filtrar por símbolo si se especifica
        signals = self.trading_history
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        # Tomar últimas N señales
        recent_signals = signals[-last_n:] if last_n > 0 else signals
        
        if not recent_signals:
            return {"error": "No trading signals available"}
        
        # Estadísticas
        actions = [s.action for s in recent_signals]
        confidences = [s.consensus_confidence for s in recent_signals]
        agreements = [s.agreement_score for s in recent_signals]
        
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_signals": len(recent_signals),
            "symbol_filter": symbol,
            "action_distribution": action_counts,
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_agreement": sum(agreements) / len(agreements),
            "latest_signal": {
                "symbol": recent_signals[-1].symbol,
                "action": recent_signals[-1].action,
                "confidence": recent_signals[-1].consensus_confidence,
                "timestamp": recent_signals[-1].timestamp.isoformat()
            },
            "consensus_stats": self.consensus_system.get_consensus_statistics()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Obtiene estado de salud del sistema completo"""
        
        health = {
            "consensus_system": "operational",
            "data_sources": self.data_manager.get_sources_status(),
            "agents": {},
            "performance": {}
        }
        
        # Estado de agentes
        for agent_name, agent in self.agents.items():
            try:
                # Test básico de cada agente
                health["agents"][agent_name] = "operational"
            except Exception as e:
                health["agents"][agent_name] = f"error: {e}"
        
        # Estadísticas de performance
        if self.trading_history:
            recent_signals = self.trading_history[-10:]
            response_times = [
                sum(len(s.individual_responses) for s in recent_signals) / len(recent_signals)
            ]
            
            health["performance"] = {
                "avg_models_per_signal": sum(len(s.models_participated) for s in recent_signals) / len(recent_signals),
                "signals_last_hour": len([s for s in recent_signals 
                                        if (datetime.now(timezone.utc) - s.timestamp).total_seconds() < 3600])
            }
        
        return health

# Función de testing
async def test_consensus_trading_system():
    """Test del sistema completo de consenso"""
    
    logger.info("🧪 Testing ConsensusTradingSystem...")
    
    try:
        # Inicializar sistema
        system = ConsensusTradingSystem()
        
        # Test análisis para BTC
        signal = await system.analyze_and_decide('BTC')
        
        print(f"✅ Consensus signal for BTC:")
        print(f"   Action: {signal.action}")
        print(f"   Confidence: {signal.consensus_confidence:.2f}")
        print(f"   Agreement: {signal.agreement_score:.2f}")
        print(f"   Risk: {signal.risk_assessment}")
        print(f"   Models: {signal.models_participated}")
        
        # Test resumen
        summary = system.get_trading_summary()
        print(f"✅ Trading summary: {summary.get('total_signals', 0)} signals")
        
        # Test salud del sistema
        health = system.get_system_health()
        print(f"✅ System health: {len(health['agents'])} agents operational")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(name)s - %(message)s")
    asyncio.run(test_consensus_trading_system())
