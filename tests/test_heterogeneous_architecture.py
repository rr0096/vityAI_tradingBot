# tests/test_heterogeneous_architecture.py
"""
Tests para la nueva arquitectura heterogénea de modelos LLM.
Valida el sistema de consenso, validación JSON y gestión de memoria.
"""

import pytest
from unittest.mock import patch

from agents.json_validator import (
    TradingSignalValidator, 
    ConstitutionalFinancialPrompt,
    MultiModelConsensus
)
from config.heterogeneous_models import (
    HeterogeneousModelManager,
    ModelType,
    HeterogeneousModelConfig
)

class TestTradingSignalValidator:
    """Tests para el validador de señales de trading."""
    
    def setup_method(self):
        self.validator = TradingSignalValidator()
    
    def test_valid_json_direct_parsing(self):
        """Test parsing directo de JSON válido."""
        valid_response = '''
        {
            "action": "BUY",
            "confidence": 0.85,
            "reasoning": "Strong bullish signals detected with high volume confirmation",
            "price_target": 50000.0,
            "stop_loss": 45000.0
        }
        '''
        
        result = self.validator.validate_and_repair(valid_response, "trading")
        
        assert result is not None
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.85
        assert len(result["reasoning"]) > 20
    
    def test_json_repair_missing_quotes(self):
        """Test reparación de JSON con comillas faltantes."""
        malformed_response = '''
        {
            action: "BUY",
            confidence: 0.75,
            reasoning: "Good technical setup but missing quotes in keys"
        }
        '''
        
        result = self.validator.validate_and_repair(malformed_response, "trading")
        
        assert result is not None
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.75
    
    def test_json_repair_with_thinking_tags(self):
        """Test remoción de tags <think> antes del parsing."""
        response_with_thinking = '''
        <think>
        Let me analyze this market situation carefully...
        The indicators show bullish momentum...
        </think>
        
        {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Bullish momentum confirmed by multiple indicators"
        }
        '''
        
        result = self.validator.validate_and_repair(response_with_thinking, "trading")
        
        assert result is not None
        assert result["action"] == "BUY"
        assert "think" not in str(result).lower()
    
    def test_minimal_valid_structure_creation(self):
        """Test creación de estructura mínima válida."""
        completely_invalid = "This is not JSON at all, just random text"
        
        result = self.validator.validate_and_repair(completely_invalid, "trading")
        
        assert result is not None
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.1
        assert "JSON malformado" in result["reasoning"]
    
    def test_qabba_schema_validation(self):
        """Test validación específica para esquema QABBA."""
        qabba_response = '''
        {
            "qabba_signal": "BUY_QABBA",
            "qabba_confidence": 0.9,
            "reasoning_short": "Bollinger Bands squeeze with bullish breakout probability",
            "squeeze_detection": true,
            "breakout_probability": 0.85
        }
        '''
        
        result = self.validator.validate_and_repair(qabba_response, "qabba")
        
        assert result is not None
        assert result["qabba_signal"] == "BUY_QABBA"
        assert result["qabba_confidence"] == 0.9
        assert result["squeeze_detection"] is True
    
    def test_visual_schema_validation(self):
        """Test validación específica para esquema visual."""
        visual_response = '''
        {
            "overall_visual_assessment": "BULLISH",
            "suggested_action_based_on_visuals": "CONSIDER_BUY",
            "reasoning": "Clear uptrend with strong support levels visible",
            "pattern_clarity_score": 0.85,
            "chart_timeframe_analyzed": "1h"
        }
        '''
        
        result = self.validator.validate_and_repair(visual_response, "visual")
        
        assert result is not None
        assert result["overall_visual_assessment"] == "BULLISH"
        assert result["suggested_action_based_on_visuals"] == "CONSIDER_BUY"
        assert result["chart_timeframe_analyzed"] == "1h"


class TestConstitutionalFinancialPrompt:
    """Tests para prompting constitucional financiero."""
    
    def test_constitutional_prompt_creation(self):
        """Test creación de prompt constitucional."""
        base_prompt = "Analyze the current market sentiment for Bitcoin"
        
        constitutional_prompt = ConstitutionalFinancialPrompt.create_constitutional_prompt(
            base_prompt, "sentiment"
        )
        
        assert "principios constitucionales" in constitutional_prompt.lower()
        assert "nunca especules" in constitutional_prompt.lower()
        assert "reconoce incertidumbre" in constitutional_prompt.lower()
        assert base_prompt in constitutional_prompt
        assert "json" in constitutional_prompt.lower()
    
    def test_constitutional_principles_included(self):
        """Test que todos los principios constitucionales estén incluidos."""
        base_prompt = "Perform technical analysis"
        
        constitutional_prompt = ConstitutionalFinancialPrompt.create_constitutional_prompt(
            base_prompt, "technical"
        )
        
        principles = ConstitutionalFinancialPrompt.CONSTITUTIONAL_PRINCIPLES
        for principle in principles:
            # Verificar que la esencia de cada principio esté presente
            if "especules" in principle:
                assert "especul" in constitutional_prompt.lower()
            elif "distingue" in principle:
                assert "distingue" in constitutional_prompt.lower()


class TestMultiModelConsensus:
    """Tests para sistema de consenso multi-modelo."""
    
    def setup_method(self):
        self.consensus = MultiModelConsensus(consensus_threshold=0.7, min_models=2)
    
    def test_full_consensus_high_confidence(self):
        """Test consenso completo con alta confianza."""
        model_responses = [
            {"action": "BUY", "confidence": 0.85, "reasoning": "Model 1 analysis"},
            {"action": "BUY", "confidence": 0.80, "reasoning": "Model 2 analysis"},
            {"action": "BUY", "confidence": 0.90, "reasoning": "Model 3 analysis"}
        ]
        
        result = self.consensus.get_consensus_decision(model_responses)
        
        assert result["action"] == "BUY"
        assert result["consensus_type"] == "FULL_CONSENSUS"
        assert result["confidence"] >= 0.7
        assert result["participating_models"] == 3
    
    def test_majority_consensus(self):
        """Test consenso mayoritario."""
        model_responses = [
            {"action": "BUY", "confidence": 0.85, "reasoning": "Model 1 analysis"},
            {"action": "BUY", "confidence": 0.80, "reasoning": "Model 2 analysis"},
            {"action": "SELL", "confidence": 0.75, "reasoning": "Model 3 analysis"},
            {"action": "BUY", "confidence": 0.90, "reasoning": "Model 4 analysis"}
        ]
        
        result = self.consensus.get_consensus_decision(model_responses)
        
        assert result["action"] == "BUY"
        assert result["consensus_type"] == "MAJORITY_CONSENSUS"
        assert "mayoritario" in result["reasoning"]
    
    def test_no_consensus_insufficient_confidence(self):
        """Test sin consenso por confianza insuficiente."""
        model_responses = [
            {"action": "BUY", "confidence": 0.4, "reasoning": "Model 1 analysis"},
            {"action": "BUY", "confidence": 0.5, "reasoning": "Model 2 analysis"},
            {"action": "BUY", "confidence": 0.3, "reasoning": "Model 3 analysis"}
        ]
        
        result = self.consensus.get_consensus_decision(model_responses)
        
        assert result["action"] == "HOLD"
        assert result["consensus_type"] == "NO_CONSENSUS"
        assert result["confidence"] == 0.1
    
    def test_no_consensus_conflicting_signals(self):
        """Test sin consenso por señales conflictivas."""
        model_responses = [
            {"action": "BUY", "confidence": 0.85, "reasoning": "Model 1 analysis"},
            {"action": "SELL", "confidence": 0.80, "reasoning": "Model 2 analysis"},
            {"action": "HOLD", "confidence": 0.75, "reasoning": "Model 3 analysis"}
        ]
        
        result = self.consensus.get_consensus_decision(model_responses)
        
        assert result["action"] == "HOLD"
        assert result["consensus_type"] == "NO_CONSENSUS"
        assert "sin consenso" in result["reasoning"].lower()
    
    def test_weighted_consensus(self):
        """Test consenso con pesos por modelo."""
        model_responses = [
            {"action": "BUY", "confidence": 0.85, "reasoning": "High-weight model"},
            {"action": "SELL", "confidence": 0.80, "reasoning": "Low-weight model"}
        ]
        
        weights = {"model_0": 2.0, "model_1": 0.5}  # Primer modelo tiene más peso
        
        result = self.consensus.get_consensus_decision(model_responses, weights)
        
        # Debería favorecer el modelo con mayor peso
        assert result["action"] == "BUY"
    
    def test_insufficient_models(self):
        """Test con insuficientes modelos para consenso."""
        model_responses = [
            {"action": "BUY", "confidence": 0.85, "reasoning": "Only one model"}
        ]
        
        result = self.consensus.get_consensus_decision(model_responses)
        
        assert result["action"] == "HOLD"
        assert result["consensus_type"] == "NO_CONSENSUS"
        assert "insuficientes modelos" in result["reasoning"].lower()


class TestHeterogeneousModelManager:
    """Tests para el gestor de modelos heterogéneos."""
    
    def setup_method(self):
        self.manager = HeterogeneousModelManager()
        self.manager.memory_limit_gb = 10.0  # Límite reducido para tests
    
    def test_model_config_retrieval(self):
        """Test obtención de configuración por tipo de modelo."""
        config = self.manager.get_model_config(ModelType.SENTIMENT)
        
        assert config is not None
        assert config.model_type == ModelType.SENTIMENT
        assert config.ollama_name == "qwen2.5:7b-instruct-q5_k_m"
        assert config.temperature == 0.2
        assert config.specialized_for == [
            "sentiment_analysis",
            "news_interpretation",
            "social_media_analysis", 
            "market_mood_detection"
        ]
    
    def test_model_config_by_agent_type_string(self):
        """Test obtención de configuración por string de tipo de agente."""
        config = self.manager.get_model_config_by_agent_type("sentiment")
        
        assert config is not None
        assert config.model_type == ModelType.SENTIMENT
        
        config = self.manager.get_model_config_by_agent_type("technical")
        assert config.model_type == ModelType.TECHNICAL
        
        config = self.manager.get_model_config_by_agent_type("invalid")
        assert config is None
    
    @patch('config.heterogeneous_models.logger')
    def test_memory_management_loading(self, mock_logger):
        """Test gestión de memoria al cargar modelos."""
        # Configurar modelo pequeño para test
        small_config = HeterogeneousModelConfig(
            name="test-small",
            model_type=ModelType.SENTIMENT,
            ollama_name="test:small",
            quantization="Q4",
            memory_usage_gb=3.0,
            temperature=0.2,
            max_tokens=1024
        )
        
        self.manager.models[ModelType.SENTIMENT] = small_config
        
        # Test carga exitosa
        result = self.manager.load_model(ModelType.SENTIMENT)
        
        assert result is True
        assert self.manager.loaded_models.get("test:small") is True
        assert self.manager.current_memory_usage == 3.0
    
    def test_memory_status_reporting(self):
        """Test reporte de estado de memoria."""
        # Simular algunos modelos cargados
        self.manager.current_memory_usage = 7.5
        self.manager.loaded_models = {
            "qwen2.5:7b-instruct-q5_k_m": True,
            "deepseek-r1:7b-q4_k_m": True
        }
        
        status = self.manager.get_memory_status()
        
        assert status["total_limit_gb"] == 10.0
        assert status["current_usage_gb"] == 7.5
        assert status["available_gb"] == 2.5
        assert status["usage_percentage"] == 75.0
        assert len(status["loaded_models"]) >= 0
    
    def test_essential_models_preloading(self):
        """Test precarga de modelos esenciales."""
        with patch.object(self.manager, 'load_model') as mock_load:
            mock_load.return_value = True
            
            self.manager.preload_essential_models()
            
            # Verificar que se intentó cargar modelos de prioridad 1 no on_demand
            essential_calls = [
                call for call in mock_load.call_args_list
                if call[0][0] in [ModelType.SENTIMENT, ModelType.DECISION]
            ]
            
            assert len(essential_calls) > 0
    
    def test_ensure_model_loaded(self):
        """Test asegurar que un modelo esté cargado."""
        with patch.object(self.manager, 'load_model') as mock_load:
            mock_load.return_value = True
            
            # Modelo no cargado previamente
            self.manager.loaded_models = {}
            
            result = self.manager.ensure_model_loaded(ModelType.TECHNICAL)
            
            assert result is True
            mock_load.assert_called_once_with(ModelType.TECHNICAL)


class TestIntegrationHeterogeneousArchitecture:
    """Tests de integración para la arquitectura heterogénea completa."""
    
    def setup_method(self):
        self.validator = TradingSignalValidator()
        self.consensus = MultiModelConsensus()
        self.model_manager = HeterogeneousModelManager()
    
    @patch('config.heterogeneous_models.logger')
    def test_full_pipeline_simulation(self, mock_logger):
        """Test simulación del pipeline completo."""
        # Simular respuestas de diferentes modelos
        sentiment_response = '''
        {
            "sentiment_score": 0.75,
            "market_mood": "BULLISH",
            "confidence": 0.8,
            "reasoning": "Positive news sentiment with high social media engagement"
        }
        '''
        
        technical_response = '''
        {
            "action": "BUY", 
            "confidence": 0.85,
            "reasoning": "Strong technical indicators with momentum confirmation",
            "price_target": 50000
        }
        '''
        
        visual_response = '''
        {
            "overall_visual_assessment": "BULLISH",
            "suggested_action_based_on_visuals": "CONSIDER_BUY",
            "reasoning": "Clear upward channel with volume confirmation",
            "pattern_clarity_score": 0.9,
            "chart_timeframe_analyzed": "1h"
        }
        '''
        
        # Validar cada respuesta
        sentiment_validated = self.validator.validate_and_repair(sentiment_response, "trading")
        technical_validated = self.validator.validate_and_repair(technical_response, "trading") 
        visual_validated = self.validator.validate_and_repair(visual_response, "visual")
        
        assert sentiment_validated is not None
        assert technical_validated is not None
        assert visual_validated is not None
        
        # Simular consenso entre modelos técnicos
        trading_responses = [technical_validated]
        if sentiment_validated.get("action"):  # Si sentiment tiene acción
            trading_responses.append(sentiment_validated)
        
        if len(trading_responses) >= 2:
            consensus_result = self.consensus.get_consensus_decision(trading_responses)
            assert consensus_result is not None
            assert consensus_result["action"] in ["BUY", "SELL", "HOLD"]
    
    def test_memory_optimization_scenario(self):
        """Test escenario de optimización de memoria."""
        # Configurar límite de memoria restrictivo
        self.model_manager.memory_limit_gb = 8.0
        
        # Intentar cargar modelos que excedan el límite
        models_to_test = [ModelType.SENTIMENT, ModelType.TECHNICAL, ModelType.VISUAL]
        
        loaded_count = 0
        for model_type in models_to_test:
            config = self.model_manager.get_model_config(model_type)
            if config and self.model_manager.can_load_model(model_type):
                loaded_count += 1
        
        # Verificar que el sistema respeta los límites de memoria
        assert loaded_count <= 2  # No debería poder cargar más de 2 modelos grandes
    
    def test_error_handling_and_fallbacks(self):
        """Test manejo de errores y fallbacks."""
        # Test con respuesta completamente inválida
        invalid_response = "This is not JSON and will cause errors"
        
        result = self.validator.validate_and_repair(invalid_response, "trading")
        
        # Debe crear estructura de fallback válida
        assert result is not None
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.1
        assert "JSON malformado" in result["reasoning"]
        
        # Test consenso con respuestas inválidas
        invalid_responses = [None, {}, {"invalid": "data"}]
        
        consensus_result = self.consensus.get_consensus_decision(invalid_responses)
        
        assert consensus_result["action"] == "HOLD"
        assert consensus_result["consensus_type"] == "NO_CONSENSUS"


# Fixtures para tests
@pytest.fixture
def sample_trading_responses():
    """Fixture con respuestas de trading de ejemplo."""
    return [
        {
            "action": "BUY",
            "confidence": 0.85,
            "reasoning": "Strong bullish momentum with volume confirmation"
        },
        {
            "action": "BUY", 
            "confidence": 0.80,
            "reasoning": "Technical indicators align for upward movement"
        },
        {
            "action": "HOLD",
            "confidence": 0.70,
            "reasoning": "Mixed signals, awaiting confirmation"
        }
    ]

@pytest.fixture
def sample_malformed_json():
    """Fixture con JSON malformado de ejemplo."""
    return '''
    {
        action: "BUY",  // Missing quotes on key
        confidence: 0.85,
        reasoning: "Good setup but malformed JSON",
        price_target: 50000,  // Missing closing brace
    '''

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
