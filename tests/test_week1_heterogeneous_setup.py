# tests/test_week1_heterogeneous_setup.py
"""
Tests de verificación para la Semana 1 del roadmap:
- Verificar que todos los modelos especializados están disponibles
- Validar la configuración de memoria
- Probar la gestión de memoria dinámica
- Confirmar parámetros optimizados por agente
"""

import pytest
import logging
from unittest.mock import patch, MagicMock

from config.modern_models import (
    ModelManager, 
    HeterogeneousModelManager,
    HETEROGENEOUS_MODELS_CONFIG,
    _AVAILABLE_OLLAMA_MODELS_LIST
)

logger = logging.getLogger(__name__)

class TestWeek1HeterogeneousSetup:
    """Tests para verificar el setup de la Semana 1."""
    
    def setup_method(self):
        self.model_manager = ModelManager()
        self.hetero_manager = HeterogeneousModelManager()
    
    def test_all_specialized_models_available(self):
        """Verifica que todos los modelos especializados estén disponibles en Ollama."""
        required_models = [
            'qwen2.5:7b-instruct-q5_k_m',          # Sentiment & Decision
            'deepseek-r1:7b-qwen-distill-q4_K_M',   # Technical
            'qwen2.5vl:7b-q4_K_M',                 # Visual
            'adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M'  # QABBA
        ]
        
        available_models = _AVAILABLE_OLLAMA_MODELS_LIST
        logger.info(f"Available Ollama models: {available_models}")
        
        missing_models = []
        for model in required_models:
            if model not in available_models:
                missing_models.append(model)
        
        if missing_models:
            pytest.fail(f"❌ Missing required models: {missing_models}")
        
        logger.info("✅ All specialized models are available!")
    
    def test_agent_configurations_specialized(self):
        """Verifica que cada agente tenga configuración especializada optimizada."""
        
        # Test Sentiment Agent
        sentiment_config = HETEROGENEOUS_MODELS_CONFIG['sentiment']
        assert sentiment_config.name == 'qwen2.5:7b-instruct-q5_k_m'
        assert sentiment_config.temperature == 0.2  # Consistencia emocional
        assert sentiment_config.supports_tools is True
        assert sentiment_config.context_length == 128000  # 128K context
        assert 'sentiment_analysis' in sentiment_config.specialized_for
        assert sentiment_config.priority == 1  # Alta prioridad
        assert sentiment_config.load_on_demand is False  # Mantener cargado
        
        # Test Technical Agent  
        technical_config = HETEROGENEOUS_MODELS_CONFIG['technical']
        assert technical_config.name == 'deepseek-r1:7b-qwen-distill-q4_K_M'
        assert technical_config.temperature == 0.1  # Máxima precisión
        assert technical_config.supports_tools is True
        assert technical_config.max_tokens == 1500  # Más tokens para análisis
        assert 'technical_analysis' in technical_config.specialized_for
        assert technical_config.load_on_demand is True  # Carga on-demand
        
        # Test Visual Agent
        visual_config = HETEROGENEOUS_MODELS_CONFIG['visual']
        assert visual_config.name == 'qwen2.5vl:7b-q4_K_M'
        assert visual_config.supports_vision is True  # Capacidad visual
        assert visual_config.supports_tools is False  # No tools para visual
        assert visual_config.timeout == 90  # Más tiempo para imágenes
        assert 'visual_trend_detection' in visual_config.specialized_for
        
        # Test QABBA Agent
        qabba_config = HETEROGENEOUS_MODELS_CONFIG['qabba']
        assert qabba_config.name == 'adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M'
        assert qabba_config.temperature == 0.05  # Ultra conservador
        assert qabba_config.supports_tools is True  # Function calling
        assert 'mathematical_analysis' in qabba_config.specialized_for
        
        # Test Decision Agent
        decision_config = HETEROGENEOUS_MODELS_CONFIG['decision']
        assert decision_config.name == 'qwen2.5:7b-instruct-q5_k_m'
        assert decision_config.temperature == 0.05  # Máxima consistencia
        assert decision_config.priority == 1  # Alta prioridad
        assert decision_config.load_on_demand is False  # Mantener cargado
        
        logger.info("✅ All agent configurations are properly specialized!")
    
    def test_memory_limits_mac_m4(self):
        """Verifica que los límites de memoria están configurados para Mac M4 16GB."""
        
        # Verificar límite total
        assert self.hetero_manager.memory_limit_gb == 14.0  # 14GB de 16GB total
        
        # Verificar que la suma de modelos esenciales no exceda el límite
        essential_models = [
            config for config in HETEROGENEOUS_MODELS_CONFIG.values()
            if config.priority == 1 and not config.load_on_demand
        ]
        
        total_essential_memory = sum(config.memory_usage_gb for config in essential_models)
        logger.info(f"Essential models memory usage: {total_essential_memory}GB")
        
        assert total_essential_memory <= self.hetero_manager.memory_limit_gb, \
            f"Essential models ({total_essential_memory}GB) exceed memory limit ({self.hetero_manager.memory_limit_gb}GB)"
        
        # Verificar que todos los modelos tienen memory_usage_gb configurado
        for agent_type, config in HETEROGENEOUS_MODELS_CONFIG.items():
            assert hasattr(config, 'memory_usage_gb'), f"{agent_type} missing memory_usage_gb"
            assert config.memory_usage_gb > 0, f"{agent_type} has invalid memory_usage_gb"
            assert config.memory_usage_gb < 8.0, f"{agent_type} memory too high for Mac M4"
        
        logger.info("✅ Memory limits properly configured for Mac M4 16GB!")
    
    def test_anti_hallucination_enabled(self):
        """Verifica que todas las configuraciones tienen anti-alucinación habilitada."""
        
        for agent_type, config in HETEROGENEOUS_MODELS_CONFIG.items():
            assert hasattr(config, 'anti_hallucination'), f"{agent_type} missing anti_hallucination field"
            assert config.anti_hallucination is True, f"{agent_type} anti-hallucination not enabled"
        
        logger.info("✅ Anti-hallucination enabled for all agents!")
    
    def test_quantization_levels_optimal(self):
        """Verifica que los niveles de cuantización son óptimos según la investigación."""
        
        # Sentiment y Decision: Q5_K_M (mejor balance calidad/velocidad)
        assert HETEROGENEOUS_MODELS_CONFIG['sentiment'].quantization == "Q5_K_M"
        assert HETEROGENEOUS_MODELS_CONFIG['decision'].quantization == "Q5_K_M"
        
        # Technical, Visual, QABBA: Q4_K_M (necesario para límite 16GB)
        assert HETEROGENEOUS_MODELS_CONFIG['technical'].quantization == "Q4_K_M"
        assert HETEROGENEOUS_MODELS_CONFIG['visual'].quantization == "Q4_K_M"
        assert HETEROGENEOUS_MODELS_CONFIG['qabba'].quantization == "Q4_K_M"
        
        logger.info("✅ Optimal quantization levels configured!")
    
    def test_priority_system_configured(self):
        """Verifica que el sistema de prioridades está correctamente configurado."""
        
        high_priority_agents = []
        medium_priority_agents = []
        
        for agent_type, config in HETEROGENEOUS_MODELS_CONFIG.items():
            if config.priority == 1:
                high_priority_agents.append(agent_type)
            elif config.priority == 2:
                medium_priority_agents.append(agent_type)
        
        # Sentiment y Decision deben ser alta prioridad
        assert 'sentiment' in high_priority_agents
        assert 'decision' in high_priority_agents
        
        # Technical, Visual, QABBA deben ser media prioridad o alta
        assert 'technical' in high_priority_agents or 'technical' in medium_priority_agents
        assert 'visual' in medium_priority_agents
        assert 'qabba' in medium_priority_agents
        
        logger.info(f"✅ Priority system: High={high_priority_agents}, Medium={medium_priority_agents}")
    
    def test_memory_status_reporting(self):
        """Verifica que el reporte de estado de memoria funciona."""
        
        # Simular algunos modelos cargados
        self.hetero_manager.current_memory_usage = 7.5
        self.hetero_manager.loaded_models = {
            'qwen2.5:7b-instruct-q5_k_m': True,
            'deepseek-r1:7b-qwen-distill-q4_K_M': True
        }
        
        status = self.hetero_manager.get_memory_status()
        
        # Verificar estructura del reporte
        required_fields = [
            'total_limit_gb', 'current_usage_gb', 'available_gb', 
            'usage_percentage', 'loaded_models', 'total_loaded_models'
        ]
        
        for field in required_fields:
            assert field in status, f"Missing field in memory status: {field}"
        
        # Verificar cálculos
        assert status['total_limit_gb'] == 14.0
        assert status['current_usage_gb'] == 7.5
        assert status['available_gb'] == 6.5
        assert status['usage_percentage'] == 53.6  # 7.5/14.0 * 100
        
        logger.info("✅ Memory status reporting working correctly!")
    
    def test_can_load_model_logic(self):
        """Verifica que la lógica de verificación de carga funciona."""
        
        # Reset memory
        self.hetero_manager.current_memory_usage = 0.0
        
        # Debería poder cargar sentiment (5.5GB)
        can_load = self.hetero_manager.can_load_model('qwen2.5:7b-instruct-q5_k_m')
        assert can_load is True
        
        # Simular que ya está usando 12GB
        self.hetero_manager.current_memory_usage = 12.0
        
        # No debería poder cargar visual (6.0GB) porque excedería el límite
        can_load = self.hetero_manager.can_load_model('qwen2.5vl:7b-q4_K_M')
        assert can_load is False
        
        logger.info("✅ Model loading logic working correctly!")


class TestModelManagerIntegration:
    """Tests de integración con ModelManager."""
    
    def setup_method(self):
        self.model_manager = ModelManager()
    
    def test_get_model_config_by_agent_type(self):
        """Verifica que get_model_config devuelve las configuraciones especializadas."""
        
        # Test cada tipo de agente
        for agent_type in ['sentiment', 'technical', 'visual', 'qabba', 'decision']:
            config = self.model_manager.get_model_config(agent_type)
            
            assert config is not None, f"No config returned for {agent_type}"
            assert hasattr(config, 'specialized_for'), f"{agent_type} missing specialized_for"
            assert hasattr(config, 'anti_hallucination'), f"{agent_type} missing anti_hallucination"
            assert config.anti_hallucination is True, f"{agent_type} anti-hallucination not enabled"
        
        logger.info("✅ ModelManager integration working!")
    
    def test_fallback_models_available(self):
        """Verifica que hay modelos de fallback configurados."""
        
        # Verificar que cada agente tiene al menos un fallback disponible
        from config.modern_models import HETEROGENEOUS_FALLBACK_CONFIG
        
        for agent_type in ['sentiment', 'technical', 'visual', 'qabba', 'decision']:
            fallbacks = HETEROGENEOUS_FALLBACK_CONFIG.get(agent_type, [])
            assert len(fallbacks) > 0, f"No fallback models for {agent_type}"
        
        logger.info("✅ Fallback models properly configured!")


class TestSystemReadiness:
    """Tests para verificar que el sistema está listo para Semana 2."""
    
    def test_week1_objectives_complete(self):
        """Verifica que todos los objetivos de Semana 1 están completos."""
        
        checklist = {
            "Modelos instalados": len(_AVAILABLE_OLLAMA_MODELS_LIST) >= 4,
            "Configuración actualizada": len(HETEROGENEOUS_MODELS_CONFIG) == 5,
            "Gestión de memoria implementada": HeterogeneousModelManager is not None,
            "Anti-alucinación habilitada": all(
                config.anti_hallucination for config in HETEROGENEOUS_MODELS_CONFIG.values()
            )
        }
        
        failed_items = [item for item, passed in checklist.items() if not passed]
        
        if failed_items:
            pytest.fail(f"❌ Week 1 objectives not complete: {failed_items}")
        
        logger.info("✅ All Week 1 objectives complete! Ready for Week 2.")
    
    def test_memory_optimization_for_mac_m4(self):
        """Verifica que la optimización para Mac M4 está funcionando."""
        
        hetero_manager = HeterogeneousModelManager()
        
        # Verificar configuración específica para Mac M4
        assert hetero_manager.memory_limit_gb == 14.0  # Conservador para 16GB
        
        # Verificar que los modelos esenciales pueden coexistir
        essential_configs = [
            config for config in HETEROGENEOUS_MODELS_CONFIG.values()
            if config.priority == 1 and not config.load_on_demand
        ]
        
        total_essential = sum(config.memory_usage_gb for config in essential_configs)
        assert total_essential <= hetero_manager.memory_limit_gb
        
        # Verificar que hay espacio para al menos un modelo on-demand
        remaining_memory = hetero_manager.memory_limit_gb - total_essential
        assert remaining_memory >= 4.0  # Al menos 4GB para modelos on-demand
        
        logger.info(f"✅ Mac M4 optimization: {total_essential}GB essential, {remaining_memory}GB available")


if __name__ == "__main__":
    # Configurar logging para los tests
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Ejecutar tests
    pytest.main([__file__, "-v", "--tb=short"])
