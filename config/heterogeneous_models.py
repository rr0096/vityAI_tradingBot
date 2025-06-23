# config/heterogeneous_models.py
"""
Configuración de modelos especializados basada en la investigación.
Implementa la arquitectura heterogénea recomendada para cada agente.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    VISUAL = "visual"
    QABBA = "qabba"
    DECISION = "decision"

@dataclass
class HeterogeneousModelConfig:
    """Configuración específica para cada modelo en la arquitectura heterogénea."""
    name: str
    model_type: ModelType
    ollama_name: str
    quantization: str
    memory_usage_gb: float
    temperature: float
    max_tokens: int
    supports_vision: bool = False
    supports_tools: bool = False
    context_length: int = 4096
    timeout: int = 60
    priority: int = 1  # 1=alta, 2=media, 3=baja para gestión de memoria
    load_on_demand: bool = False
    specialized_for: List[str] = field(default_factory=list)
    constitutional_prompting: bool = True
    anti_hallucination: bool = True
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica para Ollama."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_length": self.context_length,
            "timeout": self.timeout
        }

class HeterogeneousModelManager:
    """
    Gestor de modelos heterogéneos con carga/descarga dinámica
    optimizada para Mac M4 16GB.
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.loaded_models: Dict[str, bool] = {}
        self.memory_limit_gb = 14.0  # Límite conservador para Mac M4 16GB
        self.current_memory_usage = 0.0
        
    def _initialize_models(self) -> Dict[ModelType, HeterogeneousModelConfig]:
        """Inicializa configuraciones de modelos basadas en la investigación."""
        
        return {
            ModelType.SENTIMENT: HeterogeneousModelConfig(
                name="qwen2.5-7b-sentiment",
                model_type=ModelType.SENTIMENT,
                ollama_name="qwen2.5:7b-instruct-q5_k_m",
                quantization="Q5_K_M",
                memory_usage_gb=5.5,
                temperature=0.2,  # Consistencia en análisis emocional
                max_tokens=1024,
                supports_vision=False,
                supports_tools=True,
                context_length=128000,  # Ventana de contexto 128K
                timeout=45,
                priority=1,  # Alta prioridad - modelo principal
                load_on_demand=False,  # Mantener cargado
                specialized_for=[
                    "sentiment_analysis",
                    "news_interpretation", 
                    "social_media_analysis",
                    "market_mood_detection"
                ],
                constitutional_prompting=True,
                anti_hallucination=True
            ),
            
            ModelType.TECHNICAL: HeterogeneousModelConfig(
                name="qwen2.5-7b-technical",
                model_type=ModelType.TECHNICAL,
                ollama_name="qwen2.5:7b-instruct-q4_k_m",
                quantization="Q4_K_M",
                memory_usage_gb=4.5,
                temperature=0.1,  # Máxima precisión técnica
                max_tokens=1500,
                supports_vision=False,
                supports_tools=True,
                context_length=32000,
                timeout=60,
                priority=1,  # Alta prioridad
                load_on_demand=True,  # Cargar cuando se necesite
                specialized_for=[
                    "technical_analysis",
                    "pattern_recognition",
                    "indicator_interpretation", 
                    "llm4fts_analysis",
                    "step_by_step_reasoning"
                ],
                constitutional_prompting=True,
                anti_hallucination=True
            ),
            
            ModelType.VISUAL: HeterogeneousModelConfig(
                name="qwen2.5-vl-7b-visual",
                model_type=ModelType.VISUAL,
                ollama_name="qwen2.5-vl:7b-q4_k_m",
                quantization="Q4_K_M",
                memory_usage_gb=5.0,
                temperature=0.1,  # Análisis visual preciso
                max_tokens=1200,
                supports_vision=True,
                supports_tools=False,
                context_length=8192,
                timeout=90,  # Más tiempo para procesar imágenes
                priority=2,  # Media prioridad
                load_on_demand=True,  # Solo cuando se analicen gráficos
                specialized_for=[
                    "chart_pattern_analysis",
                    "candlestick_interpretation",
                    "visual_trend_detection",
                    "support_resistance_identification"
                ],
                constitutional_prompting=True,
                anti_hallucination=True
            ),
            
            ModelType.QABBA: HeterogeneousModelConfig(
                name="hermes-2-pro-8b-qabba",
                model_type=ModelType.QABBA,
                ollama_name="hermes-2-pro-llama-3:8b-q4_k_m",
                quantization="Q4_K_M",
                memory_usage_gb=5.2,
                temperature=0.05,  # Ultra conservador para cálculos
                max_tokens=800,
                supports_vision=False,
                supports_tools=True,
                context_length=8192,
                timeout=45,
                priority=2,  # Media prioridad
                load_on_demand=True,  # Cargar para análisis QABBA
                specialized_for=[
                    "bollinger_bands_analysis",
                    "quantitative_validation",
                    "function_calling",
                    "json_structured_output",
                    "mathematical_precision"
                ],
                constitutional_prompting=True,
                anti_hallucination=True
            ),
            
            ModelType.DECISION: HeterogeneousModelConfig(
                name="qwen2.5-7b-decision",
                model_type=ModelType.DECISION,
                ollama_name="qwen2.5:7b-instruct-q5_k_m",  # Mismo que sentiment
                quantization="Q5_K_M",
                memory_usage_gb=5.5,
                temperature=0.05,  # Máxima consistencia para decisiones
                max_tokens=1000,
                supports_vision=False,
                supports_tools=True,
                context_length=32000,
                timeout=60,
                priority=1,  # Alta prioridad - decisión final
                load_on_demand=False,  # Mantener cargado
                specialized_for=[
                    "decision_synthesis",
                    "multi_agent_consensus",
                    "risk_assessment",
                    "final_trading_decisions"
                ],
                constitutional_prompting=True,
                anti_hallucination=True
            )
        }
    
    def get_model_config(self, model_type: ModelType) -> Optional[HeterogeneousModelConfig]:
        """Obtiene configuración de modelo por tipo."""
        return self.models.get(model_type)
    
    def get_model_config_by_agent_type(self, agent_type: str) -> Optional[HeterogeneousModelConfig]:
        """Obtiene configuración de modelo por tipo de agente string."""
        type_mapping = {
            "sentiment": ModelType.SENTIMENT,
            "technical": ModelType.TECHNICAL,
            "visual": ModelType.VISUAL,
            "qabba": ModelType.QABBA,
            "decision": ModelType.DECISION
        }
        
        model_type = type_mapping.get(agent_type.lower())
        if model_type:
            return self.get_model_config(model_type)
        return None
    
    def can_load_model(self, model_type: ModelType) -> bool:
        """Verifica si hay memoria suficiente para cargar el modelo."""
        config = self.get_model_config(model_type)
        if not config:
            return False
            
        return (self.current_memory_usage + config.memory_usage_gb) <= self.memory_limit_gb
    
    def load_model(self, model_type: ModelType) -> bool:
        """
        Carga un modelo en memoria, descargando otros si es necesario.
        
        Returns:
            True si el modelo se cargó exitosamente
        """
        config = self.get_model_config(model_type)
        if not config:
            logger.error(f"No config found for model type {model_type}")
            return False
            
        # Si ya está cargado, no hacer nada
        if self.loaded_models.get(config.ollama_name, False):
            logger.info(f"Model {config.ollama_name} already loaded")
            return True
            
        # Verificar memoria disponible
        if not self.can_load_model(model_type):
            logger.info(f"Insufficient memory for {config.ollama_name}, freeing space...")
            if not self._free_memory_for_model(config):
                logger.error(f"Could not free enough memory for {config.ollama_name}")
                return False
        
        # Cargar modelo (simulado - en implementación real usaría Ollama API)
        try:
            logger.info(f"Loading model {config.ollama_name} ({config.memory_usage_gb}GB)")
            # Aquí iría la llamada real a Ollama para cargar el modelo
            # ollama.load_model(config.ollama_name)
            
            self.loaded_models[config.ollama_name] = True
            self.current_memory_usage += config.memory_usage_gb
            logger.info(f"Model {config.ollama_name} loaded successfully. Memory usage: {self.current_memory_usage:.1f}GB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {config.ollama_name}: {e}")
            return False
    
    def unload_model(self, model_type: ModelType) -> bool:
        """Descarga un modelo de memoria."""
        config = self.get_model_config(model_type)
        if not config:
            return False
            
        if not self.loaded_models.get(config.ollama_name, False):
            logger.info(f"Model {config.ollama_name} not loaded")
            return True
            
        try:
            logger.info(f"Unloading model {config.ollama_name}")
            # Aquí iría la llamada real a Ollama para descargar el modelo
            # ollama.unload_model(config.ollama_name)
            
            self.loaded_models[config.ollama_name] = False
            self.current_memory_usage -= config.memory_usage_gb
            logger.info(f"Model {config.ollama_name} unloaded. Memory usage: {self.current_memory_usage:.1f}GB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {config.ollama_name}: {e}")
            return False
    
    def _free_memory_for_model(self, target_config: HeterogeneousModelConfig) -> bool:
        """
        Libera memoria descargando modelos de menor prioridad.
        
        Args:
            target_config: Configuración del modelo que se quiere cargar
            
        Returns:
            True si se liberó suficiente memoria
        """
        needed_memory = target_config.memory_usage_gb
        available_memory = self.memory_limit_gb - self.current_memory_usage
        
        if available_memory >= needed_memory:
            return True
            
        memory_to_free = needed_memory - available_memory
        
        # Ordenar modelos cargados por prioridad (menor prioridad = candidatos a descarga)
        loaded_models_info = []
        for model_type, config in self.models.items():
            if self.loaded_models.get(config.ollama_name, False):
                loaded_models_info.append((model_type, config))
        
        # Ordenar por prioridad (3=baja, 2=media, 1=alta) y load_on_demand
        loaded_models_info.sort(key=lambda x: (x[1].priority, not x[1].load_on_demand), reverse=True)
        
        freed_memory = 0.0
        for model_type, config in loaded_models_info:
            if freed_memory >= memory_to_free:
                break
                
            # No descargar modelos de prioridad 1 que no son on_demand
            if config.priority == 1 and not config.load_on_demand:
                continue
                
            logger.info(f"Freeing memory by unloading {config.ollama_name}")
            if self.unload_model(model_type):
                freed_memory += config.memory_usage_gb
        
        return freed_memory >= memory_to_free
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Obtiene estado actual de memoria."""
        loaded_models_list = [
            {
                "name": config.ollama_name,
                "type": config.model_type.value,
                "memory_gb": config.memory_usage_gb,
                "priority": config.priority
            }
            for config in self.models.values()
            if self.loaded_models.get(config.ollama_name, False)
        ]
        
        return {
            "total_limit_gb": self.memory_limit_gb,
            "current_usage_gb": round(self.current_memory_usage, 2),
            "available_gb": round(self.memory_limit_gb - self.current_memory_usage, 2),
            "usage_percentage": round((self.current_memory_usage / self.memory_limit_gb) * 100, 1),
            "loaded_models": loaded_models_list,
            "total_loaded_models": len(loaded_models_list)
        }
    
    def preload_essential_models(self):
        """Precarga modelos esenciales (prioridad 1 y no on_demand)."""
        essential_types = [
            model_type for model_type, config in self.models.items()
            if config.priority == 1 and not config.load_on_demand
        ]
        
        logger.info(f"Preloading essential models: {[t.value for t in essential_types]}")
        
        for model_type in essential_types:
            if not self.load_model(model_type):
                logger.warning(f"Failed to preload essential model {model_type.value}")
    
    def ensure_model_loaded(self, model_type: ModelType) -> bool:
        """
        Asegura que un modelo esté cargado, cargándolo si es necesario.
        
        Returns:
            True si el modelo está disponible para uso
        """
        config = self.get_model_config(model_type)
        if not config:
            return False
            
        if self.loaded_models.get(config.ollama_name, False):
            return True
            
        return self.load_model(model_type)

# Instancia global del gestor
heterogeneous_model_manager = HeterogeneousModelManager()

# Configuración de límites de memoria específicos para Mac M4
MAC_M4_CONFIG = {
    "total_ram_gb": 16,
    "system_reserved_gb": 2,
    "available_for_llms_gb": 14,
    "max_concurrent_models": 3,
    "emergency_memory_threshold_gb": 1,
    "apple_mlx_optimization": True,
    "unified_memory_architecture": True
}

def get_recommended_setup_commands() -> List[str]:
    """Obtiene comandos recomendados para configurar Ollama en Mac M4."""
    return [
        "# Configurar límites de memoria GPU para Mac M4",
        "sudo sysctl iogpu.wired_limit_mb=14336  # 14GB para LLMs",
        "",
        "# Variables de entorno Ollama optimizadas",
        "export OLLAMA_GPU_PERCENT=90",
        "export OLLAMA_MAX_LOADED_MODELS=2",
        "export OLLAMA_FLASH_ATTENTION=1",
        "",
        "# Instalar modelos especializados",
        "ollama pull qwen2.5:7b-instruct-q5_k_m",
        "ollama pull deepseek-r1:7b-q4_k_m",
        "ollama pull qwen2.5-vl:7b-q4_k_m",
        "ollama pull hermes-2-pro-llama-3:8b-q4_k_m",
        "",
        "# Verificar instalación",
        "ollama list"
    ]

# Exportar para compatibilidad con imports existentes
HETEROGENEOUS_AGENT_MODELS = {
    'sentiment': 'qwen2.5:7b-instruct-q5_k_m',
    'technical': 'qwen2.5:7b-instruct-q4_k_m',
    'visual': 'qwen2.5vl:7b-q4_K_M',
    'qabba': 'adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M',
    'decision': 'qwen2.5:7b-instruct-q5_k_m'
}
