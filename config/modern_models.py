# config/modern_models.py
"""
Configuración optimizada de modelos LLM especializados para la arquitectura heterogénea.
Basada en la investigación para Mac M4 16GB con técnicas anti-alucinación.
"""

from typing import Dict, List
from dataclasses import dataclass, field
import dataclasses
import logging
import subprocess

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuración para cada modelo con parámetros optimizados"""
    name: str
    supports_tools: bool # True if the model supports OpenAI-style function calling/tools
    supports_vision: bool # True if the model can process image inputs
    context_length: int
    best_for: List[str]
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 60 # Default timeout in seconds for LLM calls
    # Nuevos campos para arquitectura heterogénea
    quantization: str = "Q4_K_M"
    memory_usage_gb: float = 4.0
    priority: int = 2
    load_on_demand: bool = True
    specialized_for: List[str] = field(default_factory=list)
    anti_hallucination: bool = True

class HeterogeneousModelManager:
    """
    Gestor de modelos heterogéneos con carga/descarga dinámica
    optimizada para Mac M4 16GB.
    """
    
    def __init__(self):
        self.available_ollama_models: List[str] = _AVAILABLE_OLLAMA_MODELS_LIST
        self.loaded_models: Dict[str, bool] = {}
        self.memory_limit_gb = 14.0  # Límite conservador para Mac M4 16GB
        self.current_memory_usage = 0.0
        self.model_health: Dict[str, Dict[str, int]] = {}
        logger.info(f"HeterogeneousModelManager initialized. Available models: {self.available_ollama_models}")
        
        if not self.available_ollama_models:
            logger.warning(
                "Ollama model list is empty. "
                "This might be due to Ollama not running or 'ollama list' failing. "
                "Model selection will rely on configured names and may fail if models are not truly available."
            )
    
    def can_load_model(self, model_name: str) -> bool:
        """Verifica si hay memoria suficiente para cargar el modelo."""
        for config in HETEROGENEOUS_MODELS_CONFIG.values():
            if config.name == model_name:
                return (self.current_memory_usage + config.memory_usage_gb) <= self.memory_limit_gb
        return False
    
    def get_memory_status(self) -> Dict[str, any]:
        """Obtiene estado actual de memoria."""
        loaded_models_list = [
            {
                "name": config.name,
                "memory_gb": config.memory_usage_gb,
                "priority": config.priority
            }
            for config in HETEROGENEOUS_MODELS_CONFIG.values()
            if self.loaded_models.get(config.name, False)
        ]
        
        return {
            "total_limit_gb": self.memory_limit_gb,
            "current_usage_gb": round(self.current_memory_usage, 2),
            "available_gb": round(self.memory_limit_gb - self.current_memory_usage, 2),
            "usage_percentage": round((self.current_memory_usage / self.memory_limit_gb) * 100, 1),
            "loaded_models": loaded_models_list,
            "total_loaded_models": len(loaded_models_list)
        }


class ModelManager:
    def __init__(self):
        self.available_ollama_models: List[str] = _AVAILABLE_OLLAMA_MODELS_LIST
        self.model_health: Dict[str, Dict[str, int]] = {}
        self.heterogeneous_manager = HeterogeneousModelManager()
        logger.info(f"ModelManager initialized. Available models: {self.available_ollama_models}")
        if not self.available_ollama_models:
             logger.warning(
                "Ollama model list is empty. "
                "This might be due to Ollama not running or 'ollama list' failing. "
                "Model selection will rely on configured names and may fail if models are not truly available."
            )

    def _is_model_explicitly_available(self, model_name_tag: str) -> bool:
        if not self.available_ollama_models: # If list is empty, cannot confirm availability
            logger.warning(f"Cannot confirm availability of '{model_name_tag}' as Ollama model list is empty. Assuming unavailable for safety.")
            return False
        return model_name_tag in self.available_ollama_models

# --- CONFIGURACIÓN DE MODELOS ESPECIALIZADOS HETEROGÉNEOS ---
# Basado en la investigación para reducir alucinaciones del 41% a <5%

_DEFAULT_LLAMA_MODEL = "llama3.2:1b" # Modelo de emergencia

# Fetch available Ollama models once
def _get_ollama_models_list() -> List[str]:
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            available = [line.split()[0] for line in lines[1:] if line.strip()]
            logger.info(f"Successfully fetched Ollama models: {available}")
            return available
        else:
            logger.error(f"Could not fetch Ollama models list. Error: {result.stderr.strip()}")
            return []
    except FileNotFoundError:
        logger.error("Ollama command-line tool not found. Please ensure Ollama is installed and in PATH.")
        return []
    except Exception as e:
        logger.error(f"Error fetching available Ollama models: {e}")
        return []

_AVAILABLE_OLLAMA_MODELS_LIST = _get_ollama_models_list()

if not _AVAILABLE_OLLAMA_MODELS_LIST:
    logger.warning("No Ollama models detected or Ollama CLI failed. Model selection will be limited and may not work as expected.")

# CONFIGURACIÓN DE MODELOS ESPECIALIZADOS HETEROGÉNEOS
HETEROGENEOUS_MODELS_CONFIG: Dict[str, ModelConfig] = {
    'sentiment': ModelConfig(
        name='qwen2.5:7b-instruct-q5_k_m',  # ✅ Disponible en Ollama
        supports_tools=True,  # Excelente para function calling
        supports_vision=False,
        context_length=128000,  # Ventana de contexto 128K para procesar mucho contenido
        best_for=['sentiment_analysis', 'news_interpretation', 'social_media_analysis', 'market_mood_detection'],
        temperature=0.2,  # Consistencia en análisis emocional
        max_tokens=1024,
        timeout=45,
        quantization="Q5_K_M",  # Mejor balance calidad/velocidad
        memory_usage_gb=5.5,
        priority=1,  # Alta prioridad - modelo principal
        load_on_demand=False,  # Mantener cargado
        specialized_for=['sentiment_analysis', 'news_interpretation', 'social_media_analysis', 'market_mood_detection'],
        anti_hallucination=True
    ),
    'technical': ModelConfig(
        name='qwen2.5:7b-instruct-q5_k_m',  # ✅ Using same reliable model as sentiment
        supports_tools=True,  # Capacidades de razonamiento superiores
        supports_vision=False,
        context_length=32000,
        best_for=['technical_analysis', 'pattern_recognition', 'indicator_interpretation', 'llm4fts_analysis', 'step_by_step_reasoning'],
        temperature=0.1,  # Máxima precisión técnica
        max_tokens=1500,
        timeout=60,
        quantization="Q5_K_M",  # Better quantization for reliability
        memory_usage_gb=5.5,  # Adjusted for Q5_K_M
        priority=1,  # Alta prioridad
        load_on_demand=True,  # Cargar cuando se necesite
        specialized_for=['technical_analysis', 'pattern_recognition', 'indicator_interpretation', 'llm4fts_analysis'],
        anti_hallucination=True
    ),
    'visual': ModelConfig(
        name='qwen2.5vl:7b-q4_K_M',  # ✅ Disponible en Ollama
        supports_tools=False,  # Vision models don't typically use tools
        supports_vision=True,  # Capacidad multimodal real
        context_length=8192,
        best_for=['chart_pattern_analysis', 'candlestick_interpretation', 'visual_trend_detection', 'support_resistance_identification'],
        temperature=0.1,  # Análisis visual preciso
        max_tokens=1200,
        timeout=90,  # Más tiempo para procesar imágenes
        quantization="Q4_K_M",
        memory_usage_gb=6.0,  # Ajustado según tamaño real
        priority=2,  # Media prioridad
        load_on_demand=True,  # Solo cuando se analicen gráficos
        specialized_for=['chart_analysis', 'pattern_recognition', 'visual_trend_detection'],
        anti_hallucination=True
    ),
    'qabba': ModelConfig(
        name='adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M',  # ✅ Disponible en Ollama
        supports_tools=True,  # 90% precisión en function calling
        supports_vision=False,
        context_length=8192,
        best_for=['bollinger_bands_analysis', 'quantitative_validation', 'function_calling', 'json_structured_output', 'mathematical_precision'],
        temperature=0.05,  # Ultra conservador para cálculos
        max_tokens=800,
        timeout=45,
        quantization="Q4_K_M",
        memory_usage_gb=4.9,  # Ajustado según tamaño real
        priority=2,  # Media prioridad
        load_on_demand=True,  # Cargar para análisis QABBA
        specialized_for=['mathematical_analysis', 'quantitative_reasoning', 'bollinger_bands'],
        anti_hallucination=True
    ),
    'decision': ModelConfig(
        name='qwen2.5:7b-instruct-q5_k_m',  # ✅ Mismo que sentiment pero configuración diferente
        supports_tools=True,  # Para síntesis y decisiones finales
        supports_vision=False,
        context_length=32000,
        best_for=['decision_synthesis', 'multi_agent_consensus', 'risk_assessment', 'final_trading_decisions'],
        temperature=0.05,  # Máxima consistencia para decisiones
        max_tokens=1000,
        timeout=60,
        quantization="Q5_K_M",
        memory_usage_gb=5.4,  # Ajustado según tamaño real
        priority=1,  # Alta prioridad - decisión final
        load_on_demand=False,  # Mantener cargado
        specialized_for=['decision_making', 'strategy_synthesis', 'risk_assessment'],
        anti_hallucination=True
    )
}

# MODELOS DE FALLBACK PARA LA ARQUITECTURA HETEROGÉNEA
HETEROGENEOUS_FALLBACK_CONFIG: Dict[str, List[ModelConfig]] = {
    'sentiment': [
        ModelConfig(
            name='qwen2.5:0.5b', 
            supports_tools=False, 
            supports_vision=False, 
            context_length=8192, 
            best_for=['fallback_sentiment'], 
            temperature=0.2, 
            max_tokens=800,
            memory_usage_gb=2.0,
            priority=3
        ),
        ModelConfig(
            name=_DEFAULT_LLAMA_MODEL, 
            supports_tools=False, 
            supports_vision=False, 
            context_length=8192, 
            best_for=['fallback_sentiment'], 
            temperature=0.2, 
            max_tokens=800,
            memory_usage_gb=1.5,
            priority=3
        )
    ],
    'technical': [
        ModelConfig(
            name='deepseek-r1:1.5b', 
            supports_tools=False, 
            supports_vision=False, 
            context_length=8192, 
            best_for=['fallback_technical'], 
            temperature=0.15, 
            max_tokens=1200,
            memory_usage_gb=2.5,
            priority=3
        ),
        ModelConfig(
            name='phi4:latest', 
            supports_tools=False, 
            supports_vision=False, 
            context_length=4096, 
            best_for=['fallback_technical'], 
            temperature=0.15, 
            max_tokens=1200,
            memory_usage_gb=3.0,
            priority=3
        )
    ],
    'visual': [
        ModelConfig(
            name='qwen2.5-vl:3b', 
            supports_tools=False, 
            supports_vision=True, 
            context_length=8192, 
            best_for=['fallback_visual'], 
            temperature=0.2, 
            max_tokens=1500, 
            timeout=180,
            memory_usage_gb=3.5,
            priority=3
        ),
        ModelConfig(
            name='llava:7b', 
            supports_tools=False, 
            supports_vision=True, 
            context_length=4096, 
            best_for=['fallback_visual'], 
            temperature=0.2, 
            max_tokens=1200, 
            timeout=180,
            memory_usage_gb=4.0,
            priority=3
        )
    ],
    'qabba': [
        ModelConfig(
            name='hermes-2-pro:3b', 
            supports_tools=True, 
            supports_vision=False, 
            context_length=4096, 
            best_for=['fallback_qabba'], 
            temperature=0.1, 
            max_tokens=800,
            memory_usage_gb=2.5,
            priority=3
        ),
        ModelConfig(
            name=_DEFAULT_LLAMA_MODEL, 
            supports_tools=False, 
            supports_vision=False, 
            context_length=8192, 
            best_for=['fallback_qabba'], 
            temperature=0.1, 
            max_tokens=800,
            memory_usage_gb=1.5,
            priority=3
        )
    ],
    'decision': [
        ModelConfig(
            name='qwen2.5:3b', 
            supports_tools=False, 
            supports_vision=False, 
            context_length=8192, 
            best_for=['fallback_decision'], 
            temperature=0.15, 
            max_tokens=1000,
            memory_usage_gb=3.0,
            priority=3
        ),
        ModelConfig(
            name='llama3.2:3b', 
            supports_tools=False, 
            supports_vision=False, 
            context_length=8192, 
            best_for=['fallback_decision'], 
            temperature=0.15, 
            max_tokens=1000,
            memory_usage_gb=3.5,
            priority=3
        )
    ]
}

# COMPATIBILIDAD CON SISTEMA ANTERIOR
MODERN_MODELS_CONFIG = HETEROGENEOUS_MODELS_CONFIG
FALLBACK_MODELS_CONFIG = HETEROGENEOUS_FALLBACK_CONFIG

class ModelManager:
    def __init__(self):
        self.available_ollama_models: List[str] = _AVAILABLE_OLLAMA_MODELS_LIST
        self.model_health: Dict[str, Dict[str, int]] = {} # Basic health tracking
        logger.info(f"ModelManager initialized. Available models: {self.available_ollama_models}")
        if not self.available_ollama_models:
             logger.warning(
                "Ollama model list is empty. "
                "This might be due to Ollama not running or 'ollama list' failing. "
                "Model selection will rely on configured names and may fail if models are not truly available."
            )

    def _is_model_explicitly_available(self, model_name_tag: str) -> bool:
        if not self.available_ollama_models: # If list is empty, cannot confirm availability
            logger.warning(f"Cannot confirm availability of '{model_name_tag}' as Ollama model list is empty. Assuming unavailable for safety.")
            return False
        return model_name_tag in self.available_ollama_models

    def get_model_config(self, agent_type: str) -> ModelConfig:
        default_config = ModelConfig(
            name=_DEFAULT_LLAMA_MODEL if self._is_model_explicitly_available(_DEFAULT_LLAMA_MODEL) else "ollama_model_not_found",
            supports_tools=False,
            supports_vision=False,
            context_length=4096,
            best_for=["general_fallback"],
            temperature=0.2,
            max_tokens=1024,
            timeout=60
        )
        if agent_type not in MODERN_MODELS_CONFIG:
            logger.error(f"Unknown agent type: '{agent_type}'. Using default model: {default_config.name}")
            return default_config

        primary_config = MODERN_MODELS_CONFIG[agent_type]
        
        if self._is_model_explicitly_available(primary_config.name):
            logger.info(f"Using primary model for {agent_type}: {primary_config.name}")
            return primary_config
        else:
            logger.warning(f"Primary model '{primary_config.name}' for {agent_type} is not in available Ollama models list: {self.available_ollama_models}. Attempting fallback.")

        for fallback_config in FALLBACK_MODELS_CONFIG.get(agent_type, []):
            if self._is_model_explicitly_available(fallback_config.name):
                logger.info(f"Using fallback model for {agent_type}: {fallback_config.name}")
                # Return a copy of the fallback_config to prevent modification of the original
                return dataclasses.replace(fallback_config)


        logger.error(f"No primary or fallback models available for agent type '{agent_type}' from the configured lists that are present in Ollama. Using absolute default: {default_config.name}")
        if default_config.name == "ollama_model_not_found":
            logger.critical(f"CRITICAL: Absolute default model '{_DEFAULT_LLAMA_MODEL}' is also not found in Ollama. LLM calls will likely fail.")
        return default_config

    def record_model_result(self, model_name: str, success: bool):
        """Records the outcome of a model request for basic health tracking."""
        if not isinstance(model_name, str):
            logger.warning(f"Invalid model_name type for recording result: {type(model_name)}")
            return
        if model_name not in self.model_health:
            self.model_health[model_name] = {'successes': 0, 'failures': 0, 'total_requests': 0}
        
        self.model_health[model_name]['total_requests'] += 1
        if success:
            self.model_health[model_name]['successes'] += 1
        else:
            self.model_health[model_name]['failures'] += 1
        # logger.debug(f"Model health for {model_name}: {self.model_health[model_name]}")


model_manager = ModelManager()

def print_model_availability_guide():
    print("\n🚀 MODEL AVAILABILITY GUIDE")
    print("=" * 70)
    
    print("\n📋 AVAILABLE OLLAMA MODELS (Detected by `ollama list`):")
    if model_manager.available_ollama_models:
        for model in sorted(model_manager.available_ollama_models):
            print(f"  ✅ {model}")
    else:
        print("  ⚠️ No Ollama models detected or 'ollama list' command failed.")
        print("      Ensure Ollama is running and the CLI is accessible.")

    print("\n⚙️ CONFIGURED MODELS AND THEIR STATUS:")
    for agent_type, config in MODERN_MODELS_CONFIG.items():
        status = "✅ Primary Available" if model_manager._is_model_explicitly_available(config.name) else "❌ Primary UNAVAILABLE"
        effective_config = model_manager.get_model_config(agent_type) # This will resolve to fallback if primary is unavailable
        
        print(f"\n🔹 AGENT TYPE: {agent_type.upper()}")
        print(f"   Configured Primary: {config.name} ({status})")
        if config.name != effective_config.name:
            fallback_status = "✅ Fallback Active" if model_manager._is_model_explicitly_available(effective_config.name) else "❌ Fallback UNAVAILABLE"
            print(f"   Effective Model:    {effective_config.name} ({fallback_status})")
        else:
            print(f"   Effective Model:    {effective_config.name}")

        print(f"     - Vision Support: {'👁️ Yes' if effective_config.supports_vision else '⛔ No'}")
        print(f"     - Tool Support:   {'🛠️ Yes (Instructor may be attempted)' if effective_config.supports_tools else '⛔ No (Raw query preferred)'}")
        print(f"     - Timeout:        {effective_config.timeout}s")
        print(f"     - Max Tokens:     {effective_config.max_tokens}")
        print(f"     - Temperature:    {effective_config.temperature}")


    print("\n" + "=" * 70)
    print("Ensure that the 'Effective Model' for each agent type is one of the 'AVAILABLE OLLAMA MODELS'.")
    print("If a primary model is unavailable, the system attempts to use a configured fallback.")
    print(f"If no configured models are available, it defaults to '{_DEFAULT_LLAMA_MODEL}' or 'ollama_model_not_found'.")
    print("Adjust 'config/modern_models.py' to match your installed Ollama models for optimal performance.")

if __name__ == "__main__":
    # This will now use the function from the initialized model_manager instance
    print_model_availability_guide()

# Exportar para compatibilidad con imports existentes
MODERN_MODELS = {
    'sentiment': 'qwen2.5:7b-instruct-q5_k_m',
    'technical': 'deepseek-r1:7b-qwen-distill-q4_K_M', 
    'visual': 'qwen2.5vl:7b-q4_K_M',
    'qabba': 'adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M',
    'decision': 'qwen2.5:7b-instruct-q5_k_m'
}
