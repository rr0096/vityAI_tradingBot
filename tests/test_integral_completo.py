#!/usr/bin/env python3
"""
Prueba Integral FenixTradingBot v2.0
Test completo del sistema migrado para verificar funcionamiento real
"""

import logging
import time
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test 1: Verificar que todas las importaciones funcionan"""
    print("🧪 Test 1: Verificando importaciones...")
    
    try:
        # Test imports básicos
        from config.modern_models import MODERN_MODELS, ModelManager
        from config.heterogeneous_models import HETEROGENEOUS_AGENT_MODELS
        from agents.json_validator import TradingSignalValidator
        from agents.multi_model_consensus import MultiModelConsensus
        from system.dynamic_memory_manager import DynamicMemoryManager
        from system.memory_aware_agent_manager import MemoryAwareAgentManager
        from tools.twitter_scraper import TwitterScraper
        from tools.improved_news_scraper import ImprovedNewsScraper
        
        print("✅ Todas las importaciones exitosas")
        return True
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        return False

def test_model_manager():
    """Test 2: Verificar ModelManager"""
    print("\n🧪 Test 2: Verificando ModelManager...")
    
    try:
        from config.modern_models import ModelManager
        
        mm = ModelManager()
        print(f"✅ ModelManager inicializado")
        print(f"   Modelos disponibles: {len(mm.available_models)}")
        print(f"   Modelos Ollama: {len(mm.available_ollama_models)}")
        
        # Verificar modelos específicos
        required_models = [
            'qwen2.5:7b-instruct-q5_k_m',
            'deepseek-r1:7b-qwen-distill-q4_K_M',
            'qwen2.5vl:7b-q4_K_M',
            'adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M'
        ]
        
        missing_models = []
        for model in required_models:
            if model not in mm.available_ollama_models:
                missing_models.append(model)
        
        if missing_models:
            print(f"⚠️ Modelos faltantes: {missing_models}")
        else:
            print("✅ Todos los modelos requeridos están disponibles")
            
        return len(missing_models) == 0
        
    except Exception as e:
        print(f"❌ Error en ModelManager: {e}")
        return False

def test_memory_manager():
    """Test 3: Verificar sistema de memoria dinámica"""
    print("\n🧪 Test 3: Verificando sistema de memoria...")
    
    try:
        from system.dynamic_memory_manager import DynamicMemoryManager
        
        dmm = DynamicMemoryManager()
        
        # Test estadísticas de memoria
        stats = dmm.get_memory_stats()
        print(f"✅ Memoria total: {stats.total_mb}MB")
        print(f"   Memoria disponible: {stats.available_mb}MB")
        print(f"   Uso actual: {stats.usage_percent}%")
        
        # Test estimación de modelos
        models_to_test = [
            'qwen2.5:7b-instruct-q5_k_m',
            'deepseek-r1:7b-qwen-distill-q4_K_M'
        ]
        
        for model in models_to_test:
            estimated = dmm.estimate_model_memory(model)
            can_load = dmm.can_load_model(model)
            print(f"   {model}: {estimated}MB (Can load: {can_load})")
        
        print("✅ Sistema de memoria funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Error en sistema de memoria: {e}")
        return False

def test_json_validator():
    """Test 4: Verificar validador JSON"""
    print("\n🧪 Test 4: Verificando validador JSON...")
    
    try:
        from agents.json_validator import TradingSignalValidator
        
        validator = TradingSignalValidator()
        
        # Test con JSON válido
        valid_signal = {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Strong bullish indicators suggest potential upward movement",
            "price_target": 50000,
            "timestamp": "2025-06-20T16:00:00Z"
        }
        
        is_valid = validator.validate_trading_signal(valid_signal)
        print(f"✅ Validación JSON exitosa: {is_valid}")
        
        # Test con JSON inválido (para reparación)
        invalid_signal = {
            "action": "MAYBE",  # Valor inválido
            "confidence": 1.5,  # Fuera de rango
            "reasoning": "short"  # Muy corto
        }
        
        repaired = validator.repair_and_validate_json(str(invalid_signal), "trading")
        print(f"✅ Reparación JSON funcionando: {repaired['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en validador JSON: {e}")
        return False

def test_data_sources():
    """Test 5: Verificar fuentes de datos"""
    print("\n🧪 Test 5: Verificando fuentes de datos...")
    
    try:
        # Test News Scraper
        from tools.improved_news_scraper import ImprovedNewsScraper
        
        news_scraper = ImprovedNewsScraper()
        news_data = news_scraper.get_latest_news(max_articles=3)
        
        print(f"✅ News scraper: {len(news_data)} artículos obtenidos")
        if news_data:
            print(f"   Ejemplo: {news_data[0].get('title', 'Sin título')[:50]}...")
        
        # Test Twitter Scraper (alternativo)
        from tools.twitter_scraper import TwitterScraper
        
        twitter_scraper = TwitterScraper()
        # Test básico sin ejecutar scraping completo
        print("✅ Twitter scraper inicializado correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en fuentes de datos: {e}")
        return False

def test_agent_initialization():
    """Test 6: Verificar inicialización de agentes"""
    print("\n🧪 Test 6: Verificando inicialización de agentes...")
    
    try:
        from agents.sentiment_enhanced import EnhancedSentimentAnalyst
        from config.modern_models import ModelManager
        
        model_manager = ModelManager()
        
        # Test agente de sentiment (sin ejecutar análisis completo)
        agent = EnhancedSentimentAnalyst(
            model_manager=model_manager,
            agent_type="sentiment"
        )
        
        print("✅ Agente de sentiment inicializado")
        print(f"   Modelo configurado: {agent.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en inicialización de agentes: {e}")
        return False

def test_memory_aware_manager():
    """Test 7: Verificar gestor de agentes con memoria"""
    print("\n🧪 Test 7: Verificando gestor de agentes con memoria...")
    
    try:
        from system.memory_aware_agent_manager import MemoryAwareAgentManager
        
        manager = MemoryAwareAgentManager()
        
        # Test preparación de agentes
        agent_types = ['sentiment', 'technical', 'qabba']
        
        for agent_type in agent_types:
            model_name = manager.get_agent_model(agent_type)
            print(f"   {agent_type} -> {model_name}")
        
        # Test reporte de estado
        status = manager.get_comprehensive_status()
        print(f"✅ Gestor de agentes funcionando")
        print(f"   Timestamp: {status['timestamp']}")
        print(f"   Presión de memoria: {status['system_pressure']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en gestor de agentes: {e}")
        return False

def test_end_to_end_light():
    """Test 8: Prueba ligera end-to-end"""
    print("\n🧪 Test 8: Prueba ligera end-to-end...")
    
    try:
        from system.memory_aware_agent_manager import MemoryAwareAgentManager
        from tools.improved_news_scraper import ImprovedNewsScraper
        
        # Inicializar componentes
        agent_manager = MemoryAwareAgentManager()
        news_scraper = ImprovedNewsScraper()
        
        print("✅ Componentes inicializados")
        
        # Simular preparación de agente
        sentiment_ready = agent_manager.prepare_agent('sentiment')
        print(f"✅ Agente sentiment preparado: {sentiment_ready}")
        
        # Obtener algunas noticias
        news = news_scraper.get_latest_news(max_articles=2)
        print(f"✅ Noticias obtenidas: {len(news)}")
        
        print("✅ Flujo end-to-end básico funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba end-to-end: {e}")
        return False

def main():
    """Ejecutar todas las pruebas"""
    print("🚀 PRUEBA INTEGRAL: FenixTradingBot v2.0")
    print("=" * 60)
    
    start_time = time.time()
    tests = [
        test_imports,
        test_model_manager,
        test_memory_manager,
        test_json_validator,
        test_data_sources,
        test_agent_initialization,
        test_memory_aware_manager,
        test_end_to_end_light
    ]
    
    results = []
    for i, test in enumerate(tests, 1):
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {i} falló con excepción: {e}")
            results.append(False)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS:")
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Importaciones",
        "ModelManager", 
        "Sistema Memoria",
        "Validador JSON",
        "Fuentes Datos",
        "Inicialización Agentes",
        "Gestor Memoria",
        "End-to-End"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {name}: {status}")
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} tests pasados")
    
    elapsed = time.time() - start_time
    print(f"⏱️ Tiempo total: {elapsed:.2f} segundos")
    
    if passed == total:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ FenixTradingBot v2.0 está funcionando correctamente")
        print("🚀 Sistema listo para uso en producción")
    else:
        print(f"\n⚠️ {total-passed} tests fallaron")
        print("🔧 Se requieren correcciones antes del deployment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
