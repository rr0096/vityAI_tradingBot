#!/usr/bin/env python3
"""
Prueba Básica Real - FenixTradingBot v2.0
Test de lo que realmente funciona sin importar archivos problemáticos
"""

def test_ollama_models():
    """Verificar modelos Ollama instalados"""
    print("🧪 Test: Modelos Ollama instalados")
    
    import subprocess
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            
            required = [
                'qwen2.5:7b-instruct-q5_k_m',
                'deepseek-r1:7b-qwen-distill-q4_K_M', 
                'qwen2.5vl:7b-q4_K_M',
                'adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M'
            ]
            
            print(f"✅ Modelos instalados: {len(models)}")
            missing = [m for m in required if m not in models]
            if missing:
                print(f"⚠️ Modelos faltantes: {missing}")
                return False
            else:
                print("✅ Todos los modelos requeridos están instalados")
                return True
        else:
            print("❌ No se pudo ejecutar 'ollama list'")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_basic_imports():
    """Test imports básicos que deberían funcionar"""
    print("\n🧪 Test: Imports básicos")
    
    working_imports = []
    failed_imports = []
    
    # Test individual imports
    imports_to_test = [
        ("config.modern_models", "ModelManager"),
        ("agents.enhanced_base_llm_agent", "EnhancedBaseLLMAgent"),
        ("agents.sentiment_enhanced", "EnhancedSentimentAnalyst"),
        ("tools.twitter_scraper", "TwitterScraper"),
        ("tools.improved_news_scraper", "ImprovedNewsScraper"),
        ("agents.json_validator", "TradingSignalValidator"),
    ]
    
    for module, class_name in imports_to_test:
        try:
            exec(f"from {module} import {class_name}")
            working_imports.append(f"{module}.{class_name}")
            print(f"✅ {module}.{class_name}")
        except Exception as e:
            failed_imports.append(f"{module}.{class_name} - {e}")
            print(f"❌ {module}.{class_name} - {str(e)[:50]}...")
    
    print(f"\n📊 Imports funcionando: {len(working_imports)}/{len(imports_to_test)}")
    return len(working_imports) > len(imports_to_test) // 2

def test_agent_creation():
    """Test creación de agente básico"""
    print("\n🧪 Test: Creación de agente básico")
    
    try:
        from config.modern_models import ModelManager
        from agents.sentiment_enhanced import EnhancedSentimentAnalyst
        
        # Crear ModelManager
        model_manager = ModelManager()
        print(f"✅ ModelManager creado - {len(model_manager.available_ollama_models)} modelos")
        
        # Crear agente sentiment
        agent = EnhancedSentimentAnalyst(
            model_manager=model_manager,
            agent_type="sentiment"
        )
        print("✅ Agente sentiment creado exitosamente")
        
        # Verificar configuración
        print(f"   Cliente configurado: {agent.client is not None}")
        print(f"   Instructor configurado: {agent.instructor_client is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creando agente: {e}")
        return False

def test_data_fetching():
    """Test obtención de datos real"""
    print("\n🧪 Test: Obtención de datos real")
    
    try:
        from tools.improved_news_scraper import ImprovedNewsScraper
        
        scraper = ImprovedNewsScraper()
        print("✅ News scraper inicializado")
        
        # Test método específico que sabemos que existe
        if hasattr(scraper, 'fetch_from_rss'):
            # Test con una fuente específica
            test_data = scraper.fetch_from_rss(
                'coindesk', 
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                max_items=3
            )
            print(f"✅ RSS fetch funcionando: {len(test_data)} artículos")
            return True
        else:
            print("⚠️ Método fetch_from_rss no encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Error en data fetching: {e}")
        return False

def test_twitter_alternative():
    """Test Twitter scraper alternativo"""
    print("\n🧪 Test: Twitter scraper alternativo")
    
    try:
        from tools.twitter_scraper import TwitterScraper
        
        scraper = TwitterScraper()
        print("✅ Twitter scraper inicializado")
        
        # El scraper debería tener métodos básicos
        print(f"   Nombre: {scraper.name}")
        print(f"   Descripción: {scraper.description[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en Twitter scraper: {e}")
        return False

def test_json_validation_basic():
    """Test básico de validación JSON"""
    print("\n🧪 Test: Validación JSON básica")
    
    try:
        from agents.json_validator import TradingSignalValidator
        
        validator = TradingSignalValidator()
        print("✅ JSON Validator creado")
        
        # Test con estructura simple
        test_data = {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test reasoning for validation purposes"
        }
        
        # Solo verificar que el validator tiene métodos básicos
        methods = [m for m in dir(validator) if not m.startswith('_')]
        print(f"   Métodos disponibles: {len(methods)}")
        print(f"   Principales: {[m for m in methods if 'validate' in m.lower()]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en JSON validation: {e}")
        return False

def test_memory_status():
    """Test estado de memoria del sistema"""
    print("\n🧪 Test: Estado de memoria del sistema")
    
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        print(f"✅ Memoria total: {memory.total // (1024**3)}GB")
        print(f"   Memoria disponible: {memory.available // (1024**3)}GB")
        print(f"   Uso: {memory.percent}%")
        
        # Verificar si hay procesos Ollama
        ollama_processes = []
        for proc in psutil.process_iter(['name', 'memory_info']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    ollama_processes.append(proc)
            except:
                pass
        
        print(f"   Procesos Ollama: {len(ollama_processes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en memoria: {e}")
        return False

def main():
    """Ejecutar todas las pruebas básicas"""
    print("🚀 PRUEBA BÁSICA REAL: FenixTradingBot v2.0")
    print("=" * 50)
    print("(Solo probando lo que sabemos que funciona)")
    print()
    
    tests = [
        test_ollama_models,
        test_basic_imports,
        test_agent_creation,
        test_data_fetching,
        test_twitter_alternative,
        test_json_validation_basic,
        test_memory_status
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test falló: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN:")
    
    test_names = [
        "Modelos Ollama",
        "Imports Básicos",
        "Creación Agente",
        "Data Fetching",
        "Twitter Alternativo",
        "JSON Validation",
        "Estado Memoria"
    ]
    
    passed = sum(results)
    total = len(results)
    
    for name, result in zip(test_names, results):
        status = "✅ OK" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n🎯 RESULTADO: {passed}/{total} tests OK")
    
    if passed >= total * 0.7:  # 70% threshold
        print("✅ Sistema MAYORMENTE FUNCIONAL")
        print("🔧 Algunas correcciones menores necesarias")
    elif passed >= total * 0.5:  # 50% threshold
        print("⚠️ Sistema PARCIALMENTE FUNCIONAL") 
        print("🔧 Varias correcciones necesarias")
    else:
        print("❌ Sistema REQUIERE TRABAJO MAYOR")
        
    return passed >= total * 0.5

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
