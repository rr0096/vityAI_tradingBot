# test_integration_fixed.py
"""
CORREGIDO: Test de integración completo que verifica todos los componentes
con manejo robusto de errores y fallbacks
"""
import asyncio # Importación no utilizada directamente en este script, podría ser para módulos importados.
import logging
from pathlib import Path
import sys
import time
import os
from typing import Dict, Any, List, Optional # Optional no se usa directamente, pero es buena práctica mantenerlo si los módulos importados lo usan.
import json
from datetime import datetime # Importación no utilizada directamente, podría ser para módulos importados o futura expansión.

# Configurar logging mejorado
# Se crea el directorio 'logs' antes de configurar el FileHandler para evitar errores si no existe.
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # Usar sys.stdout explícitamente.
        logging.FileHandler(Path("logs/integration_test.log"))
    ]
)
logger = logging.getLogger("IntegrationTest")

# Crear directorios necesarios de forma centralizada
REQUIRED_DIRECTORIES = ["data", "performance_reports"] # 'logs' ya se crea arriba.
for directory in REQUIRED_DIRECTORIES:
    Path(directory).mkdir(exist_ok=True)

# Añadir paths necesarios
# Añade el directorio padre del script actual al sys.path para permitir importaciones relativas.
sys.path.append(str(Path(__file__).resolve().parent))

class IntegrationTestResult:
    """
    Clase para almacenar el resultado de un test específico.
    Mantiene información sobre el éxito, errores, advertencias, tiempo de ejecución y datos adicionales.
    """
    def __init__(self, name: str):
        self.name: str = name
        self.success: bool = False
        self.error_message: str = ""
        self.warnings: List[str] = []
        self.execution_time: float = 0.0
        self.data: Dict[str, Any] = {}

class IntegrationTestSuite:
    """
    Suite completa de tests de integración.
    Esta clase gestiona la ejecución de múltiples tests, recopila sus resultados
    y genera un informe final.
    """
    
    def __init__(self):
        self.results: Dict[str, IntegrationTestResult] = {}
        self.total_start_time: float = time.monotonic() # Usar time.monotonic para mediciones de tiempo más precisas.
        
    def run_all_tests(self) -> bool:
        """
        Ejecuta todos los tests definidos en la suite.
        Retorna True si todos los tests pasan, False en caso contrario.
        """
        
        logger.info("🚀 INICIANDO SUITE DE TESTS DE INTEGRACIÓN FÉNIX BOT\n")
        
        # Lista de tests a ejecutar. Cada tupla contiene el nombre del test y la función a llamar.
        tests_to_run = [
            ("environment", self.test_environment),
            ("imports", self.test_imports),
            ("technical_tools", self.test_technical_tools),
            ("models", self.test_unified_models),
            ("chart_generation", self.test_chart_generation),
            ("api_clients", self.test_api_clients),
            ("agents_individual", self.test_agents_individual),
            ("agents_integration", self.test_agents_integration),
            ("risk_manager", self.test_risk_manager),
            ("websocket_simulation", self.test_websocket_simulation),
            ("performance_logging", self.test_performance_logging)
        ]
        
        # Ejecutar cada test
        for test_name, test_func in tests_to_run:
            result = IntegrationTestResult(test_name)
            start_time = time.monotonic()
            
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🔍 EJECUTANDO: {test_name.upper()}")
                logger.info(f"{'='*50}")
                
                test_success = test_func(result) # Llama a la función de test correspondiente.
                result.success = test_success
                
                if test_success:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    # Si el error_message no fue establecido por el test, poner uno genérico.
                    if not result.error_message:
                        result.error_message = "El test falló sin un mensaje de error específico."
                    logger.error(f"❌ {test_name} FAILED: {result.error_message}")
                    
            except Exception as e:
                result.success = False
                result.error_message = f"Excepción no controlada: {str(e)}"
                logger.error(f"❌ {test_name} CRASHED: {e}", exc_info=True) # exc_info=True para loggear el traceback completo.
            
            finally:
                result.execution_time = time.monotonic() - start_time
                self.results[test_name] = result # Almacena el resultado del test.
                
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning(f"⚠️  Advertencia en {test_name}: {warning}")
        
        # Generar reporte final y retornar el estado general de la suite.
        return self.generate_final_report()
    
    def test_environment(self, result: IntegrationTestResult) -> bool:
        """
        Test del entorno: versión de Python, dependencias críticas y opcionales,
        variables de entorno y directorios necesarios.
        """
        
        # Verificar Python version
        if sys.version_info < (3, 8):
            result.error_message = f"Python 3.8+ es requerido. Versión actual: {sys.version}"
            return False
        
        result.data["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        logger.info(f"  ✅ Versión de Python: {result.data['python_version']}")
        
        # Verificar dependencias críticas
        required_packages = [
            ("numpy", "NumPy"), ("pandas", "Pandas"), ("requests", "Requests"),
            ("websockets", "WebSockets"), ("pydantic", "Pydantic"),
            ("crewai", "CrewAI"), ("ollama", "Ollama")
        ]
        
        missing_packages = []
        for package_name, display_name in required_packages:
            try:
                __import__(package_name)
                logger.info(f"  ✅ Dependencia crítica: {display_name} encontrada.")
            except ImportError:
                missing_packages.append(display_name)
                logger.error(f"  ❌ Dependencia crítica: {display_name} NO encontrada.")
        
        if missing_packages:
            result.error_message = f"Paquetes críticos faltantes: {', '.join(missing_packages)}"
            return False
        
        # Verificar dependencias opcionales
        optional_packages = [
            ("talib", "TA-Lib"), ("matplotlib", "Matplotlib"), ("binance", "Binance"),
        ]
        
        for package_name, display_name in optional_packages:
            try:
                __import__(package_name)
                logger.info(f"  ✅ Dependencia opcional: {display_name} encontrada.")
            except ImportError:
                warning_msg = f"Dependencia opcional {display_name} no está disponible."
                result.warnings.append(warning_msg)
                logger.warning(f"  ⚠️ {warning_msg}")
        
        # Verificar variables de entorno
        # Es mejor usar una lista de tuplas si se quiere dar mensajes más específicos o realizar validaciones.
        env_vars_to_check = ["BINANCE_API_KEY", "BINANCE_API_SECRET", "CRYPTOPANIC_TOKEN"]
        for var_name in env_vars_to_check:
            value = os.getenv(var_name)
            if not value or value in ["", "your_key_here", "..."]: # Chequeos básicos
                warning_msg = f"Variable de entorno '{var_name}' no está configurada o tiene un valor placeholder."
                result.warnings.append(warning_msg)
                logger.warning(f"  ⚠️ {warning_msg}")
            else:
                logger.info(f"  ✅ Variable de entorno '{var_name}' configurada.")
        
        # Verificar directorios (ya creados al inicio, aquí solo se confirma)
        for directory_name in REQUIRED_DIRECTORIES + ["logs"]: # Incluye 'logs' para la verificación.
            if Path(directory_name).is_dir():
                logger.info(f"  ✅ Directorio '{directory_name}' existe.")
            else:
                # Esto no debería ocurrir si la creación inicial fue exitosa.
                result.error_message = f"Directorio '{directory_name}' no existe y no pudo ser creado."
                logger.error(f"  ❌ {result.error_message}")
                return False
        
        return True
    
    def test_imports(self, result: IntegrationTestResult) -> bool:
        """
        Test de todas las importaciones críticas y de agentes.
        Verifica que los módulos y sus componentes esperados puedan ser importados.
        """
        
        # Lista de importaciones a probar: (ruta_del_modulo, [lista_de_componentes_a_importar])
        import_tests_critical = [
            ("models.outputs", ["SentimentOutput", "TechnicalAnalysisOutput", "FinalDecisionOutput"]),
            ("tools.technical_tools", ["add_kline", "get_current_indicators", "close_buf"]), # Asumiendo que close_buf es un typo y debería ser clear_all_buffers o similar.
            ("tools.chart_generator_fixed", ["ChartGenerator", "generate_chart_for_visual_agent"]),
            ("utils.error_handling", ["RobustAPIClient", "APIClientFactory"]),
        ]
        
        failed_critical_imports = []
        for module_path, items_to_import in import_tests_critical:
            try:
                module = __import__(module_path, fromlist=items_to_import) # CORRECCIÓN: fromlist debe ser usado correctamente.
                for item_name in items_to_import:
                    if hasattr(module, item_name):
                        logger.info(f"  ✅ Importación crítica: {module_path}.{item_name}")
                    else:
                        failed_critical_imports.append(f"{module_path}.{item_name}")
                        logger.error(f"  ❌ Componente '{item_name}' no encontrado en módulo '{module_path}'.")
                        
            except ImportError as e:
                failed_critical_imports.append(f"Módulo {module_path}: {str(e)}")
                logger.error(f"  ❌ Error al importar módulo crítico '{module_path}': {e}")
        
        if failed_critical_imports:
            result.error_message = f"Fallaron importaciones críticas: {', '.join(failed_critical_imports)}"
            result.data["failed_critical_imports"] = failed_critical_imports
            return False
        
        # Test imports de agentes (opcionales o con fallbacks)
        agent_imports_definitions = [
            ("agents.sentiment", "EnhancedSentimentAnalyst"),
            ("agents.technical_fixed", "EnhancedTechnicalAnalyst"),
            ("agents.enhanced_visual_analyst", "EnhancedVisualAnalystAgent"),
            ("agents.risk_v2", "AdvancedRiskManager"),
        ]
        
        available_agents = []
        unavailable_agents_warnings = []
        for module_path, class_name in agent_imports_definitions:
            try:
                module = __import__(module_path, fromlist=[class_name])
                if hasattr(module, class_name):
                    available_agents.append(class_name)
                    logger.info(f"  ✅ Importación de agente: {module_path}.{class_name}")
                else:
                    warning_msg = f"Agente '{class_name}' no encontrado en módulo '{module_path}'."
                    unavailable_agents_warnings.append(warning_msg)
                    logger.warning(f"  ⚠️ {warning_msg}")
            except ImportError as e:
                warning_msg = f"Módulo de agente '{module_path}' no encontrado o con error de importación: {e}"
                unavailable_agents_warnings.append(warning_msg)
                logger.warning(f"  ⚠️ {warning_msg}")
        
        result.data["available_agents"] = available_agents
        if unavailable_agents_warnings:
            result.warnings.extend(unavailable_agents_warnings)
            result.data["unavailable_agents_warnings"] = unavailable_agents_warnings
            
        return True # Este test pasa si las importaciones críticas están OK, los agentes son opcionales.

    def test_technical_tools(self, result: IntegrationTestResult) -> bool:
        """Test de las herramientas técnicas: validación, adición de datos y cálculo de indicadores."""
        try:
            from tools.technical_tools import (
                add_kline, get_current_indicators, get_buffer_status,
                clear_all_buffers, validate_kline_data
            )
            logger.info("  ✅ Imports de 'tools.technical_tools' exitosos.")
            
            clear_all_buffers()
            logger.info("  ✅ Buffers técnicos limpiados.")

            # Test de validación de datos OHLCV
            if not validate_kline_data(open_price=100.0, high_price=101.0, low_price=99.0, close_price=100.5, volume=1000.0): # Asumiendo que validate_kline_data toma 5 args
                result.error_message = "Validación de datos OHLCV básica falló para datos válidos."
                return False
            logger.info("  ✅ Validación de datos OHLCV (datos válidos) pasó.")

            if validate_kline_data(open_price=100.0, high_price=99.0, low_price=101.0, close_price=100.5, volume=1000.0):  # High < Low
                result.warnings.append("Validación de datos OHLCV no rechazó High < Low.")
                logger.warning("  ⚠️ Validación de datos OHLCV no rechazó High < Low.")
            else:
                logger.info("  ✅ Validación de datos OHLCV (High < Low) rechazada correctamente.")

            # Añadir datos de prueba (suficientes para la mayoría de los indicadores)
            test_klines_count = 60
            base_price = 100.0
            for i in range(test_klines_count):
                # Simular datos de velas con algo de variabilidad
                open_p = base_price + (i * 0.05) + ((-1)**i * 0.2)
                close_p = open_p + ((-1)**(i+1) * 0.3) + (i * 0.02)
                high_p = max(open_p, close_p) + abs(hash(str(i)) % 50) / 100.0 # Asegurar high >= open/close
                low_p = min(open_p, close_p) - abs(hash(str(i+1)) % 50) / 100.0 # Asegurar low <= open/close
                volume_v = 1000.0 + (i * 10)
                
                # add_kline espera (close, high, low, volume) según el código original. Ajustar si es necesario.
                # Si add_kline espera OHLCV, se debe pasar open_p también.
                # El código original pasaba (close, high, low, volume)
                if not add_kline(close_p, high_p, low_p, volume_v):
                    result.error_message = f"add_kline falló en la iteración {i} con datos: C={close_p}, H={high_p}, L={low_p}, V={volume_v}"
                    return False
            logger.info(f"  ✅ Añadidas {test_klines_count} velas de prueba a los buffers.")

            buffer_status = get_buffer_status()
            result.data["buffer_status_after_add"] = buffer_status
            if not buffer_status or buffer_status.get("close_buf", 0) != test_klines_count:
                result.error_message = f"Tamaño del buffer incorrecto: esperado {test_klines_count}, obtenido {buffer_status.get('close_buf', 0)}."
                return False
            logger.info(f"  ✅ Estado de buffers verificado: {buffer_status}")

            indicators = get_current_indicators()
            if not indicators:
                result.error_message = "get_current_indicators() retornó vacío o None."
                return False
            
            # Formatear indicadores para el log y reporte
            result.data["calculated_indicators"] = {
                k: f"{v:.4f}" if isinstance(v, float) else v
                for k, v in indicators.items()
            }
            logger.info(f"  ✅ Calculados {len(indicators)} indicadores: {result.data['calculated_indicators']}")

            required_indicators_keys = ["rsi", "macd_line", "atr", "last_price"] # Ajustar según lo esperado
            missing_keys = [key for key in required_indicators_keys if key not in indicators]
            if missing_keys:
                result.error_message = f"Indicadores clave faltantes en el resultado: {', '.join(missing_keys)}"
                return False

            # Validaciones básicas de rangos de indicadores
            if not (0 <= indicators.get("rsi", -1) <= 100):
                result.warnings.append(f"Valor de RSI fuera de rango (0-100): {indicators.get('rsi')}")
            if indicators.get("atr", -1) <= 0:
                result.warnings.append(f"Valor de ATR no es positivo: {indicators.get('atr')}")
            
            return True
            
        except ImportError:
            result.error_message = "No se pudieron importar las herramientas técnicas (tools.technical_tools)."
            logger.error(f"  ❌ {result.error_message}")
            return False
        except Exception as e:
            result.error_message = f"Error inesperado en test_technical_tools: {str(e)}"
            logger.error(f"  ❌ {result.error_message}", exc_info=True)
            return False

    def test_unified_models(self, result: IntegrationTestResult) -> bool:
        """Test de los modelos Pydantic unificados: creación, mockeo y conversión."""
        try:
            from models.outputs import (
                SentimentOutput, TechnicalAnalysisOutput, FinalDecisionOutput, # RiskAssessment no se usa aquí directamente
                convert_to_decision_inputs, create_mock_outputs_for_testing
            )
            logger.info("  ✅ Imports de 'models.outputs' exitosos.")

            mock_outputs_dict = create_mock_outputs_for_testing()
            if not mock_outputs_dict:
                result.error_message = "create_mock_outputs_for_testing() retornó vacío."
                return False
            result.data["mock_outputs_created"] = list(mock_outputs_dict.keys())
            
            required_mock_types = ["sentiment", "technical", "visual", "qabba"] # Asumiendo qabba es un tipo de output
            for output_type_key in required_mock_types:
                if output_type_key not in mock_outputs_dict:
                    result.error_message = f"Mock output para '{output_type_key}' no fue creado."
                    return False
                # Verificar que es un modelo Pydantic (tiene model_dump)
                if not hasattr(mock_outputs_dict[output_type_key], "model_dump"):
                    result.error_message = f"Mock output '{output_type_key}' no parece ser un modelo Pydantic válido (falta model_dump)."
                    return False
            logger.info(f"  ✅ Mock outputs creados para: {', '.join(mock_outputs_dict.keys())}")

            # Test de conversión para DecisionAgent
            # Asegurarse que los mocks existen antes de pasarlos
            sentiment_mock = mock_outputs_dict.get("sentiment")
            technical_mock = mock_outputs_dict.get("technical")
            visual_mock = mock_outputs_dict.get("visual")
            qabba_mock = mock_outputs_dict.get("qabba")

            if not all([sentiment_mock, technical_mock, visual_mock, qabba_mock]):
                missing_mocks_for_conversion = [
                    k for k,v in {"sentiment":sentiment_mock, "technical":technical_mock, "visual":visual_mock, "qabba":qabba_mock}.items() if not v
                ]
                result.error_message = f"Faltan mocks para la conversión a inputs de decisión: {missing_mocks_for_conversion}"
                return False

            # El código original no define los tipos de input, se asume que la conversión es correcta si no hay error.
            # Se podría añadir una validación más profunda si se conocen los tipos de input esperados.
            try:
                converted_inputs = convert_to_decision_inputs(
                    sentiment_mock, technical_mock, visual_mock, qabba_mock
                )
                # Ejemplo de validación (si se conocen los campos esperados):
                # sentiment_input, technical_input, visual_input, qabba_input = converted_inputs
                # if sentiment_input.overall_sentiment != sentiment_mock.overall_sentiment:
                #    result.error_message = "Conversión de sentiment_input falló la validación de datos."
                #    return False
                logger.info("  ✅ Conversión a inputs de DecisionAgent (convert_to_decision_inputs) ejecutada.")
                result.data["converted_inputs_types"] = [type(inp).__name__ for inp in converted_inputs]

            except Exception as e_conv:
                result.error_message = f"Error durante convert_to_decision_inputs: {str(e_conv)}"
                return False


            # Test de validación de campos (ejemplo con SentimentOutput)
            try:
                SentimentOutput(
                    overall_sentiment="ESTO_ES_INVALIDO_SEGURO", # Debería fallar la validación de Pydantic si 'overall_sentiment' es un Enum o Literal
                    positive_texts_count=5,
                    negative_texts_count=5,
                    neutral_texts_count=10,
                    reasoning="Test de validación",
                    confidence_score=0.8 # Asegurar que el score sea válido (e.g. 0 a 1)
                )
                # Si llega aquí, la validación no funcionó como se esperaba para este caso.
                result.warnings.append("Validación de SentimentOutput no detectó un valor inválido para 'overall_sentiment' (o el valor de prueba era válido).")
                logger.warning("  ⚠️  La prueba de validación de SentimentOutput con valor inválido no falló como se esperaba.")
            except ValueError: # O pydantic.ValidationError
                logger.info("  ✅ Validación de modelos Pydantic (ej. SentimentOutput con datos inválidos) funciona correctamente (lanzó error).")
            except Exception as e_val: # Capturar otras excepciones de validación
                 logger.info(f"  ✅ Validación de modelos Pydantic (ej. SentimentOutput) lanzó una excepción esperada: {type(e_val).__name__}")


            return True
            
        except ImportError:
            result.error_message = "No se pudieron importar los modelos de 'models.outputs'."
            logger.error(f"  ❌ {result.error_message}")
            return False
        except Exception as e:
            result.error_message = f"Error inesperado en test_unified_models: {str(e)}"
            logger.error(f"  ❌ {result.error_message}", exc_info=True)
            return False

    def test_chart_generation(self, result: IntegrationTestResult) -> bool:
        """Test del generador de gráficos: creación de imagen y función de conveniencia."""
        try:
            from tools.chart_generator_fixed import ChartGenerator, generate_chart_for_visual_agent
            import random # Para generar datos de prueba
            logger.info("  ✅ Import de 'tools.chart_generator_fixed' exitoso.")

            n_points = 50
            base_price = 100.0
            # Generar datos de prueba más realistas para OHLCV
            closes = []
            highs = []
            lows = []
            volumes = []
            current_price = base_price
            for _ in range(n_points):
                current_price += random.uniform(-0.5, 0.5) # fluctuación
                open_p = current_price + random.uniform(-0.1, 0.1)
                close_p = current_price + random.uniform(-0.1, 0.1)
                high_p = max(open_p, close_p) + random.uniform(0, 0.2)
                low_p = min(open_p, close_p) - random.uniform(0, 0.2)
                vol = random.uniform(1000, 5000)
                
                closes.append(close_p)
                highs.append(high_p)
                lows.append(low_p)
                volumes.append(vol)
            
            tech_metrics_sample = {
                'last_price': closes[-1], 'rsi': random.uniform(30,70),
                'atr': random.uniform(0.5, 2.0), 'curr_vol': volumes[-1]
            }

            generator = ChartGenerator() # Asumiendo que no necesita argumentos en __init__
            chart_b64_output = generator.generate_chart_image(
                symbol="TEST/USDT", timeframe="1m",
                close_prices=closes, high_prices=highs, low_prices=lows, volumes=volumes,
                tech_metrics=tech_metrics_sample, lookback_periods=n_points
            )

            if not chart_b64_output or len(chart_b64_output) < 500: # Un gráfico base64 debería ser más grande
                result.error_message = f"Gráfico generado (ChartGenerator.generate_chart_image) es demasiado pequeño o vacío (longitud: {len(chart_b64_output)})."
                return False
            result.data["chart_generator_output_size_bytes"] = len(chart_b64_output) # Es longitud de string, no bytes directamente.
            logger.info(f"  ✅ Gráfico generado por ChartGenerator: {len(chart_b64_output)} caracteres base64.")

            # Test de la función de conveniencia
            # Asumiendo que los buffers son listas de precios/volúmenes
            chart_b64_convenience_output = generate_chart_for_visual_agent(
                symbol="TEST2/USDT",
                close_buf=closes, high_buf=highs, low_buf=lows, vol_buf=volumes,
                tech_metrics=tech_metrics_sample
            )
            if not chart_b64_convenience_output or len(chart_b64_convenience_output) < 500:
                result.warnings.append(f"Gráfico de función de conveniencia (generate_chart_for_visual_agent) es pequeño o vacío (longitud: {len(chart_b64_convenience_output)}).")
            else:
                logger.info(f"  ✅ Gráfico de función de conveniencia generado: {len(chart_b64_convenience_output)} caracteres base64.")
            result.data["chart_convenience_output_size_bytes"] = len(chart_b64_convenience_output)


            # Test con datos insuficientes (debería manejarlo sin crashear)
            short_closes = [100.0, 101.0]
            short_highs = [100.5, 101.5]
            short_lows = [99.5, 100.5]
            short_volumes = [100, 120]
            try:
                error_chart_output = generator.generate_chart_image(
                    symbol="ERROR/TEST", timeframe="1m",
                    close_prices=short_closes, high_prices=short_highs, low_prices=short_lows, volumes=short_volumes,
                    tech_metrics={}, lookback_periods=n_points # lookback > len(data)
                )
                if not error_chart_output: # Podría retornar None o una imagen de error
                    result.warnings.append("ChartGenerator.generate_chart_image con datos insuficientes retornó None/vacío en lugar de una imagen de error o manejo graceful.")
                else:
                     logger.info("  ✅ ChartGenerator.generate_chart_image con datos insuficientes manejado (no crasheó).")
            except Exception as e_short_data:
                result.warnings.append(f"ChartGenerator.generate_chart_image crasheó con datos insuficientes: {e_short_data}")
                logger.warning(f"  ⚠️ ChartGenerator.generate_chart_image crasheó con datos insuficientes: {e_short_data}")

            return True
            
        except ImportError:
            result.error_message = "No se pudo importar 'tools.chart_generator_fixed'."
            logger.error(f"  ❌ {result.error_message}")
            return False
        except Exception as e:
            result.error_message = f"Error inesperado en test_chart_generation: {str(e)}"
            logger.error(f"  ❌ {result.error_message}", exc_info=True)
            return False

    def test_api_clients(self, result: IntegrationTestResult) -> bool:
        """Test de los clientes API robustos: factory, estadísticas y funcionalidades básicas."""
        try:
            from utils.error_handling import APIClientFactory, RobustAPIClient
            logger.info("  ✅ Import de 'utils.error_handling' (API clients) exitoso.")

            # Test del factory
            # Asumiendo que los clientes no hacen llamadas reales en __init__ o get_client
            try:
                news_client = APIClientFactory.get_news_client("test_api_token_news")
                fg_client = APIClientFactory.get_fear_greed_client() # Asume que no necesita token
                reddit_client = APIClientFactory.get_reddit_client() # Asume que no necesita config específica aquí
            except Exception as e_factory:
                result.error_message = f"Error al crear clientes API vía Factory: {e_factory}"
                return False

            if not all([news_client, fg_client, reddit_client]):
                result.error_message = "APIClientFactory no creó todos los clientes esperados (alguno es None)."
                return False
            logger.info("  ✅ APIClientFactory creó instancias de news, fear_greed, y reddit clients.")
            result.data["factory_clients_created"] = [type(c).__name__ for c in [news_client, fg_client, reddit_client]]


            # Test de estadísticas iniciales
            initial_stats = APIClientFactory.get_all_stats()
            result.data["initial_api_stats"] = initial_stats
            expected_client_names_in_stats = ["news", "fear_greed", "reddit"] # Nombres como se registran en el factory
            
            for client_name_stat in expected_client_names_in_stats:
                if client_name_stat not in initial_stats:
                    result.warnings.append(f"Cliente '{client_name_stat}' no encontrado en las estadísticas iniciales de APIClientFactory.")
                else:
                    client_stat_data = initial_stats[client_name_stat]
                    if client_stat_data.get("total_calls", 0) != 0:
                        result.warnings.append(f"Cliente '{client_name_stat}' ya tiene llamadas ({client_stat_data.get('total_calls')}) registradas en estadísticas iniciales.")
            logger.info(f"  ✅ Estadísticas iniciales de APIClientFactory obtenidas para {len(initial_stats)} clientes.")

            # Test básico de RobustAPIClient (circuit breaker y cache)
            # Crear una instancia directa para probar funcionalidades aisladas si es posible
            # Si RobustAPIClient es una clase base abstracta, este test necesitaría una subclase mock.
            # Asumiendo que se puede instanciar:
            try:
                test_robust_client = RobustAPIClient(base_url="http://test.com", client_name="test_robust", timeout=1, max_retries=1, backoff_factor=0.1)
            except TypeError: # Si es abstracta
                 result.warnings.append("RobustAPIClient podría ser abstracta, no se pudo instanciar directamente para test de circuit breaker/cache. Omitiendo esta parte.")
                 logger.warning("  ⚠️ RobustAPIClient no se pudo instanciar directamente.")
                 return True # Omitir el resto de este test si no se puede instanciar.

            # Test Circuit Breaker (estado inicial)
            if not hasattr(test_robust_client, 'circuit_breaker') or not hasattr(test_robust_client.circuit_breaker, 'can_execute'):
                 result.warnings.append("Instancia de RobustAPIClient no tiene 'circuit_breaker' o 'can_execute' como se esperaba.")
            elif not test_robust_client.circuit_breaker.can_execute():
                result.error_message = "Circuit breaker de RobustAPIClient debería estar cerrado (can_execute=True) inicialmente."
                return False
            else:
                logger.info("  ✅ Circuit breaker de RobustAPIClient (estado inicial) verificado.")

            # Test Cache (set/get)
            if not hasattr(test_robust_client, 'cache') or not hasattr(test_robust_client.cache, 'set') or not hasattr(test_robust_client.cache, 'get'):
                result.warnings.append("Instancia de RobustAPIClient no tiene 'cache' o métodos set/get como se esperaba.")
            else:
                test_key = "my_test_cache_key"
                test_value = {"data": "important_cached_data"}
                test_robust_client.cache.set(test_key, test_value, ttl_seconds=60)
                cached_value = test_robust_client.cache.get(test_key)
                if not cached_value or cached_value.get("data") != "important_cached_data":
                    result.error_message = "Cache de RobustAPIClient (set/get) no funciona correctamente."
                    return False
                logger.info("  ✅ Cache de RobustAPIClient (set/get) verificado.")
            
            return True
            
        except ImportError:
            result.error_message = "No se pudo importar 'utils.error_handling'."
            logger.error(f"  ❌ {result.error_message}")
            return False
        except Exception as e:
            result.error_message = f"Error inesperado en test_api_clients: {str(e)}"
            logger.error(f"  ❌ {result.error_message}", exc_info=True)
            return False

    def test_agents_individual(self, result: IntegrationTestResult) -> bool:
        """
        Test individual de cada agente: verifica que se puedan instanciar.
        No se ejecutan sus métodos principales para evitar llamadas a LLMs o APIs externas.
        """
        agent_instantiation_results = {}
        all_agents_instantiated = True

        # Lista de agentes a probar: (module_path, class_name, init_args_dict)
        agents_to_test = [
            ("agents.sentiment", "EnhancedSentimentAnalyst", {}),
            ("agents.technical_fixed", "EnhancedTechnicalAnalyst", {}),
            ("agents.enhanced_visual_analyst", "EnhancedVisualAnalystAgent", {}),
            ("agents.risk_v2", "AdvancedRiskManager", { # Argumentos para AdvancedRiskManager
                "symbol_tick_size": 0.01,
                "symbol_step_size": 0.001,
                "min_notional": 5.0
            }),
        ]

        for module_path, class_name, init_args in agents_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                AgentClass = getattr(module, class_name)
                agent_instance = AgentClass(**init_args) # Instanciar con sus argumentos
                
                if agent_instance:
                    agent_instantiation_results[class_name] = "✅ Instanciado exitosamente."
                    logger.info(f"  ✅ Agente '{class_name}' instanciado.")
                else: # Poco probable si getattr no falla, pero por si acaso.
                    agent_instantiation_results[class_name] = f"❌ Falla al instanciar (retornó None) '{class_name}'."
                    all_agents_instantiated = False
                    logger.error(f"  ❌ Agente '{class_name}' no se pudo instanciar (retornó None).")

            except ImportError as e_imp:
                agent_instantiation_results[class_name] = f"⚠️ No se pudo importar el módulo '{module_path}' o la clase '{class_name}': {e_imp}"
                result.warnings.append(agent_instantiation_results[class_name]) # Es una advertencia si un agente no está
                logger.warning(f"  ⚠️ {agent_instantiation_results[class_name]}")
            except Exception as e_init:
                agent_instantiation_results[class_name] = f"❌ Error al instanciar '{class_name}': {str(e_init)}"
                all_agents_instantiated = False # Si un agente crítico falla al instanciar, el test general podría fallar.
                logger.error(f"  ❌ Error al instanciar agente '{class_name}': {e_init}", exc_info=False) # exc_info=False para no llenar el log si es común.

        result.data["agent_instantiation_results"] = agent_instantiation_results
        
        # Decidir si el test pasa. Podría ser que todos deben instanciarse, o solo algunos "críticos".
        # El código original consideraba éxito si al menos 2 funcionaban.
        # Aquí, si alguno falla la instanciación (que no sea ImportError), se considera un problema.
        successful_instantiations = sum(1 for status in agent_instantiation_results.values() if "✅" in status)
        total_agents_attempted = len(agents_to_test)

        if successful_instantiations < total_agents_attempted:
            # Contar cuántos fallaron por error vs cuántos por importación (advertencia)
            critical_failures = sum(1 for status in agent_instantiation_results.values() if "❌" in status)
            if critical_failures > 0:
                 result.error_message = f"{critical_failures} agente(s) fallaron la instanciación críticamente."
                 # No retornamos False aquí para que el reporte muestre todos los resultados,
                 # el generador del reporte final decidirá el estado.
                 # Sin embargo, para la lógica de run_all_tests, un error aquí debería ser False.
                 return False # Si un agente no se puede instanciar por error, es un fallo.
            else: # Solo fallos de importación (advertencias)
                 logger.info(f"  ✅ {successful_instantiations}/{total_agents_attempted} agentes instanciados (algunos módulos de agente podrían no estar presentes).")

        else:
            logger.info(f"  ✅ Todos los {successful_instantiations}/{total_agents_attempted} agentes definidos fueron instanciados exitosamente.")
        
        return True # Pasa si no hay errores críticos de instanciación. Las fallas de importación son warnings.

    def test_agents_integration(self, result: IntegrationTestResult) -> bool:
        """Test de integración entre agentes: mockeo de outputs y serialización."""
        try:
            from models.outputs import create_mock_outputs_for_testing, convert_to_decision_inputs
            logger.info("  ✅ Imports para 'test_agents_integration' (models.outputs) exitosos.")

            mock_outputs = create_mock_outputs_for_testing()
            if not mock_outputs:
                result.error_message = "create_mock_outputs_for_testing() retornó vacío en test_agents_integration."
                return False
            
            # Test de conversión a inputs (similar a test_unified_models, pero enfocado en el flujo)
            sentiment_mock = mock_outputs.get("sentiment")
            technical_mock = mock_outputs.get("technical")
            visual_mock = mock_outputs.get("visual")
            qabba_mock = mock_outputs.get("qabba") # Asumiendo que existe

            if not all([sentiment_mock, technical_mock, visual_mock, qabba_mock]):
                result.error_message = "Faltan outputs mock para la prueba de integración de conversión."
                return False
            
            try:
                # El código original no usaba el resultado de convert_to_decision_inputs directamente para validación aquí.
                # Se asume que si no hay error en la conversión, está bien para este test.
                _ = convert_to_decision_inputs(
                    sentiment_mock, technical_mock, visual_mock, qabba_mock
                )
                logger.info("  ✅ Conversión de outputs mock a inputs de decisión (flujo de integración) funcionó.")
            except Exception as e_conv_integ:
                result.error_message = f"Error en conversión (integración): {str(e_conv_integ)}"
                return False

            # Test de serialización (model_dump y conversión a JSON)
            serialization_issues = []
            for output_type, output_instance in mock_outputs.items():
                try:
                    dumped_data = output_instance.model_dump()
                    if not isinstance(dumped_data, dict):
                        serialization_issues.append(f"model_dump() de '{output_type}' no retornó un diccionario.")
                        continue # No intentar json.dumps si no es dict

                    json_string = json.dumps(dumped_data, default=str) # default=str para manejar tipos como datetime
                    if len(json_string) < 10: # Un JSON serializado no debería ser tan corto
                        serialization_issues.append(f"JSON string para '{output_type}' parece demasiado corto o vacío.")
                    
                    # Opcional: deserializar para confirmar
                    # loaded_data = json.loads(json_string)
                    # if not loaded_data:
                    #    serialization_issues.append(f"JSON string para '{output_type}' no pudo ser recargado o resultó vacío.")

                except AttributeError:
                     serialization_issues.append(f"'{output_type}' (tipo: {type(output_instance).__name__}) no tiene model_dump(). No es Pydantic o está mal formado.")
                except Exception as e_serial:
                    serialization_issues.append(f"Error serializando '{output_type}': {str(e_serial)}")
            
            if serialization_issues:
                result.warnings.extend(serialization_issues) # Son advertencias, pero podrían indicar problemas.
                result.data["serialization_issues"] = serialization_issues
                logger.warning(f"  ⚠️ Problemas de serialización detectados: {serialization_issues}")
            else:
                logger.info("  ✅ Serialización de todos los outputs mock (model_dump y JSON) funcionó.")

            result.data["integration_test_status"] = "Completado con éxito (o con advertencias)."
            return True # Pasa incluso con advertencias de serialización, el reporte lo detallará.
            
        except ImportError:
            result.error_message = "No se pudieron importar modelos para 'test_agents_integration'."
            logger.error(f"  ❌ {result.error_message}")
            return False
        except Exception as e:
            result.error_message = f"Error inesperado en test_agents_integration: {str(e)}"
            logger.error(f"  ❌ {result.error_message}", exc_info=True)
            return False

    def test_risk_manager(self, result: IntegrationTestResult) -> bool:
        """Test específico del Risk Manager: instanciación y ejecución con decisiones mock."""
        try:
            from agents.risk_v2 import AdvancedRiskManager
            # from models.outputs import create_mock_outputs_for_testing # No se usa directamente aquí, pero sí los conceptos.
            logger.info("  ✅ Import de 'agents.risk_v2.AdvancedRiskManager' exitoso.")

            try:
                risk_manager_instance = AdvancedRiskManager(
                    symbol_tick_size=0.01, symbol_step_size=0.001, min_notional=5.0
                )
            except Exception as e_init_rm:
                result.error_message = f"Error al instanciar AdvancedRiskManager: {e_init_rm}"
                return False
            logger.info("  ✅ AdvancedRiskManager instanciado.")

            # Test con decisión HOLD (debería ser vetada o manejada específicamente)
            # El método 'run' espera: proposal_decision, current_balance, tech_metrics
            # tech_metrics debe contener al menos 'last_price' y 'atr' según el código original.
            hold_assessment = risk_manager_instance.run(
                proposal_decision="HOLD", # String, como en el código original
                current_balance=10000.0,
                tech_metrics={"last_price": 100.0, "atr": 1.5, "rsi": 50, "adx": 20} # Añadir más métricas si son usadas
            )
            
            # Asumiendo que RiskAssessment tiene 'verdict'
            if not hasattr(hold_assessment, 'verdict'):
                result.error_message = "Resultado de RiskManager.run() no tiene atributo 'verdict'."
                return False

            if hold_assessment.verdict != "VETO": # El código original esperaba VETO para HOLD.
                result.warnings.append(f"Risk Manager no vetó una decisión 'HOLD' como se esperaba. Veredicto: {hold_assessment.verdict}")
                logger.warning(f"  ⚠️ Risk Manager no vetó 'HOLD'. Veredicto: {hold_assessment.verdict}")
            else:
                logger.info("  ✅ Risk Manager vetó decisión 'HOLD' correctamente.")
            result.data["hold_assessment_verdict"] = hold_assessment.verdict


            # Test con decisión BUY válida
            buy_assessment = risk_manager_instance.run(
                proposal_decision="BUY",
                current_balance=10000.0,
                tech_metrics={"last_price": 100.0, "atr": 1.5, "rsi": 45.0, "adx": 30.0} # Ejemplo de métricas
            )
            if not hasattr(buy_assessment, 'verdict'): # Chequeo redundante si el anterior pasó, pero bueno para robustez.
                result.error_message = "Resultado de RiskManager.run() para BUY no tiene atributo 'verdict'."
                return False

            valid_verdicts = ["APPROVE", "APPROVE_REDUCED", "VETO"] # Veredictos esperados
            if buy_assessment.verdict not in valid_verdicts:
                result.error_message = f"Veredicto de Risk Manager para BUY ('{buy_assessment.verdict}') no es uno de los esperados: {valid_verdicts}"
                return False
            logger.info(f"  ✅ Risk Manager evaluó BUY. Veredicto: {buy_assessment.verdict}")
            result.data["buy_assessment_verdict"] = buy_assessment.verdict

            if buy_assessment.verdict in ["APPROVE", "APPROVE_REDUCED"]:
                if not hasattr(buy_assessment, 'order_details') or not buy_assessment.order_details:
                    result.warnings.append("Evaluación de BUY aprobada/reducida pero sin 'order_details' o está vacío.")
                else:
                    details = buy_assessment.order_details
                    # Asumiendo que order_details es un objeto/dict con estos campos
                    if not hasattr(details, 'position_size_contracts') or details.position_size_contracts <= 0:
                         result.warnings.append("Order_details para BUY aprobado/reducido tiene position_size_contracts <= 0.")
                    if not hasattr(details, 'reward_risk_ratio') or details.reward_risk_ratio < 1.0: # Un R:R < 1 es generalmente malo
                         result.warnings.append("Order_details para BUY aprobado/reducido tiene reward_risk_ratio < 1.0.")
                    
                    log_order_details = {
                        "size": getattr(details, 'position_size_contracts', 'N/A'),
                        "rr_ratio": getattr(details, 'reward_risk_ratio', 'N/A'),
                        "risk_usd": getattr(details, 'risk_amount_usd', 'N/A')
                    }
                    logger.info(f"    Detalles de orden para BUY: {log_order_details}")
                    result.data["buy_order_details"] = log_order_details


            # Guardar un resumen del assessment para el reporte
            result.data["final_risk_assessment_summary"] = {
                "verdict": getattr(buy_assessment, 'verdict', 'N/A'),
                "reason": getattr(buy_assessment, 'reason', 'N/A'),
                "risk_score": getattr(buy_assessment, 'risk_score', 'N/A')
            }
            return True
            
        except ImportError:
            result.error_message = "No se pudo importar 'agents.risk_v2.AdvancedRiskManager'."
            logger.error(f"  ❌ {result.error_message}")
            return False
        except Exception as e:
            result.error_message = f"Error inesperado en test_risk_manager: {str(e)}"
            logger.error(f"  ❌ {result.error_message}", exc_info=True)
            return False

    def test_websocket_simulation(self, result: IntegrationTestResult) -> bool:
        """Test de simulación de datos WebSocket: procesamiento de klines y obtención de indicadores."""
        try:
            from tools.technical_tools import add_kline, get_current_indicators, clear_all_buffers, get_buffer_status
            logger.info("  ✅ Imports para 'test_websocket_simulation' (technical_tools) exitosos.")

            clear_all_buffers()
            logger.info("  ✅ Buffers técnicos limpiados para simulación WebSocket.")

            # Mock de datos kline como streams de WebSocket los enviarían (ej. JSON strings o dicts)
            # El código original usa dicts con keys 'c', 'h', 'l', 'v' como strings.
            mock_klines_stream = [
                {'c': '100.50', 'h': '101.00', 'l': '100.00', 'v': '1500.0'},
                {'c': '100.75', 'h': '101.25', 'l': '100.25', 'v': '1600.0'},
                {'c': '100.60', 'h': '101.10', 'l': '100.40', 'v': '1550.0'},
                # Añadir más klines si se necesitan para que los indicadores se calculen (e.g., > 20 para RSI/MACD)
                *[ {'c': str(100.60 + i*0.01), 'h': str(101.10 + i*0.01), 'l': str(100.40 + i*0.01), 'v': str(1550 + i*10) } for i in range(20) ]
            ]
            
            processed_klines_count = 0
            for idx, kline_data_dict in enumerate(mock_klines_stream):
                try:
                    # Convertir strings a float como se espera en add_kline
                    close_price = float(kline_data_dict['c'])
                    high_price = float(kline_data_dict['h'])
                    low_price = float(kline_data_dict['l'])
                    volume = float(kline_data_dict['v'])
                    
                    # add_kline espera (close, high, low, volume)
                    if add_kline(close_price, high_price, low_price, volume):
                        processed_klines_count += 1
                    else:
                        result.warnings.append(f"add_kline retornó False para kline simulada #{idx+1}.")
                        logger.warning(f"  ⚠️ add_kline falló para kline simulada #{idx+1}: C={close_price}, H={high_price}, L={low_price}, V={volume}")
                        
                except KeyError as ke:
                    result.warnings.append(f"Kline simulada #{idx+1} tiene formato incorrecto (falta key: {ke}).")
                    logger.warning(f"  ⚠️ Kline simulada #{idx+1} con formato incorrecto: {kline_data_dict}")
                except ValueError as ve:
                    result.warnings.append(f"Error convirtiendo datos de kline simulada #{idx+1} a float: {ve}.")
                    logger.warning(f"  ⚠️ Error de conversión en kline simulada #{idx+1}: {kline_data_dict}")
                except Exception as e_proc:
                    result.warnings.append(f"Error inesperado procesando kline simulada #{idx+1}: {e_proc}.")
                    logger.warning(f"  ⚠️ Error procesando kline simulada #{idx+1}: {e_proc}")

            if processed_klines_count == 0 and mock_klines_stream: # Si había klines para procesar pero ninguna se procesó
                result.error_message = "No se pudo procesar ninguna kline simulada de WebSocket."
                return False
            
            logger.info(f"  ✅ Procesadas {processed_klines_count}/{len(mock_klines_stream)} klines simuladas.")
            result.data["processed_simulated_klines_count"] = processed_klines_count
            
            # Verificar que se pueden obtener indicadores si hay suficientes datos
            buffer_info = get_buffer_status()
            min_data_for_indicators = 20 # Un umbral razonable para la mayoría de los indicadores
            if buffer_info.get("close_buf", 0) >= min_data_for_indicators:
                final_indicators = get_current_indicators()
                if not final_indicators:
                     result.warnings.append("get_current_indicators retornó vacío después de procesar klines simuladas.")
                else:
                    result.data["final_simulated_indicators"] = {
                        k: f"{v:.4f}" if isinstance(v, float) else v
                        for k, v in final_indicators.items()
                    }
                    logger.info(f"  ✅ Obtenidos {len(final_indicators)} indicadores finales de simulación: {result.data['final_simulated_indicators']}")

                    # Verificar el último precio si es posible
                    if "last_price" in final_indicators and mock_klines_stream:
                        expected_last_price = float(mock_klines_stream[-1]['c'])
                        actual_last_price = final_indicators["last_price"]
                        if abs(actual_last_price - expected_last_price) > 0.001: # Pequeña tolerancia
                            result.warnings.append(f"Precio final de simulación incorrecto: esperado {expected_last_price}, obtenido {actual_last_price}.")
            elif mock_klines_stream : # Si había klines pero no suficientes para indicadores
                 result.warnings.append(f"No se procesaron suficientes klines ({buffer_info.get('close_buf',0)}) para calcular indicadores robustos en simulación.")


            return True
            
        except ImportError:
            result.error_message = "No se pudieron importar herramientas técnicas para 'test_websocket_simulation'."
            logger.error(f"  ❌ {result.error_message}")
            return False
        except Exception as e:
            result.error_message = f"Error inesperado en test_websocket_simulation: {str(e)}"
            logger.error(f"  ❌ {result.error_message}", exc_info=True)
            return False

    def test_performance_logging(self, result: IntegrationTestResult) -> bool:
        """Test del sistema de logging de performance: escritura y lectura de logs."""
        # El directorio "logs" ya debería existir.
        # El directorio "performance_reports" también.
        
        performance_log_file_path = Path("logs") / "test_performance_entries.jsonl" # Usar un archivo de test dedicado

        try:
            # Datos de prueba para un log de performance
            test_log_entry_data = {
                "timestamp": time.time(), # Usar time.time() para timestamp UNIX flotante
                "event_type": "TRADE_DECISION",
                "symbol": "TEST/USDT",
                "final_decision": "BUY",
                "risk_verdict": "APPROVE",
                "confidence": 0.85,
                "details": {
                    "sentiment_score": 0.7,
                    "technical_signal_strength": 0.9,
                    "price_at_decision": 12345.67
                },
                "execution_time_ms": 150.7
            }

            # Escribir entrada de log de prueba
            try:
                with open(performance_log_file_path, "a") as f: # Usar 'a' para append, aunque para test podría ser 'w'
                    f.write(json.dumps(test_log_entry_data) + "\n")
                logger.info(f"  ✅ Entrada de log de performance de prueba escrita en '{performance_log_file_path}'.")
            except IOError as e_io_write:
                result.error_message = f"Error de IO al escribir en el log de performance de prueba '{performance_log_file_path}': {e_io_write}"
                return False

            # Verificar que se puede leer y que el contenido es correcto
            try:
                with open(performance_log_file_path, "r") as f:
                    lines = f.readlines()
                if not lines:
                    result.error_message = f"Archivo de log de performance de prueba '{performance_log_file_path}' está vacío después de escribir."
                    return False
                
                # Leer la última línea (la que acabamos de escribir si usamos 'a', o la única si usamos 'w')
                last_line_json = lines[-1].strip()
                loaded_log_entry = json.loads(last_line_json)
                
                # Comparar algunos campos clave
                if loaded_log_entry.get("symbol") != test_log_entry_data["symbol"] or \
                   loaded_log_entry.get("final_decision") != test_log_entry_data["final_decision"]:
                    result.error_message = "Log de performance de prueba no se guardó o cargó correctamente (discrepancia de datos)."
                    result.data["loaded_test_log_entry"] = loaded_log_entry # Para debug
                    return False
                logger.info("  ✅ Entrada de log de performance de prueba leída y verificada.")

            except IOError as e_io_read:
                result.error_message = f"Error de IO al leer el log de performance de prueba '{performance_log_file_path}': {e_io_read}"
                return False
            except json.JSONDecodeError as e_json:
                result.error_message = f"Error al decodificar JSON del log de performance de prueba '{performance_log_file_path}': {e_json}"
                return False
            
            # Limpiar archivo de prueba (opcional, pero bueno para no dejar basura)
            try:
                performance_log_file_path.unlink()
                logger.info(f"  ✅ Archivo de log de performance de prueba '{performance_log_file_path}' eliminado.")
            except OSError as e_os_unlink:
                result.warnings.append(f"No se pudo eliminar el archivo de log de prueba '{performance_log_file_path}': {e_os_unlink}")


            # Verificar que los directorios de logs y reportes existen (ya hecho al inicio y en test_environment)
            # Este chequeo aquí es más una confirmación del estado general.
            for log_related_dir in ["logs", "performance_reports"]:
                if not Path(log_related_dir).is_dir():
                    # Esto sería un error grave si ocurre a estas alturas.
                    result.error_message = f"Directorio '{log_related_dir}' que es esencial para logging no existe."
                    return False # Fallo crítico.
            logger.info("  ✅ Directorios 'logs' y 'performance_reports' confirmados.")
            
            return True
            
        except Exception as e:
            result.error_message = f"Error inesperado en test_performance_logging: {str(e)}"
            logger.error(f"  ❌ {result.error_message}", exc_info=True)
            # Asegurarse de limpiar el archivo de test si se creó y hubo un error después.
            if performance_log_file_path.exists():
                try:
                    performance_log_file_path.unlink()
                except OSError:
                    pass # Ignorar error de borrado en cleanup de error
            return False

    def generate_final_report(self) -> bool:
        """
        Genera el reporte final de tests en consola y en un archivo JSON.
        Retorna True si todos los tests pasaron, False en caso contrario.
        """
        
        total_execution_time = time.monotonic() - self.total_start_time
        
        # Contar resultados
        total_tests_run = len(self.results)
        if total_tests_run == 0:
            logger.warning("No se ejecutó ningún test.")
            # Crear un reporte JSON vacío o con error
            report_file_path = Path("logs") / "integration_test_report.json"
            try:
                with open(report_file_path, "w") as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "error": "No tests were run or results collected.",
                        "summary": {"total_tests": 0, "passed_tests": 0, "failed_tests": 0, "success_rate": 0, "total_warnings":0},
                        "test_results": {}
                    }, f, indent=2, default=str)
                logger.info(f"📄 Reporte de error guardado en: {report_file_path}")
            except IOError:
                logger.error(f"No se pudo escribir el reporte de error en {report_file_path}")
            return False # No pasaron tests si no se ejecutaron.

        passed_tests_count = sum(1 for r in self.results.values() if r.success)
        failed_tests_count = total_tests_run - passed_tests_count
        total_warnings_count = sum(len(r.warnings) for r in self.results.values())
        
        # Loggear resumen en consola
        logger.info(f"\n\n{'='*60}")
        logger.info(f"📊 REPORTE FINAL DE INTEGRACIÓN ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        logger.info(f"{'='*60}")
        logger.info(f"⏱️  Tiempo total de ejecución: {total_execution_time:.2f} segundos")
        logger.info(f"📈 Tests ejecutados: {total_tests_run}")
        logger.info(f"✅ Tests pasados: {passed_tests_count}")
        logger.info(f"❌ Tests fallidos: {failed_tests_count}")
        logger.info(f"⚠️  Total de advertencias: {total_warnings_count}")
        
        logger.info(f"\n📋 DETALLE POR TEST:")
        for test_name, test_result_obj in sorted(self.results.items()): # Ordenar por nombre para consistencia
            status_icon = "✅ PASS" if test_result_obj.success else "❌ FAIL"
            time_str = f"{test_result_obj.execution_time:.2f}s"
            warnings_info = f" ({len(test_result_obj.warnings)} advertencias)" if test_result_obj.warnings else ""
            
            logger.info(f"  {status_icon:<8} {test_name:25} {time_str:>8} {warnings_info}")
            
            if not test_result_obj.success and test_result_obj.error_message:
                logger.error(f"    └─ Error: {test_result_obj.error_message}")
            
            for warning_msg in test_result_obj.warnings:
                logger.warning(f"    └─ Advertencia: {warning_msg}")
            
            # Opcional: loggear datos del test si existen y son relevantes
            # if test_result_obj.data:
            #    logger.debug(f"    └─ Datos: {json.dumps(test_result_obj.data, indent=2, default=str)}")

        # Preparar datos para el reporte JSON
        report_data_dict = {
            "report_generated_at": datetime.now().isoformat(),
            "total_execution_time_seconds": round(total_execution_time, 3),
            "summary": {
                "total_tests": total_tests_run,
                "passed_tests": passed_tests_count,
                "failed_tests": failed_tests_count,
                "success_rate_percent": round((passed_tests_count / total_tests_run) * 100, 2) if total_tests_run > 0 else 0,
                "total_warnings": total_warnings_count
            },
            "detailed_test_results": {
                name: {
                    "status": "PASSED" if res.success else "FAILED",
                    "execution_time_seconds": round(res.execution_time, 3),
                    "error_message": res.error_message if not res.success else "",
                    "warnings": res.warnings,
                    "collected_data": res.data # Incluir datos recolectados por el test
                }
                for name, res in self.results.items()
            }
        }
        
        # Guardar reporte detallado en JSON
        report_file_path = Path("logs") / "integration_test_report.json"
        try:
            with open(report_file_path, "w") as f:
                json.dump(report_data_dict, f, indent=2, default=str) # default=str para manejar tipos no serializables
            logger.info(f"\n📄 Reporte detallado JSON guardado en: {report_file_path.resolve()}")
        except IOError as e_io_report:
            logger.error(f"Error al guardar el reporte JSON en '{report_file_path}': {e_io_report}")
        except TypeError as e_type_report: # Por si json.dump falla con algún tipo de dato no manejado por default=str
            logger.error(f"Error de tipo al serializar el reporte JSON: {e_type_report}. Revisa los 'collected_data'.")


        # Lógica de recomendaciones finales basada en el éxito
        all_tests_passed = (passed_tests_count == total_tests_run)

        if all_tests_passed:
            logger.info(f"\n🎉 ¡TODOS LOS {total_tests_run} TESTS PASARON EXITOSAMENTE!")
            if total_warnings_count > 0:
                logger.info(f"⚠️  ATENCIÓN: Se detectaron {total_warnings_count} advertencias. Revísalas antes de continuar.")
            else:
                logger.info("✅ El sistema parece estar en buen estado según los tests de integración.")
            
            logger.info(f"\n🚀 PRÓXIMOS PASOS SUGERIDOS:")
            logger.info("  1. Revisar cualquier advertencia generada (si aplica).")
            logger.info("  2. Asegurar que las variables de entorno (.env o similar) estén correctamente configuradas para el entorno deseado.")
            logger.info("  3. Verificar que servicios externos (como Ollama, APIs) estén operativos y accesibles.")
            logger.info("  4. Considerar pruebas en un entorno de staging o testnet antes de producción.")
            
        elif passed_tests_count >= total_tests_run * 0.7:  # Umbral de "mayoría pasó" (e.g., 70%)
            logger.warning(f"\n🟡 ATENCIÓN: {passed_tests_count}/{total_tests_run} TESTS PASARON. ({failed_tests_count} fallaron).")
            logger.warning("   El sistema es parcialmente funcional pero requiere correcciones.")
            logger.warning(f"   Revisa los {failed_tests_count} tests fallidos y {total_warnings_count} advertencias.")
            
            logger.error(f"\n🚨 ERRORES CRÍTICOS A CORREGIR:")
            for name, res_obj in self.results.items():
                if not res_obj.success:
                    logger.error(f"  • Test '{name}': {res_obj.error_message}")
        
        else: # Menos del 70% pasaron
            logger.error(f"\n🔴 ALERTA CRÍTICA: SOLO {passed_tests_count}/{total_tests_run} TESTS PASARON. ({failed_tests_count} fallaron).")
            logger.error("   El sistema presenta problemas significativos y NO está listo para uso.")
            
            logger.info(f"\n🔧 CHECKLIST DE SOLUCIÓN DE PROBLEMAS:")
            logger.info("  1. Revisa los logs detallados en 'logs/integration_test.log' y el reporte JSON.")
            logger.info("  2. Verifica la instalación completa y correcta de TODAS las dependencias (críticas y opcionales si son usadas).")
            logger.info("  3. Confirma que los paths de importación y las estructuras de los módulos son correctos.")
            logger.info("  4. Asegura la correcta definición y validación de los modelos Pydantic.")
            logger.info("  5. Verifica la configuración y disponibilidad de servicios externos (Ollama, APIs, etc.).")
        
        return all_tests_passed


def main():
    """
    Función principal para ejecutar la suite de tests de integración.
    Configura el entorno básico y luego instancia y ejecuta IntegrationTestSuite.
    El código de salida del script reflejará el éxito (0) o fallo (1) de la suite.
    """
    
    # Asegurar que los directorios base existen (redundante si ya se hizo globalmente, pero seguro)
    # 'logs' se crea en la configuración del logger.
    # REQUIRED_DIRECTORIES se crea globalmente.
    # Esta función es el punto de entrada, así que es bueno tenerlo aquí también.
    
    # Crear y ejecutar suite de tests
    test_suite = IntegrationTestSuite()
    overall_success = test_suite.run_all_tests()
    
    # Determinar el código de salida
    exit_code = 0 if overall_success else 1
    
    if overall_success:
        print(f"\n🏆 SUITE DE INTEGRACIÓN COMPLETADA EXITOSAMENTE. Código de salida: {exit_code}")
    else:
        print(f"\n💥 SUITE DE INTEGRACIÓN COMPLETADA CON FALLOS. Código de salida: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    # sys.exit llama a sys.exit() internamente, así que no es necesario anidarlo.
    # Simplemente llama a main y usa su resultado para sys.exit().
    final_exit_code = main()
    sys.exit(final_exit_code)
