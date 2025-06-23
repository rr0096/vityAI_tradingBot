#!/usr/bin/env python3
"""
System Health Check for FenixTradingBot
Verifies all components are properly configured and working.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthChecker:
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.overall_status = True

    def check_ollama_installation(self) -> bool:
        """Check if Ollama is installed and running."""
        logger.info("🔍 Checking Ollama installation...")
        
        try:
            # Check if ollama command exists
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.results['ollama'] = {
                    'status': 'ERROR',
                    'message': 'Ollama command not found. Please install Ollama.',
                    'details': result.stderr
                }
                return False

            # Check if Ollama service is running
            import requests
            try:
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    self.results['ollama'] = {
                        'status': 'OK',
                        'message': f'Ollama is running with {len(models)} models',
                        'version': result.stdout.strip(),
                        'models_count': len(models)
                    }
                    return True
                else:
                    self.results['ollama'] = {
                        'status': 'WARNING',
                        'message': 'Ollama service is not responding properly',
                        'status_code': response.status_code
                    }
                    return False
            except requests.RequestException as e:
                self.results['ollama'] = {
                    'status': 'WARNING',
                    'message': 'Ollama service is not running. Run: ollama serve',
                    'error': str(e)
                }
                return False

        except subprocess.TimeoutExpired:
            self.results['ollama'] = {
                'status': 'ERROR',
                'message': 'Ollama command timed out'
            }
            return False
        except FileNotFoundError:
            self.results['ollama'] = {
                'status': 'ERROR',
                'message': 'Ollama not installed. Visit: https://ollama.ai'
            }
            return False
        except Exception as e:
            self.results['ollama'] = {
                'status': 'ERROR',
                'message': f'Unexpected error checking Ollama: {e}'
            }
            return False

    def check_python_environment(self) -> bool:
        """Check Python version and virtual environment."""
        logger.info("🐍 Checking Python environment...")
        
        try:
            python_version = sys.version_info
            if python_version < (3, 11):
                self.results['python'] = {
                    'status': 'ERROR',
                    'message': f'Python 3.11+ required, found {python_version.major}.{python_version.minor}'
                }
                return False

            # Check if in virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            
            self.results['python'] = {
                'status': 'OK',
                'message': f'Python {python_version.major}.{python_version.minor}.{python_version.micro}',
                'virtual_env': in_venv,
                'executable': sys.executable
            }
            
            if not in_venv:
                self.results['python']['warning'] = 'Not in virtual environment (recommended)'
            
            return True

        except Exception as e:
            self.results['python'] = {
                'status': 'ERROR',
                'message': f'Error checking Python environment: {e}'
            }
            return False

    def check_required_packages(self) -> bool:
        """Check if required packages are installed."""
        logger.info("📦 Checking required packages...")
        
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'mplfinance', 'requests',
            'pydantic', 'crewai', 'instructor', 'openai', 'binance',
            'python-binance', 'talib', 'psutil'
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                installed_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.results['packages'] = {
                'status': 'ERROR',
                'message': f'Missing packages: {", ".join(missing_packages)}',
                'missing': missing_packages,
                'installed': installed_packages,
                'fix': 'Run: pip install -r requirements.txt'
            }
            return False
        else:
            self.results['packages'] = {
                'status': 'OK',
                'message': f'All {len(installed_packages)} required packages installed',
                'installed_count': len(installed_packages)
            }
            return True

    def check_models_configuration(self) -> bool:
        """Check model configuration and availability."""
        logger.info("🤖 Checking models configuration...")
        
        try:
            from config.modern_models import model_manager, MODERN_MODELS_CONFIG
            
            available_models = model_manager.available_ollama_models
            configured_models = list(MODERN_MODELS_CONFIG.keys())
            
            model_status = {}
            working_agents = 0
            
            for agent_type in configured_models:
                config = model_manager.get_model_config(agent_type)
                is_available = model_manager._is_model_explicitly_available(config.name)
                
                model_status[agent_type] = {
                    'configured_model': config.name,
                    'available': is_available,
                    'supports_tools': config.supports_tools,
                    'supports_vision': config.supports_vision
                }
                
                if is_available:
                    working_agents += 1
            
            if working_agents == len(configured_models):
                status = 'OK'
                message = f'All {working_agents} agent models are available'
            elif working_agents > 0:
                status = 'WARNING'
                message = f'{working_agents}/{len(configured_models)} agent models available'
            else:
                status = 'ERROR'
                message = 'No configured models are available'
            
            self.results['models'] = {
                'status': status,
                'message': message,
                'available_models': available_models,
                'agent_models': model_status,
                'working_agents': working_agents,
                'total_agents': len(configured_models)
            }
            
            return status in ['OK', 'WARNING']

        except Exception as e:
            self.results['models'] = {
                'status': 'ERROR',
                'message': f'Error checking models: {e}'
            }
            return False

    def check_configuration_files(self) -> bool:
        """Check if all configuration files exist and are valid."""
        logger.info("⚙️ Checking configuration files...")
        
        config_files = [
            'config/config.yaml',
            'config/modern_models.py',
            'config/monitoring_config.py'
        ]
        
        missing_files = []
        valid_files = []
        
        for config_file in config_files:
            file_path = project_root / config_file
            if file_path.exists():
                try:
                    if config_file.endswith('.yaml'):
                        import yaml
                        with open(file_path, 'r') as f:
                            yaml.safe_load(f)
                    elif config_file.endswith('.py'):
                        # Try to import the module
                        module_name = config_file.replace('/', '.').replace('.py', '')
                        __import__(module_name)
                    valid_files.append(config_file)
                except Exception as e:
                    self.results[f'config_{config_file}'] = {
                        'status': 'ERROR',
                        'message': f'Invalid config file {config_file}: {e}'
                    }
                    return False
            else:
                missing_files.append(config_file)
        
        if missing_files:
            self.results['config'] = {
                'status': 'ERROR',
                'message': f'Missing config files: {", ".join(missing_files)}',
                'missing': missing_files
            }
            return False
        else:
            self.results['config'] = {
                'status': 'OK',
                'message': f'All {len(valid_files)} configuration files are valid',
                'files': valid_files
            }
            return True

    def check_monitoring_system(self) -> bool:
        """Check monitoring system components."""
        logger.info("📊 Checking monitoring system...")
        
        try:
            from monitoring.metrics_collector import MetricsCollector
            from monitoring.alerts import AlertManager
            
            # Test metrics collector initialization
            MetricsCollector()
            
            # Test alert manager initialization
            AlertManager()
            
            self.results['monitoring'] = {
                'status': 'OK',
                'message': 'All monitoring components initialized successfully',
                'components': ['MetricsCollector', 'AlertManager']
            }
            return True

        except Exception as e:
            self.results['monitoring'] = {
                'status': 'ERROR',
                'message': f'Error initializing monitoring: {e}'
            }
            return False

    def check_chart_generator(self) -> bool:
        """Check chart generation system."""
        logger.info("📈 Checking chart generator...")
        
        try:
            from tools.chart_generator import generate_chart_for_visual_agent
            
            # Test with minimal data
            test_data = [100.0, 101.0, 99.0, 102.0, 98.0] * 20  # 100 points
            
            try:
                base64_img, _ = generate_chart_for_visual_agent(
                    symbol="TEST/USDT",
                    timeframe="1m",
                    close_buf=test_data,
                    high_buf=[x + 1 for x in test_data],
                    low_buf=[x - 1 for x in test_data],
                    vol_buf=[1000.0] * len(test_data),
                    lookback_periods=50,
                    save_chart=False
                )
                
                if base64_img and len(base64_img) > 100:  # Valid base64 image
                    self.results['chart_generator'] = {
                        'status': 'OK',
                        'message': 'Chart generator working correctly',
                        'test_result': 'Generated test chart successfully'
                    }
                    return True
                else:
                    self.results['chart_generator'] = {
                        'status': 'WARNING',
                        'message': 'Chart generator returned invalid image',
                        'image_length': len(base64_img) if base64_img else 0
                    }
                    return False

            except Exception as e:
                self.results['chart_generator'] = {
                    'status': 'WARNING',
                    'message': f'Chart generation test failed: {e}',
                    'note': 'May work with real data'
                }
                return False

        except ImportError as e:
            self.results['chart_generator'] = {
                'status': 'ERROR',
                'message': f'Cannot import chart generator: {e}'
            }
            return False

    def check_agent_initialization(self) -> bool:
        """Check if agents can be initialized."""
        logger.info("🤖 Checking agent initialization...")
        
        try:
            from agents.enhanced_base_llm_agent import EnhancedBaseLLMAgent
            
            # Test creating a simple agent
            class TestAgent(EnhancedBaseLLMAgent):
                def __init__(self):
                    super().__init__(
                        role="Test Agent",
                        goal="Test system functionality",
                        backstory="A test agent for system verification",
                        agent_type="sentiment",
                        verbose=False
                    )

            # Try to initialize
            TestAgent()  # Just instantiate to test initialization
            
            self.results['agents'] = {
                'status': 'OK',
                'message': 'Agent initialization successful',
                'base_class': 'EnhancedBaseLLMAgent'
            }
            return True

        except Exception as e:
            self.results['agents'] = {
                'status': 'ERROR',
                'message': f'Agent initialization failed: {e}'
            }
            return False

    def generate_report(self) -> str:
        """Generate a comprehensive health check report."""
        logger.info("📋 Generating health check report...")
        
        report = []
        report.append("=" * 60)
        report.append("🔥 FENIX TRADING BOT - SYSTEM HEALTH CHECK")
        report.append("=" * 60)
        report.append("")
        
        # Overall status
        if self.overall_status:
            report.append("✅ OVERALL STATUS: HEALTHY")
        else:
            report.append("❌ OVERALL STATUS: ISSUES DETECTED")
        report.append("")
        
        # Component details
        for component, details in self.results.items():
            status = details['status']
            if status == 'OK':
                icon = "✅"
            elif status == 'WARNING':
                icon = "⚠️"
            else:
                icon = "❌"
            
            report.append(f"{icon} {component.upper()}: {details['message']}")
            
            # Add additional details for some components
            if component == 'models' and 'agent_models' in details:
                for agent, info in details['agent_models'].items():
                    status_icon = "✅" if info['available'] else "❌"
                    report.append(f"   {status_icon} {agent}: {info['configured_model']}")
            
            if 'warning' in details:
                report.append(f"   ⚠️  {details['warning']}")
            
            if 'fix' in details:
                report.append(f"   🔧 Fix: {details['fix']}")
            
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("")
        
        recommendations = []
        
        if self.results.get('python', {}).get('virtual_env') is False:
            recommendations.append("• Use a virtual environment for better dependency management")
        
        if self.results.get('ollama', {}).get('status') != 'OK':
            recommendations.append("• Install and start Ollama service: https://ollama.ai")
        
        if self.results.get('models', {}).get('working_agents', 0) < 5:
            recommendations.append("• Install missing Ollama models (see docs/OLLAMA_SETUP.md)")
        
        if not recommendations:
            recommendations.append("• System looks good! Ready for trading.")
        
        for rec in recommendations:
            report.append(rec)
        
        report.append("")
        report.append("📚 For detailed setup instructions, see:")
        report.append("   • docs/OLLAMA_SETUP.md - Ollama configuration")
        report.append("   • docs/MONITORING.md - Monitoring system")
        report.append("   • README.md - General setup and usage")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

    def run_all_checks(self) -> bool:
        """Run all health checks."""
        logger.info("🚀 Starting FenixTradingBot health check...")
        
        checks = [
            self.check_python_environment,
            self.check_required_packages,
            self.check_ollama_installation,
            self.check_configuration_files,
            self.check_models_configuration,
            self.check_monitoring_system,
            self.check_chart_generator,
            self.check_agent_initialization
        ]
        
        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                logger.error(f"Check failed with exception: {e}")
                results.append(False)
        
        self.overall_status = all(results)
        return self.overall_status

def main():
    """Main entry point."""
    checker = HealthChecker()
    
    try:
        # Run all checks
        success = checker.run_all_checks()
        
        # Generate and display report
        report = checker.generate_report()
        print(report)
        
        # Save detailed results to file
        results_file = project_root / "logs" / "health_check_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'overall_status': success,
                'results': checker.results
            }, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
