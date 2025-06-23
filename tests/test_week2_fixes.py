#!/usr/bin/env python3
"""
Test para verificar las correcciones de la Semana 2 - JSON Validation Integration
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported correctly."""
    print("=== Testing Module Imports ===")
    
    try:
        from agents.json_validator import TradingSignalValidator
        print("✅ JSON Validator imported successfully")
    except Exception as e:
        print(f"❌ JSON Validator import failed: {e}")
        return False
        
    try:
        from agents.enhanced_base_llm_agent import EnhancedBaseLLMAgent
        print("✅ Enhanced Base Agent imported successfully")
    except Exception as e:
        print(f"❌ Enhanced Base Agent import failed: {e}")
        return False
        
    try:
        from agents.sentiment_enhanced import EnhancedSentimentAnalyst
        print("✅ Sentiment Agent imported successfully")
    except Exception as e:
        print(f"❌ Sentiment Agent import failed: {e}")
        return False
        
    return True

def test_agent_creation():
    """Test creating agents with the new configuration."""
    print("\n=== Testing Agent Creation ===")
    
    try:
        from agents.sentiment_enhanced import EnhancedSentimentAnalyst
        
        # Create sentiment agent with proper agent_type
        agent = EnhancedSentimentAnalyst(agent_type='sentiment')
        print(f"✅ Sentiment agent created: {agent.agent_type}")
        print(f"   Model: {agent._llm_model_name}")
        print(f"   Max texts per source: {agent._max_texts_to_fetch_per_source}")
        print(f"   Min quality threshold: {agent._min_text_quality_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_configuration():
    """Test that models are properly configured."""
    print("\n=== Testing Model Configuration ===")
    
    try:
        from config.modern_models import model_manager, HETEROGENEOUS_MODELS_CONFIG
        
        print("Available models in Ollama:")
        for model in model_manager.available_ollama_models:
            print(f"  - {model}")
        
        print("\nConfigured agent types:")
        for agent_type, config in HETEROGENEOUS_MODELS_CONFIG.items():
            available = model_manager._is_model_explicitly_available(config.name)
            status = "✅" if available else "❌"
            print(f"  {status} {agent_type}: {config.name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        return False

def test_json_validation():
    """Test the JSON validation system."""
    print("\n=== Testing JSON Validation ===")
    
    try:
        from agents.json_validator import TradingSignalValidator
        
        validator = TradingSignalValidator()
        
        # Test with valid JSON
        valid_json = '{"signal": "BUY", "confidence": 0.8, "reasoning": "Strong bullish signals"}'
        result = validator.validate_and_repair(valid_json, "trading")
        
        if result:
            print("✅ Valid JSON processed correctly")
            print(f"   Result: {result}")
        else:
            print("❌ Valid JSON validation failed")
            return False
            
        # Test with invalid JSON
        invalid_json = '{"signal": "BUY", "confidence": 0.8, "reasoning": "Strong bullish signals"'  # Missing closing brace
        result = validator.validate_and_repair(invalid_json, "trading")
        
        print(f"✅ Invalid JSON handled: {result is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Week 2 Fixes - JSON Validation Integration\n")
    
    tests = [
        test_imports,
        test_model_configuration,
        test_json_validation,
        test_agent_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Week 2 integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
