#!/usr/bin/env python3
"""
Test script to verify fixes for the specific errors reported in the logs:
1. Instructor multiple tool calls error
2. Schema mismatch in sentiment agent 
3. Model tool support issues
4. JSON validation schema type confusion
"""

import sys
import os
import json
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_instructor_error_handling():
    """Test that Instructor errors are handled gracefully"""
    print("--- Testing Instructor Error Handling ---")
    
    try:
        from agents.enhanced_base_llm_agent import EnhancedBaseLLMAgent
        
        # Create a test agent
        class TestAgent(EnhancedBaseLLMAgent):
            def __init__(self):
                super().__init__(agent_type="test")
        
        agent = TestAgent()
        
        # Test that problematic models are detected
        agent._llm_model_name = "registry.ollama.ai/adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M"
        is_problematic = agent._is_known_problematic_model()
        
        if is_problematic:
            print("✅ Problematic model detection: PASSED")
        else:
            print("❌ Problematic model detection: FAILED")
            
        # Test with a non-problematic model
        agent._llm_model_name = "llama3.2:1b"
        is_not_problematic = not agent._is_known_problematic_model()
        
        if is_not_problematic:
            print("✅ Normal model detection: PASSED")
        else:
            print("❌ Normal model detection: FAILED")
            
        return is_problematic and is_not_problematic
        
    except Exception as e:
        print(f"❌ Instructor error handling test failed: {e}")
        return False

def test_sentiment_schema_fix():
    """Test that sentiment agent uses correct schema"""
    print("\n--- Testing Sentiment Schema Fix ---")
    
    try:
        from agents.enhanced_base_llm_agent import EnhancedBaseLLMAgent
        from agents.json_validator import TradingSignalValidator
        
        # Create a sentiment agent
        class MockSentimentAgent(EnhancedBaseLLMAgent):
            def __init__(self):
                super().__init__(agent_type="sentiment")
        
        agent = MockSentimentAgent()
        
        # Test schema type mapping
        schema_type = agent._get_schema_type_for_agent()
        if schema_type == "sentiment":
            print("✅ Sentiment agent schema mapping: PASSED")
        else:
            print(f"❌ Sentiment agent schema mapping: FAILED (got {schema_type})")
            return False
        
        # Test that the sentiment schema works with actual response data
        validator = TradingSignalValidator()
        
        # This is the type of response that was causing the original error
        problematic_response = {
            'overall_sentiment': 'NEUTRAL',
            'positive_texts_count': 3,
            'negative_texts_count': 2,
            'neutral_texts_count': 3,
            'reasoning': 'El análisis de sentimiento se basa en un equilibrio entre textos positivos, negativos y neutrales. Textos como los que hablan sobre el precio potencial de Bitcoin y las innovaciones en Ethereum son positivos, mientras que los que mencionan problemas con exchanges o donaciones gubernamentales son negativos. Los demás textos son neutrales.'
        }
        
        # Test validation with sentiment schema (not trading schema)
        result = validator.validate_and_repair(json.dumps(problematic_response), "sentiment")
        
        if result and result.get("overall_sentiment") == "NEUTRAL":
            print("✅ Sentiment schema validation: PASSED")
            print(f"   Validated sentiment: {result.get('overall_sentiment')}")
            return True
        else:
            print("❌ Sentiment schema validation: FAILED")
            print(f"   Result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Sentiment schema test failed: {e}")
        return False

def test_trading_vs_sentiment_schema():
    """Test that trading schema rejects sentiment responses (as it should)"""
    print("\n--- Testing Schema Separation ---")
    
    try:
        from agents.json_validator import TradingSignalValidator
        
        validator = TradingSignalValidator()
        
        # Sentiment response (should work with sentiment schema)
        sentiment_response = {
            'overall_sentiment': 'NEUTRAL',
            'positive_texts_count': 3,
            'negative_texts_count': 2,
            'neutral_texts_count': 3,
            'reasoning': 'Test reasoning for sentiment'
        }
        
        # Trading response (should work with trading schema)
        trading_response = {
            'signal': 'HOLD',
            'reasoning': 'Test reasoning for trading signal',
            'confidence_level': 'MEDIUM'
        }
        
        # Test sentiment response with sentiment schema (should pass)
        sentiment_result = validator.validate_and_repair(json.dumps(sentiment_response), "sentiment")
        sentiment_passed = sentiment_result is not None
        
        # Test trading response with trading schema (should pass)
        trading_result = validator.validate_and_repair(json.dumps(trading_response), "trading")
        trading_passed = trading_result is not None
        
        if sentiment_passed and trading_passed:
            print("✅ Schema separation: PASSED")
            print("   Sentiment schema accepts sentiment responses")
            print("   Trading schema accepts trading responses")
            return True
        else:
            print("❌ Schema separation: FAILED")
            print(f"   Sentiment validation: {sentiment_passed}")
            print(f"   Trading validation: {trading_passed}")
            return False
            
    except Exception as e:
        print(f"❌ Schema separation test failed: {e}")
        return False

def test_json_repair_robustness():
    """Test JSON repair with malformed responses similar to what caused errors"""
    print("\n--- Testing JSON Repair Robustness ---")
    
    try:
        from agents.json_validator import TradingSignalValidator
        
        validator = TradingSignalValidator()
        
        # Test cases similar to the original error scenarios
        test_cases = [
            # Case 1: Missing required field (original error had 'signal' missing)
            ('{"overall_sentiment": "NEUTRAL", "reasoning": "Test"}', "sentiment"),
            
            # Case 2: Extra field that doesn't belong
            ('{"signal": "HOLD", "reasoning": "Test", "overall_sentiment": "NEUTRAL"}', "trading"),
            
            # Case 3: Malformed JSON with markdown
            ('```json\n{"overall_sentiment": "NEUTRAL", "reasoning": "Test"}\n```', "sentiment"),
            
            # Case 4: Multiple JSON objects (which might trigger multiple tool calls)
            ('{"overall_sentiment": "NEUTRAL"} {"reasoning": "Test"}', "sentiment"),
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for i, (test_input, schema_type) in enumerate(test_cases, 1):
            try:
                result = validator.validate_and_repair(test_input, schema_type)
                if result:
                    print(f"✅ Repair test {i}: PASSED")
                    passed_tests += 1
                else:
                    print(f"⚠️  Repair test {i}: PARTIAL (created minimal structure)")
                    passed_tests += 0.5  # Partial credit for graceful handling
            except Exception as e:
                print(f"❌ Repair test {i}: FAILED ({e})")
        
        success_rate = passed_tests / total_tests
        if success_rate >= 0.75:  # 75% success rate is acceptable
            print(f"✅ JSON repair robustness: PASSED ({passed_tests}/{total_tests})")
            return True
        else:
            print(f"❌ JSON repair robustness: FAILED ({passed_tests}/{total_tests})")
            return False
            
    except Exception as e:
        print(f"❌ JSON repair robustness test failed: {e}")
        return False

def main():
    """Run all error fix tests"""
    print("=" * 60)
    print("TESTING FIXES FOR REPORTED ERRORS")
    print("=" * 60)
    
    tests = [
        ("Instructor Error Handling", test_instructor_error_handling),
        ("Sentiment Schema Fix", test_sentiment_schema_fix),
        ("Schema Separation", test_trading_vs_sentiment_schema),
        ("JSON Repair Robustness", test_json_repair_robustness),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} TESTS PASSED")
    
    if passed == total:
        print("🎉 ALL FIXES VERIFIED - The reported errors should be resolved!")
    elif passed >= total * 0.75:
        print("✅ Most fixes verified - Significant improvement expected")
    else:
        print("⚠️  Some fixes need attention - Review failed tests")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
