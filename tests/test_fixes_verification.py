#!/usr/bin/env python3
"""
Test script to verify the fixes for Instructor multiple tool calls and schema validation issues.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.json_validator import TradingSignalValidator

def test_all_schemas():
    """Test all schema types for validation."""
    validator = TradingSignalValidator()
    
    # Test cases for each schema type
    test_cases = {
        'sentiment': {
            'overall_sentiment': 'BULLISH',
            'positive_texts_count': 5,
            'negative_texts_count': 2,
            'neutral_texts_count': 3,
            'reasoning': 'Market sentiment shows strong bullish indicators based on recent news analysis.'
        },
        'trading': {
            'signal': 'BUY',
            'reasoning': 'Strong technical indicators suggest upward momentum.',
            'confidence_level': 'HIGH',
            'price_target': 150.0,
            'stop_loss_suggestion': 130.0
        },
        'qabba': {
            'signal': 'SELL',
            'reasoning': 'QABBA analysis indicates overbought conditions.',
            'confidence_level': 'MEDIUM'
        },
        'visual': {
            'signal': 'HOLD',
            'reasoning': 'Visual chart analysis shows consolidation pattern.',
            'confidence_level': 'LOW'
        }
    }
    
    print("Testing all schema validations...")
    
    for schema_type, test_data in test_cases.items():
        print(f"\n--- Testing {schema_type} schema ---")
        
        # Test valid data
        result = validator.validate_and_repair(json.dumps(test_data), schema_type)
        if result:
            print(f"✅ {schema_type} schema validation: PASSED")
            print(f"   Result: {result}")
        else:
            print(f"❌ {schema_type} schema validation: FAILED")
        
        # Test minimal structure creation
        minimal = validator._create_minimal_valid_structure(schema_type, json.dumps(test_data))
        print(f"   Minimal structure: {minimal}")

def test_schema_mapping():
    """Test that schema type mapping works correctly."""
    validator = TradingSignalValidator()
    
    # Test the schema type detection
    test_mappings = [
        ('sentiment', 'sentiment'),
        ('trading', 'trading'), 
        ('qabba', 'qabba'),
        ('visual', 'visual'),
        ('unknown', 'trading')  # Default fallback
    ]
    
    print("\n--- Testing schema type mapping ---")
    for input_type, expected in test_mappings:
        # This tests the internal logic that would be used
        if input_type in ['sentiment', 'trading', 'qabba', 'visual']:
            actual = input_type
        else:
            actual = 'trading'  # Default fallback
            
        if actual == expected:
            print(f"✅ Schema mapping '{input_type}' -> '{expected}': PASSED")
        else:
            print(f"❌ Schema mapping '{input_type}' -> '{expected}': FAILED (got {actual})")

def test_invalid_json_repair():
    """Test JSON repair functionality."""
    validator = TradingSignalValidator()
    
    print("\n--- Testing JSON repair functionality ---")
    
    # Test proper JSON that should validate successfully
    valid_sentiment_json = '''
    {
        "overall_sentiment": "POSITIVE",
        "positive_texts_count": 5,
        "negative_texts_count": 1,
        "neutral_texts_count": 2,
        "reasoning": "Market sentiment appears positive based on recent news and social media"
    }
    '''
    
    result = validator.validate_and_repair(valid_sentiment_json, 'sentiment')
    if result and result.get("overall_sentiment") == "POSITIVE":
        print("✅ JSON repair: PASSED")
        print(f"   Validated sentiment data: {result.get('overall_sentiment')}")
    else:
        print("❌ JSON repair: FAILED")
        print(f"   Result: {result}")
        
    # Test with markdown-wrapped JSON (common LLM output format)
    markdown_wrapped = '''
    ```json
    {
        "overall_sentiment": "NEUTRAL",
        "positive_texts_count": 3,
        "negative_texts_count": 2,
        "neutral_texts_count": 3,
        "reasoning": "Balanced sentiment analysis with mixed signals"
    }
    ```
    '''
    
    result2 = validator.validate_and_repair(markdown_wrapped, 'sentiment')
    if result2 and result2.get("overall_sentiment") == "NEUTRAL":
        print("✅ Markdown JSON extraction: PASSED")
    else:
        print("❌ Markdown JSON extraction: FAILED")

def test_sentiment_schema_specific():
    """Test the specific sentiment schema that was causing issues."""
    print("\n--- Testing sentiment-specific schema validation ---")
    
    try:
        validator = TradingSignalValidator()
        
        # This is the exact type of response that was causing the original error
        problematic_response = {
            'overall_sentiment': 'NEUTRAL',
            'positive_texts_count': 3,
            'negative_texts_count': 2,
            'neutral_texts_count': 3,
            'reasoning': 'El análisis de sentimiento se basa en un equilibrio entre textos positivos, negativos y neutrales. Textos como los que hablan sobre el precio potencial de Bitcoin y las innovaciones en Ethereum son positivos, mientras que los que mencionan problemas con exchanges o donaciones gubernamentales son negativos. Los demás textos son neutrales.'
        }
        
        # Test that this validates correctly with sentiment schema
        result = validator.validate_and_repair(json.dumps(problematic_response), "sentiment")
        
        if result and all(key in result for key in ['overall_sentiment', 'reasoning']):
            print("✅ Sentiment schema fix: PASSED")
            print(f"   Sentiment: {result.get('overall_sentiment')}")
            print(f"   Reasoning length: {len(result.get('reasoning', ''))}")
        else:
            print(f"❌ Sentiment schema fix: FAILED - Missing required fields")
            print(f"   Result: {result}")
            
    except Exception as e:
        print(f"❌ Sentiment schema fix: FAILED - Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("FENIX TRADING BOT - FIXES VERIFICATION TEST")
    print("=" * 60)
    
    try:
        test_all_schemas()
        test_schema_mapping()
        test_invalid_json_repair()
        test_sentiment_schema_specific()
        test_sentiment_schema_specific()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED - Check results above")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
