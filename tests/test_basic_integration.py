#!/usr/bin/env python3

print("🧪 Testing Week 2 Integration")

try:
    from agents.json_validator import TradingSignalValidator
    print("✅ JSON Validator imported")
    
    validator = TradingSignalValidator()
    print("✅ JSON Validator created")
    
    # Test simple
    result = validator.validate_and_repair('{"action": "BUY", "confidence": 0.8, "reasoning": "test"}', "trading")
    print(f"✅ JSON validation result: {bool(result)}")
    
except Exception as e:
    print(f"❌ JSON Validator error: {e}")

try:
    from agents.sentiment_enhanced import EnhancedSentimentAnalyst
    print("✅ Sentiment Agent imported")
    
    # Solo test de atributos sin inicialización completa
    agent = EnhancedSentimentAnalyst.__new__(EnhancedSentimentAnalyst)
    print("✅ Sentiment Agent created (basic)")
    
except Exception as e:
    print(f"❌ Sentiment Agent error: {e}")

print("✅ Basic integration test completed")
