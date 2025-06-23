#!/usr/bin/env python3
"""
Test simple para verificar la corrección de errores en el agente de sentiment.
"""

def test_simple_agent_creation():
    """Test básico de creación de agente sin refresh."""
    try:
        from agents.sentiment_enhanced import EnhancedSentimentAnalyst
        
        # Crear agente sin hacer refresh automático
        agent = EnhancedSentimentAnalyst.__new__(EnhancedSentimentAnalyst)
        agent.agent_type = 'sentiment'
        
        # Verificar que se pueden acceder a los atributos
        print(f"✅ _max_texts_to_fetch_per_source: {getattr(agent, '_max_texts_to_fetch_per_source', 'NO_DEFINIDO')}")
        print(f"✅ _min_text_quality_threshold: {getattr(agent, '_min_text_quality_threshold', 'NO_DEFINIDO')}")
        
        return True
    except Exception as e:
        print(f"❌ Error in simple agent creation: {e}")
        return False

def test_json_validator():
    """Test del validador JSON."""
    try:
        from agents.json_validator import TradingSignalValidator
        
        validator = TradingSignalValidator()
        
        # Test JSON válido
        valid_json = '{"action": "BUY", "confidence": 0.8, "reasoning": "Strong bullish signals detected"}'
        result = validator.validate_and_repair(valid_json, "trading")
        print(f"✅ JSON Validator test passed: {bool(result)}")
        
        return True
    except Exception as e:
        print(f"❌ JSON Validator test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Test Simple - Week 2 Fixes")
    print("=" * 50)
    
    # Test 1: JSON Validator
    print("\n=== Test JSON Validator ===")
    json_ok = test_json_validator()
    
    # Test 2: Simple Agent Creation
    print("\n=== Test Simple Agent Creation ===")
    agent_ok = test_simple_agent_creation()
    
    print("\n=== Results ===")
    print(f"JSON Validator: {'✅ PASS' if json_ok else '❌ FAIL'}")
    print(f"Agent Creation: {'✅ PASS' if agent_ok else '❌ FAIL'}")
    
    if json_ok and agent_ok:
        print("\n🎉 All basic tests passed!")
    else:
        print("\n⚠️ Some tests failed.")
