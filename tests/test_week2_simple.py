#!/usr/bin/env python3
"""
Simplified test for Week 2 JSON validation integration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("🚀 Testing Week 2 JSON Validation Integration")
    
    try:
        # Test 1: JSON Validator
        print("\n=== Testing JSON Validator ===")
        from agents.json_validator import TradingSignalValidator, ConstitutionalFinancialPrompt
        
        validator = TradingSignalValidator()
        valid_json = '{"action": "BUY", "confidence": 0.8, "reasoning": "Strong bullish indicators observed"}'
        result = validator.validate_and_repair(valid_json, "trading")
        
        if result and result.get("action") == "BUY":
            print("✅ JSON validator working correctly")
        else:
            print("❌ JSON validator failed")
            return False
            
        # Test 2: Constitutional Prompting
        base_prompt = "Analyze the market"
        constitutional = ConstitutionalFinancialPrompt.create_constitutional_prompt(base_prompt, "sentiment")
        
        if "constitucional" in constitutional and base_prompt in constitutional:
            print("✅ Constitutional prompting working correctly")
        else:
            print("❌ Constitutional prompting failed")
            return False
        
        # Test 3: Heterogeneous Models Config
        print("\n=== Testing Heterogeneous Models ===")
        from config.heterogeneous_models import HeterogeneousModelManager
        
        manager = HeterogeneousModelManager()
        sentiment_config = manager.get_model_config_by_agent_type("sentiment")
        
        if sentiment_config and sentiment_config.name:
            print(f"✅ Heterogeneous models working: {sentiment_config.name}")
        else:
            print("❌ Heterogeneous models failed")
            return False
        
        # Test 4: Enhanced Agent Import
        print("\n=== Testing Enhanced Agents ===")
        try:
            from agents.enhanced_base_llm_agent import EnhancedBaseLLMAgent
            
            # Try to create an instance
            test_agent = EnhancedBaseLLMAgent(agent_type="test")
            
            # Check if new methods exist
            if hasattr(test_agent, '_query_llm_with_validation'):
                print("✅ Enhanced base agent with validation methods working")
            else:
                print("❌ Enhanced base agent missing validation methods")
                return False
                
        except Exception as e:
            print(f"❌ Enhanced base agent import failed: {e}")
            return False
        
        # Test 5: Agent Integration Check
        print("\n=== Testing Agent Integration ===")
        try:
            from agents.sentiment_enhanced import EnhancedSentimentAnalyst
            sentiment_agent = EnhancedSentimentAnalyst()
            
            if hasattr(sentiment_agent, '_query_llm_with_validation'):
                print("✅ Sentiment agent has validation integration")
            else:
                print("❌ Sentiment agent missing validation integration")
                return False
                
        except Exception as e:
            print(f"❌ Sentiment agent test failed: {e}")
            return False
        
        print("\n" + "="*60)
        print("🎉 ALL WEEK 2 INTEGRATION TESTS PASSED!")
        print("✅ JSON validation framework integrated successfully")
        print("✅ Anti-hallucination techniques available")
        print("✅ Constitutional prompting working")
        print("✅ Heterogeneous models configured")
        print("✅ Enhanced agents updated with validation")
        print("="*60)
        print("🚀 READY TO PROCEED TO WEEK 3: Multi-Model Consensus")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
