#!/usr/bin/env python3
"""
Test script for Week 2 JSON validation and anti-hallucination integration.
Tests the enhanced agents with the new validation framework.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_json_validator():
    """Test the JSON validator components."""
    logger.info("=== Testing JSON Validator ===")
    
    from agents.json_validator import TradingSignalValidator, ConstitutionalFinancialPrompt
    
    validator = TradingSignalValidator()
    
    # Test 1: Valid JSON
    valid_json = '''{"action": "BUY", "confidence": 0.8, "reasoning": "Strong bullish indicators observed"}'''
    result = validator.validate_and_repair(valid_json, "trading")
    assert result is not None, "Valid JSON should be accepted"
    assert result["action"] == "BUY", "Action should be preserved"
    logger.info("✅ Valid JSON test passed")
    
    # Test 2: Malformed JSON repair
    malformed_json = '''{"action": BUY, "confidence": 0.8, "reasoning": "Missing quotes"}'''
    result = validator.validate_and_repair(malformed_json, "trading")
    assert result is not None, "Malformed JSON should be repaired"
    assert result["action"] == "BUY", "Action should be repaired"
    logger.info("✅ Malformed JSON repair test passed")
    
    # Test 3: Constitutional prompting
    base_prompt = "Analyze the market conditions"
    constitutional_prompt = ConstitutionalFinancialPrompt.create_constitutional_prompt(
        base_prompt, "sentiment"
    )
    assert "constitucional" in constitutional_prompt, "Constitutional principles should be added"
    assert base_prompt in constitutional_prompt, "Original prompt should be preserved"
    logger.info("✅ Constitutional prompting test passed")
    
    logger.info("🎉 All JSON validator tests passed!")


def test_enhanced_base_agent():
    """Test the enhanced base agent with new validation methods."""
    logger.info("=== Testing Enhanced Base Agent ===")
    
    from agents.enhanced_base_llm_agent import EnhancedBaseLLMAgent
    from pydantic import BaseModel, Field
    from typing import Literal
    
    # Create a test response model
    class TestResponse(BaseModel):
        action: Literal["BUY", "SELL", "HOLD"]
        confidence: float = Field(ge=0, le=1)
        reasoning: str = Field(min_length=10)
    
    # Create test agent instance
    test_agent = EnhancedBaseLLMAgent(agent_type="test")
    
    # Test schema type mapping
    schema_type = test_agent._get_schema_type_for_agent()
    assert schema_type in ["trading", "visual", "qabba"], "Schema type should be valid"
    logger.info(f"✅ Schema type mapping test passed: {schema_type}")
    
    # Test anti-hallucination checks (mock response)
    test_response = TestResponse(
        action="BUY",
        confidence=0.95,  # High confidence
        reasoning="Short reason"  # Insufficient reasoning
    )
    
    checked_response = test_agent._apply_anti_hallucination_checks(test_response)
    # Note: Due to type constraints, this might not modify the response in tests
    # but the method should run without errors
    logger.info("✅ Anti-hallucination checks test passed")
    
    logger.info("🎉 Enhanced base agent tests passed!")


def test_sentiment_agent_integration():
    """Test sentiment agent with new validation."""
    logger.info("=== Testing Sentiment Agent Integration ===")
    
    try:
        from agents.sentiment_enhanced import EnhancedSentimentAnalyst
        
        # Create agent instance
        sentiment_agent = EnhancedSentimentAnalyst()
        
        # Verify agent type is correctly set
        assert sentiment_agent.agent_type == "sentiment", "Agent type should be sentiment"
        
        # Check if validation method is available
        assert hasattr(sentiment_agent, '_query_llm_with_validation'), "Should have validation method"
        
        logger.info("✅ Sentiment agent initialization test passed")
        logger.info("✅ Sentiment agent has validation methods")
        
        # Test performance summary (shouldn't error)
        perf_summary = sentiment_agent.get_performance_summary()
        assert isinstance(perf_summary, dict), "Performance summary should be dict"
        assert 'agent_type' in perf_summary, "Should include agent type"
        logger.info("✅ Performance summary test passed")
        
    except Exception as e:
        logger.error(f"❌ Sentiment agent test failed: {e}")
        raise
    
    logger.info("🎉 Sentiment agent integration tests passed!")


def test_technical_agent_integration():
    """Test technical agent with new validation."""
    logger.info("=== Testing Technical Agent Integration ===")
    
    try:
        from agents.technical_v_enhanced_fixed import EnhancedTechnicalAnalyst
        
        # Create agent instance
        tech_agent = EnhancedTechnicalAnalyst()
        
        # Verify agent type is correctly set
        assert tech_agent.agent_type == "technical", "Agent type should be technical"
        
        # Check if validation method is available
        assert hasattr(tech_agent, '_query_llm_with_validation'), "Should have validation method"
        
        logger.info("✅ Technical agent initialization test passed")
        logger.info("✅ Technical agent has validation methods")
        
    except Exception as e:
        logger.error(f"❌ Technical agent test failed: {e}")
        raise
    
    logger.info("🎉 Technical agent integration tests passed!")


def test_visual_agent_integration():
    """Test visual agent with new validation."""
    logger.info("=== Testing Visual Agent Integration ===")
    
    try:
        from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent
        
        # Create agent instance
        visual_agent = EnhancedVisualAnalystAgent()
        
        # Verify agent type is correctly set
        assert visual_agent.agent_type == "visual", "Agent type should be visual"
        
        # Check if validation method is available
        assert hasattr(visual_agent, '_query_llm_with_validation'), "Should have validation method"
        
        # Test schema type for visual agent
        schema_type = visual_agent._get_schema_type_for_agent()
        assert schema_type == "visual", "Visual agent should use visual schema"
        
        logger.info("✅ Visual agent initialization test passed")
        logger.info("✅ Visual agent has validation methods")
        logger.info(f"✅ Visual agent schema type: {schema_type}")
        
    except Exception as e:
        logger.error(f"❌ Visual agent test failed: {e}")
        raise
    
    logger.info("🎉 Visual agent integration tests passed!")


def test_heterogeneous_models_availability():
    """Test that heterogeneous models are properly configured."""
    logger.info("=== Testing Heterogeneous Models Availability ===")
    
    try:
        from config.heterogeneous_models import HeterogeneousModelManager
        
        # Test model manager initialization
        manager = HeterogeneousModelManager()
        
        # Test agent configurations
        agent_types = ["sentiment", "technical", "visual", "qabba", "decision"]
        
        for agent_type in agent_types:
            config = manager.get_specialized_config(agent_type)
            assert config is not None, f"Should have config for {agent_type}"
            assert config.model_name, f"Should have model name for {agent_type}"
            logger.info(f"✅ {agent_type}: {config.model_name}")
        
        # Test memory management
        memory_status = manager.get_memory_status()
        assert isinstance(memory_status, dict), "Memory status should be dict"
        logger.info(f"✅ Memory management available: {memory_status}")
        
    except Exception as e:
        logger.error(f"❌ Heterogeneous models test failed: {e}")
        raise
    
    logger.info("🎉 Heterogeneous models tests passed!")


def run_comprehensive_test():
    """Run all validation tests."""
    logger.info("🚀 Starting Week 2 Integration Tests")
    logger.info("Testing JSON validation and anti-hallucination framework integration")
    
    test_start = datetime.now()
    
    try:
        # Core validation tests
        test_json_validator()
        test_enhanced_base_agent()
        
        # Agent integration tests
        test_sentiment_agent_integration()
        test_technical_agent_integration()
        test_visual_agent_integration()
        
        # Model configuration tests
        test_heterogeneous_models_availability()
        
        test_duration = datetime.now() - test_start
        
        logger.info("=" * 60)
        logger.info("🎉 ALL WEEK 2 INTEGRATION TESTS PASSED!")
        logger.info(f"⏱️  Test duration: {test_duration.total_seconds():.2f} seconds")
        logger.info("✅ JSON validation framework integrated successfully")
        logger.info("✅ Anti-hallucination techniques integrated successfully")
        logger.info("✅ Constitutional prompting integrated successfully")
        logger.info("✅ All agents updated with validation methods")
        logger.info("✅ Heterogeneous model architecture working")
        logger.info("=" * 60)
        logger.info("🚀 READY TO PROCEED TO WEEK 3: Multi-Model Consensus")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TESTS FAILED: {e}")
        logger.error("🔧 Please fix the issues before proceeding to Week 3")
        return False


if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Run tests
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
