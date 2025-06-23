#!/usr/bin/env python3

"""
Test script to verify that qwen2.5 models are properly identified as problematic
and skip Instructor setup to avoid the "multiple tool calls" issue.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.enhanced_base_llm_agent import EnhancedBaseLLMAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen_model_detection():
    """Test that qwen2.5 models are properly detected as problematic."""
    
    test_models = [
        "qwen2.5:7b-instruct-q5_k_m",
        "qwen2.5:7b-instruct", 
        "qwen2.5",
        "nous-hermes2pro",  # Known problematic
        "llama3.1:8b",  # Should not be problematic
    ]
    
    expected_results = {
        "qwen2.5:7b-instruct-q5_k_m": True,
        "qwen2.5:7b-instruct": True,
        "qwen2.5": True,
        "nous-hermes2pro": True,
        "llama3.1:8b": False,
    }
    
    # Create a minimal agent instance to test the method
    class TestAgent(EnhancedBaseLLMAgent):
        agent_type = "test"
        
        def __init__(self, model_name):
            # Minimal initialization
            self._llm_model_name = model_name
    
    all_passed = True
    
    for model_name, expected_problematic in expected_results.items():
        agent = TestAgent(model_name)
        is_problematic = agent._is_known_problematic_model()
        
        if is_problematic == expected_problematic:
            status = "✅"
        else:
            status = "❌"
            all_passed = False
            
        logger.info(f"{status} {model_name}: Expected {expected_problematic}, Got {is_problematic}")
    
    return all_passed

if __name__ == "__main__":
    success = test_qwen_model_detection()
    if success:
        print("\n🎉 All model detection tests passed! qwen2.5 models will now skip Instructor and go straight to raw query mode.")
    else:
        print("\n❌ Some tests failed. Check the logs above.")
        sys.exit(1)
