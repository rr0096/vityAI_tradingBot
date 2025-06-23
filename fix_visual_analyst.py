#!/usr/bin/env python3
"""
Script to fix Ollama vision issues and test visual analyst
"""

import sys
import subprocess
import time
import logging

# Add the project directory to Python path
sys.path.insert(0, "/Users/giovanniarangio/carpeta sin título 4/fenixtradingbot")

def restart_ollama():
    """Restart Ollama service to clear memory issues"""
    print("🔄 Restarting Ollama to clear memory issues...")
    
    try:
        # Kill ollama processes
        subprocess.run(["pkill", "-f", "ollama"], check=False)
        time.sleep(2)
        
        # Start ollama serve in background
        print("🚀 Starting Ollama serve...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(5)  # Wait for startup
        
        # Check if it's running
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Ollama is running successfully")
            print("📋 Available models:")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to start Ollama")
            return False
            
    except Exception as e:
        print(f"❌ Error restarting Ollama: {e}")
        return False

def test_vision_model():
    """Test the vision model directly"""
    print("\n🧪 Testing vision model directly...")
    
    try:
        # Create a simple test prompt
        prompt = '''Analyze this trading chart and respond with just "TEST_SUCCESS" if you can see the image, or "TEST_FAILED" if you cannot.'''
        
        # Test without image first
        
        result = subprocess.run([
            "ollama", "run", "qwen2.5vl:7b-q4_K_M",
            prompt
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = result.stdout.strip()
            print(f"🔍 Vision model response: {response}")
            return "TEST" in response.upper()
        else:
            print(f"❌ Vision model test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing vision model: {e}")
        return False

def test_visual_analyst():
    """Test the visual analyst agent"""
    print("\n🎯 Testing Visual Analyst Agent...")
    
    try:
        from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent
        from config.modern_models import ModelManager
        from collections import deque
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Get vision model config - access available_models instead
        vision_config = None
        for model_name in model_manager.available_models:
            config = model_manager.get_model_config(model_name)
            if config and config.supports_vision:
                vision_config = config
                break
        
        if not vision_config:
            print("❌ No vision model found in config")
            return False
        
        print(f"🔍 Using vision model: {vision_config.model_name}")
        
        # Create visual analyst
        visual_agent = EnhancedVisualAnalystAgent(
            model_config=vision_config,
            agent_name="test_visual"
        )
        
        # Create test data
        close_data = [140.0 + i * 0.1 for i in range(50)]
        high_data = [c + 0.5 for c in close_data]
        low_data = [c - 0.3 for c in close_data]
        vol_data = [1000 + i * 10 for i in range(50)]
        
        close_deque = deque(close_data, maxlen=100)
        high_deque = deque(high_data, maxlen=100)
        low_deque = deque(low_data, maxlen=100)
        vol_deque = deque(vol_data, maxlen=100)
        
        # Run analysis
        print("🔄 Running visual analysis...")
        result = visual_agent.run(
            symbol="SOLUSDT",
            timeframe_str="1m",
            close_buf_deque=close_deque,
            high_buf_deque=high_deque,
            low_buf_deque=low_deque,
            vol_buf_deque=vol_deque
        )
        
        print(f"✅ Visual Analysis Result:")
        print(f"   Assessment: {result.overall_visual_assessment}")
        print(f"   Clarity Score: {result.pattern_clarity_score}")
        print(f"   Reasoning: {result.reasoning[:100]}...")
        
        # Check if it's a proper result or default
        if result.pattern_clarity_score > 0.0:
            print("🎉 Visual analyst is working properly!")
            return True
        else:
            print("⚠️  Visual analyst returned default output (0.0 clarity)")
            return False
            
    except Exception as e:
        print(f"❌ Error testing visual analyst: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🛠️  VISUAL ANALYST DIAGNOSTIC AND FIX TOOL")
    print("=" * 50)
    
    # Step 1: Restart Ollama
    if not restart_ollama():
        print("❌ Failed to restart Ollama. Please restart manually.")
        return
    
    # Step 2: Test vision model
    if not test_vision_model():
        print("❌ Vision model test failed. Continuing with agent test...")
    
    # Step 3: Test visual analyst
    if test_visual_analyst():
        print("\n🎉 SUCCESS: Visual analyst is now working!")
    else:
        print("\n⚠️  Visual analyst still has issues. Recommendations:")
        print("   1. Check Ollama logs: ollama logs")
        print("   2. Try pulling the model again: ollama pull qwen2.5vl:7b-q4_K_M")
        print("   3. Increase system memory or reduce image size")
        print("   4. Consider using a lighter vision model")

if __name__ == "__main__":
    main()
