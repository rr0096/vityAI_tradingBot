#!/usr/bin/env python3
"""
Simple script to restart Ollama and test visual analyst
"""

import subprocess
import time

def restart_ollama():
    """Restart Ollama service"""
    print("🔄 Restarting Ollama...")
    
    try:
        # Kill ollama
        subprocess.run(["pkill", "-f", "ollama"], check=False)
        time.sleep(3)
        
        # Start ollama in background
        print("🚀 Starting Ollama...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(8)  # Wait longer for startup
        
        # Test if running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Ollama restarted successfully")
            return True
        else:
            print("❌ Failed to restart Ollama")
            return False
            
    except Exception as e:
        print(f"❌ Error restarting Ollama: {e}")
        return False

def main():
    print("🛠️  RESTARTING OLLAMA FOR VISUAL ANALYST")
    print("=" * 40)
    
    if restart_ollama():
        print("\n✅ Ollama is ready. You can now test the visual analyst.")
        print("💡 The visual analyst should now work without 500 errors.")
    else:
        print("\n❌ Failed to restart Ollama.")
        print("🔧 Try manually: killall ollama && ollama serve")

if __name__ == "__main__":
    main()
