# Ollama Configuration Guide for FenixTradingBot

This document provides comprehensive instructions for setting up and using Ollama with FenixTradingBot, ensuring optimal performance with local LLM models.

## Table of Contents
- [What is Ollama?](#what-is-ollama)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Configuration](#configuration)
- [Model Recommendations](#model-recommendations)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## What is Ollama?

Ollama is a tool that allows you to run large language models locally on your machine. FenixTradingBot is designed to work exclusively with Ollama, providing:

- **Privacy**: All data stays on your machine
- **Cost Efficiency**: No API costs for model usage
- **Reliability**: No dependency on external services
- **Customization**: Full control over model selection and parameters

## Installation

### 1. Install Ollama

#### macOS
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
Download and install from [https://ollama.ai](https://ollama.ai)

### 2. Verify Installation
```bash
ollama --version
```

### 3. Start Ollama Service
```bash
ollama serve
```

The service will run on `http://localhost:11434` by default.

## Model Setup

FenixTradingBot requires specific models for different agent types. Follow these steps to install the recommended models:

### Pull Required Models

```bash
# Sentiment Analysis Agent
ollama pull qwen2.5:0.5b
ollama pull qwen3:4b

# Technical Analysis Agent  
ollama pull granite3.3:2b
ollama pull cogito:3b

# Visual Analysis Agent (for chart analysis)
ollama pull gemma3:4b
ollama pull granite3.2-vision:latest
ollama pull qwen2.5vl:3b

# QABBA Validator Agent
ollama pull phi4-mini:3.8b
ollama pull phi4:latest

# Decision Making Agent
ollama pull phi4-mini-reasoning:latest
ollama pull mychen76/Fin-R1:Q6
ollama pull cogito:8b

# General purpose fallback
ollama pull llama3.2:1b
```

### Verify Models are Available
```bash
ollama list
```

You should see all the models you've installed.

## Configuration

### 1. Environment Variables (Optional)

You can customize Ollama settings via environment variables:

```bash
# ~/.bashrc or ~/.zshrc
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_API_KEY="ollama"  # Default value, can be anything
export OLLAMA_NUM_PARALLEL=1    # Number of parallel requests
export OLLAMA_MAX_LOADED_MODELS=1  # Memory management
```

### 2. Model Configuration

The bot automatically detects available models. You can check which models are configured by running:

```python
from config.modern_models import print_model_availability_guide
print_model_availability_guide()
```

### 3. Memory Settings

For better performance, configure Ollama memory settings:

```bash
# Set memory limits (adjust based on your system)
export OLLAMA_MAX_VRAM=8GB
export OLLAMA_MAX_RAM=16GB
```

## Model Recommendations

### Minimum System Requirements

| Agent Type | Recommended Model | Min RAM | Min VRAM | Performance |
|------------|-------------------|---------|----------|-------------|
| Sentiment | qwen2.5:0.5b | 2GB | 1GB | Fast |
| Technical | granite3.3:2b | 4GB | 2GB | Good |
| Visual | gemma3:4b | 6GB | 3GB | Excellent |
| QABBA | phi4-mini:3.8b | 4GB | 2GB | Good |
| Decision | phi4-mini-reasoning | 4GB | 2GB | Excellent |

### Performance vs Quality Trade-offs

#### Fast Setup (Lower Resource Usage)
```bash
ollama pull qwen2.5:0.5b    # Sentiment
ollama pull cogito:3b       # Technical  
ollama pull gemma3:4b       # Visual
ollama pull phi4-mini:3.8b  # QABBA
ollama pull llama3.2:1b     # Decision
```

#### Balanced Setup (Recommended)
```bash
ollama pull qwen3:4b              # Sentiment
ollama pull granite3.3:2b         # Technical
ollama pull gemma3:4b             # Visual
ollama pull phi4-mini:3.8b        # QABBA
ollama pull phi4-mini-reasoning   # Decision
```

#### High-Quality Setup (More Resources)
```bash
ollama pull qwen3:4b                # Sentiment
ollama pull granite3.3:2b           # Technical
ollama pull granite3.2-vision       # Visual
ollama pull phi4:latest             # QABBA
ollama pull mychen76/Fin-R1:Q6     # Decision
```

## Performance Optimization

### 1. GPU Acceleration

If you have a compatible GPU:

```bash
# Check GPU support
nvidia-smi  # For NVIDIA GPUs
# or
rocm-smi    # For AMD GPUs

# Ollama will automatically use GPU if available
```

### 2. Memory Management

```bash
# Limit concurrent models to save memory
export OLLAMA_MAX_LOADED_MODELS=2

# Adjust based on your system
export OLLAMA_FLASH_ATTENTION=1  # Enable flash attention for speed
```

### 3. Model Quantization

Use quantized models for better performance:

```bash
# These are typically smaller and faster
ollama pull qwen2.5:0.5b      # Already quantized
ollama pull phi4-mini:3.8b    # Optimized for speed
```

### 4. System Optimization

#### For macOS:
```bash
# Increase file descriptor limits
ulimit -n 4096
```

#### For Linux:
```bash
# Add to /etc/security/limits.conf
* soft nofile 4096
* hard nofile 8192
```

## Configuration Validation

### Test Your Setup

1. **Check Ollama Status**:
```bash
curl http://localhost:11434/api/tags
```

2. **Test Model Loading**:
```bash
ollama run qwen2.5:0.5b "Hello, test message"
```

3. **Run Bot Model Check**:
```python
python -c "from config.modern_models import model_manager; print('Models available:', model_manager.available_ollama_models)"
```

### Model Health Check

FenixTradingBot includes built-in model health monitoring. Check the logs for:

```
[INFO] ModelManager initialized. Available models: ['qwen2.5:0.5b', 'granite3.3:2b', ...]
[INFO] Using primary model for sentiment: qwen3:4b
[INFO] Using primary model for technical: granite3.3:2b
```

## Troubleshooting

### Common Issues

#### 1. "Model not found" Error
```bash
# Verify model is installed
ollama list

# If not listed, install it
ollama pull <model-name>
```

#### 2. Ollama Service Not Running
```bash
# Check if service is running
curl http://localhost:11434/api/tags

# If not responding, start service
ollama serve
```

#### 3. Out of Memory Errors
```bash
# Reduce concurrent models
export OLLAMA_MAX_LOADED_MODELS=1

# Use smaller models
ollama pull qwen2.5:0.5b  # Instead of larger variants
```

#### 4. Slow Performance
```bash
# Enable GPU if available
# Check: nvidia-smi or rocm-smi

# Use quantized models
ollama pull phi4-mini:3.8b  # Instead of phi4:latest

# Increase timeout if needed
export OLLAMA_REQUEST_TIMEOUT=120
```

#### 5. Connection Issues
```bash
# Check firewall settings
# Ensure port 11434 is open

# Try different host
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

### Advanced Troubleshooting

#### Check Model Compatibility
```python
# In Python
from config.modern_models import model_manager

# Check if specific model is available
print(model_manager._is_model_explicitly_available('qwen3:4b'))

# Get effective model for an agent
config = model_manager.get_model_config('sentiment')
print(f"Effective model: {config.name}")
```

#### Debug Model Loading
```bash
# Enable verbose logging
export OLLAMA_DEBUG=1
ollama serve

# Check model details
ollama show qwen3:4b
```

### Performance Monitoring

The bot includes built-in metrics for model performance:

```python
# Check model health
from config.modern_models import model_manager
print(model_manager.model_health)
```

## Integration Examples

### Basic Usage in Code

```python
from agents.enhanced_base_llm_agent import EnhancedBaseLLMAgent

# The agent automatically uses Ollama
agent = MyTradingAgent(agent_type="sentiment")
result = agent.analyze("Market sentiment for BTCUSDT")
```

### Custom Model Selection

```python
# Override default model selection
from config.modern_models import model_manager

# Force specific model for testing
config = model_manager.get_model_config('sentiment')
print(f"Using model: {config.name}")
```

## Security Considerations

### Network Security
- Ollama runs locally on port 11434
- No external API calls are made
- All data processing happens on your machine

### Data Privacy
- No trading data is sent to external services
- Chat logs remain local
- Model interactions are private

### Access Control
```bash
# Bind to localhost only (default)
export OLLAMA_HOST=127.0.0.1:11434

# For multi-machine setups (advanced)
export OLLAMA_HOST=0.0.0.0:11434  # Use with caution
```

## Support and Resources

### Official Resources
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama Models Library](https://ollama.ai/library)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs)

### Community
- [Ollama Discord](https://discord.gg/ollama)
- [Reddit r/ollama](https://reddit.com/r/ollama)

### FenixTradingBot Specific
- Check `logs/fenix_live_trading.log` for model-specific errors
- Use `python -m config.modern_models` to see model availability
- Monitor metrics via the dashboard for model performance

---

**Note**: This guide assumes you're running FenixTradingBot on a machine with sufficient resources. Adjust model selection based on your hardware capabilities.
