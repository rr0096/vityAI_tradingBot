# system/dynamic_memory_manager.py
"""
Dynamic Memory Manager for FenixTradingBot
Optimized for Mac M4 16GB - Manages LLM model loading/unloading automatically
"""

import logging
import time
import psutil
import subprocess
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock, Thread
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    agent_type: str
    memory_usage_mb: float
    last_used: datetime
    load_time: datetime
    usage_count: int = 0
    priority: int = 5  # 1 (highest) to 10 (lowest)
    
    def update_usage(self):
        """Update usage statistics"""
        self.last_used = datetime.now()
        self.usage_count += 1

@dataclass 
class MemoryStats:
    """System memory statistics"""
    total_mb: float
    available_mb: float
    used_mb: float
    usage_percent: float
    ollama_usage_mb: float = 0.0
    
    @classmethod
    def get_current(cls) -> 'MemoryStats':
        """Get current system memory stats"""
        memory = psutil.virtual_memory()
        total_mb = memory.total / (1024 * 1024)
        available_mb = memory.available / (1024 * 1024)
        used_mb = memory.used / (1024 * 1024)
        usage_percent = memory.percent
        
        # Try to get Ollama specific memory usage
        ollama_mb = 0.0
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                if 'ollama' in proc.info['name'].lower():
                    ollama_mb += proc.info['memory_info'].rss / (1024 * 1024)
        except:
            pass
            
        return cls(
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=used_mb,
            usage_percent=usage_percent,
            ollama_usage_mb=ollama_mb
        )

class DynamicMemoryManager:
    """
    Dynamic Memory Manager for Mac M4 optimized LLM operations
    """
    
    def __init__(self, max_memory_gb: float = 12.0, pressure_threshold: float = 0.8):
        self.max_memory_mb = max_memory_gb * 1024
        self.pressure_threshold = pressure_threshold
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.model_timeout_minutes = 5
        self.lock = Lock()
        
        # Model memory estimates (in MB)
        self.model_memory_estimates = {
            'qwen2.5:7b-instruct-q5_k_m': 5500,
            'deepseek-r1:7b-qwen-distill-q4_K_M': 4000,
            'qwen2.5vl:7b-q4_K_M': 4000,
            'adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M': 4500
        }
        
        logger.info("DynamicMemoryManager initialized:")
        logger.info(f"  Max memory: {int(self.max_memory_mb)}MB")
        logger.info(f"  Pressure threshold: {self.pressure_threshold*100}%")
        logger.info(f"  Model timeout: {self.model_timeout_minutes} minutes")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current system memory statistics"""
        return MemoryStats.get_current()
    
    def estimate_model_memory(self, model_name: str) -> float:
        """Estimate memory usage for a model in MB"""
        return self.model_memory_estimates.get(model_name, 3000)  # Default 3GB
    
    def can_load_model(self, model_name: str) -> bool:
        """Check if there's enough memory to load a model"""
        stats = self.get_memory_stats()
        estimated_mb = self.estimate_model_memory(model_name)
        
        # Conservative check: ensure we have enough available memory
        return stats.available_mb > (estimated_mb + 1000)  # +1GB buffer
    
    def is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        stats = self.get_memory_stats()
        return stats.usage_percent > (self.pressure_threshold * 100)
    
    def load_model(self, model_name: str, agent_type: str = "unknown") -> bool:
        """
        Simulate loading a model (in real implementation would use Ollama API)
        """
        with self.lock:
            if model_name in self.loaded_models:
                self.loaded_models[model_name].update_usage()
                return True
            
            if not self.can_load_model(model_name):
                logger.warning(f"Insufficient memory to load {model_name}")
                return False
            
            # Simulate model loading
            estimated_mb = self.estimate_model_memory(model_name)
            model_info = ModelInfo(
                name=model_name,
                agent_type=agent_type,
                memory_usage_mb=estimated_mb,
                last_used=datetime.now(),
                load_time=datetime.now()
            )
            
            self.loaded_models[model_name] = model_info
            logger.info(f"Model {model_name} loaded ({estimated_mb}MB)")
            return True
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        with self.lock:
            if model_name not in self.loaded_models:
                return True
            
            del self.loaded_models[model_name]
            logger.info(f"Model {model_name} unloaded")
            return True
    
    def cleanup_old_models(self):
        """Remove models that haven't been used recently"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=self.model_timeout_minutes)
            models_to_remove = []
            
            for model_name, info in self.loaded_models.items():
                if info.last_used < cutoff_time and info.priority > 1:
                    models_to_remove.append(model_name)
            
            for model_name in models_to_remove:
                self.unload_model(model_name)
    
    def get_comprehensive_report(self) -> Dict[str, any]:
        """Generate comprehensive memory and model status report"""
        stats = self.get_memory_stats()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_memory": {
                "total_mb": int(stats.total_mb),
                "available_mb": int(stats.available_mb),
                "used_mb": int(stats.used_mb),
                "usage_percent": round(stats.usage_percent, 1),
                "ollama_mb": int(stats.ollama_usage_mb)
            },
            "memory_management": {
                "max_allowed_mb": int(self.max_memory_mb),
                "pressure_threshold_percent": int(self.pressure_threshold * 100),
                "is_under_pressure": self.is_memory_pressure()
            },
            "loaded_models": [
                {
                    "name": info.name,
                    "agent_type": info.agent_type,
                    "memory_mb": int(info.memory_usage_mb),
                    "usage_count": info.usage_count,
                    "last_used": info.last_used.isoformat(),
                    "priority": info.priority
                }
                for info in self.loaded_models.values()
            ],
            "model_estimates": self.model_memory_estimates,
            "recommendations": []
        }
        
        if self.is_memory_pressure():
            report["recommendations"].append("Consider unloading low-priority models")
        
        if len(self.loaded_models) == 0:
            report["recommendations"].append("No models currently loaded")
            
        return report

# Global instance for singleton pattern
_memory_manager_instance = None
_memory_manager_lock = Lock()

def get_memory_manager() -> DynamicMemoryManager:
    """Get global memory manager instance (singleton)"""
    global _memory_manager_instance
    
    if _memory_manager_instance is None:
        with _memory_manager_lock:
            if _memory_manager_instance is None:
                _memory_manager_instance = DynamicMemoryManager()
    
    return _memory_manager_instance
