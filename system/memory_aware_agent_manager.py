# system/memory_aware_agent_manager.py
"""
Memory-Aware Agent Manager
Integrates DynamicMemoryManager with existing agents
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

from .dynamic_memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

class MemoryAwareAgentManager:
    """
    Manages agent execution with automatic memory management
    Ensures models are loaded/unloaded as needed
    """
    
    def __init__(self):
        self.memory_manager = get_memory_manager()
        
        # Agent to model mapping from our heterogeneous config
        self.agent_model_mapping = {
            'sentiment': 'qwen2.5:7b-instruct-q5_k_m',
            'technical': 'deepseek-r1:7b-qwen-distill-q4_K_M', 
            'visual': 'qwen2.5vl:7b-q4_K_M',
            'qabba': 'adrienbrault/nous-hermes2pro-llama3-8b:q4_K_M',
            'decision': 'qwen2.5:7b-instruct-q5_k_m'  # Same as sentiment but different params
        }
        
        logger.info("MemoryAwareAgentManager initialized")
        logger.info(f"Managing {len(self.agent_model_mapping)} agent types")
    
    def prepare_agent(self, agent_type: str) -> bool:
        """
        Prepare an agent for execution by ensuring its model is loaded
        
        Args:
            agent_type: Type of agent ('sentiment', 'technical', etc.)
            
        Returns:
            bool: True if agent is ready for execution
        """
        if agent_type not in self.agent_model_mapping:
            logger.error(f"Unknown agent type: {agent_type}")
            return False
            
        model_name = self.agent_model_mapping[agent_type]
        return self.memory_manager.load_model(model_name, agent_type)
    
    def get_agent_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all agent models
        
        Returns:
            Dict with agent status information
        """
        status = {}
        for agent_type, model_name in self.agent_model_mapping.items():
            status[agent_type] = {
                'model': model_name,
                'loaded': model_name in self.memory_manager.loaded_models,
                'can_load': self.memory_manager.can_load_model(model_name)
            }
        return status
    
    def cleanup_agents(self):
        """Clean up old models to free memory"""
        self.memory_manager.cleanup_old_models()
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status including memory and agent information
        """
        memory_report = self.memory_manager.get_comprehensive_report()
        agent_status = self.get_agent_model_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "memory": memory_report,
            "agents": agent_status,
            "system_pressure": self.memory_manager.is_memory_pressure(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> list:
        """Generate system recommendations"""
        recommendations = []
        
        if self.memory_manager.is_memory_pressure():
            recommendations.append("System under memory pressure - consider unloading unused models")
        
        loaded_count = len(self.memory_manager.loaded_models)
        if loaded_count == 0:
            recommendations.append("No models loaded - system ready for agent operations")
        elif loaded_count > 3:
            recommendations.append("Many models loaded - monitor memory usage")
            
        return recommendations
