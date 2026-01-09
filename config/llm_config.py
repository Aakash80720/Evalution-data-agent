"""LLM configuration settings."""
from typing import Dict, Any


class LLMConfig:
    """Configuration for different LLM models used in the system."""
    
    # Planner and Executor LLM - requires structured JSON output
    PLANNER_EXECUTOR_CONFIG = {
        "model": "gpt-4o",
        "temperature": 0.7,
        "model_kwargs": {
            "response_format": {"type": "json_object"}
        }
    }
    
    # Research Agent LLM - for web research tasks
    RESEARCHER_CONFIG = {
        "model": "gpt-4o",
        "temperature": 0.5,
    }
    
    # Chart Generator LLM - for code generation
    CHART_GENERATOR_CONFIG = {
        "model": "gpt-4o",
        "temperature": 0.3,  # Lower temperature for more deterministic code
    }
    
    # Chart Summarizer LLM - for image/chart description
    CHART_SUMMARIZER_CONFIG = {
        "model": "gpt-4o",
        "temperature": 0.7,
    }
    
    # Synthesizer LLM - for final answer synthesis
    SYNTHESIZER_CONFIG = {
        "model": "gpt-4o",
        "temperature": 0.6,
    }
    
    # Supervisor LLM - for plan validation and monitoring
    SUPERVISOR_CONFIG = {
        "model": "gpt-4o",
        "temperature": 0.5,
        "model_kwargs": {
            "response_format": {"type": "json_object"}
        }
    }
    
    @classmethod
    def get_config(cls, agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type.
        
        Args:
            agent_type: One of 'planner', 'executor', 'researcher', 
                       'chart_generator', 'chart_summarizer', 'synthesizer'
        
        Returns:
            Configuration dictionary for the specified agent
        """
        config_map = {
            "planner": cls.PLANNER_EXECUTOR_CONFIG,
            "executor": cls.PLANNER_EXECUTOR_CONFIG,
            "researcher": cls.RESEARCHER_CONFIG,
            "chart_generator": cls.CHART_GENERATOR_CONFIG,
            "chart_summarizer": cls.CHART_SUMMARIZER_CONFIG,
            "synthesizer": cls.SYNTHESIZER_CONFIG,
            "supervisor": cls.SUPERVISOR_CONFIG,
        }
        
        if agent_type not in config_map:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return config_map[agent_type].copy()


# Maximum number of replans allowed per step
MAX_REPLANS = 2

# Enabled agents in the system
# Note: chart_summarizer is optional and can be removed to go directly from chart_generator to synthesizer
ENABLED_AGENTS = [
    "web_researcher",
    "chart_generator",
    # "chart_summarizer",  # Optional: Comment out to skip and go directly to synthesizer
    "synthesizer"
]
