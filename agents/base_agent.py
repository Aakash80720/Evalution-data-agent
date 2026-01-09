"""Base agent class for all agents in the system."""
import logging
from typing import Any, Dict, Literal
from abc import ABC, abstractmethod
from langgraph.types import Command
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agents.{name}")
    
    def log_entry(self):
        """Log agent entry."""
        self.logger.info("=" * 80)
        self.logger.info(f"[{self.name.upper()}] ========== ENTER ==========")
        self.logger.info("=" * 80)
    
    def log_exit(self):
        """Log agent exit."""
        self.logger.info("=" * 80)
        self.logger.info(f"[{self.name.upper()}] ========== EXIT ==========")
        self.logger.info("=" * 80)
    
    def log_state(self, state: Dict[str, Any]):
        """Log state summary."""
        self.logger.info(f"[{self.name.upper()}] STATE SUMMARY:")
        self.logger.info(f"  - user_query: {state.get('user_query', 'N/A')}")
        self.logger.info(f"  - current_step: {state.get('current_step', 'N/A')}")
        self.logger.info(f"  - replan_flag: {state.get('replan_flag', False)}")
        self.logger.info(f"  - agent_query: {state.get('agent_query', 'N/A')}")
        self.logger.info(f"  - message_count: {len(state.get('messages', []))}")
    
    def log_command(self, command: Command):
        """Log command output."""
        self.logger.info(f"[{self.name.upper()}] COMMAND OUTPUT:")
        self.logger.info(f"  - goto: {command.goto}")
        if hasattr(command, 'update') and command.update:
            self.logger.info(f"  - update keys: {list(command.update.keys())}")
    
    @abstractmethod
    def invoke(self, state: Dict[str, Any]) -> Command:
        """Execute the agent logic."""
        pass
