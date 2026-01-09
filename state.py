"""State definitions for the agent system."""
from typing import Optional, List, Dict, Any, Annotated
from langgraph.graph import MessagesState, add_messages


class MessageContext(MessagesState):
    """Extended state for multi-agent communication."""
    user_query: Optional[str]
    enabled_agents: Optional[List[str]]
    plan: Optional[List[Dict[int, Dict[str, Any]]]]
    current_step: int
    agent_query: Optional[str]
    last_reason: Optional[str]
    replan_flag: Optional[bool]
    replan_attempts: Optional[Dict[int, Dict[int, int]]]
    supervisor_approved: Optional[bool]
    supervisor_feedback: Optional[Dict[str, Any]]
    final_answer: Optional[str]
