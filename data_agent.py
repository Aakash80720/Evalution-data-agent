from langgraph.graph import MessagesState
from typing import Optional, List, Dict, Any

class MessageContext(MessagesState):
    user_query: Optional[str]
    enabled_agents: Optional[List[str]]
    plan: Optional[List[Dict[int, Dict[str, Any]]]]
    current_step: int
    agent_query: Optional[str]
    last_reason: Optional[str]
    replan_flag: Optional[bool]
    replan_attempts: Optional[Dict[int, Dict[int, int]]]

