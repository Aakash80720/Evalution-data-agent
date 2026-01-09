"""Supervisor agent for monitoring and validating plans."""
import time
from typing import Any, Dict, Literal
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from config import LLMConfig, MAX_REPLANS
from prompts import build_supervisor_prompt


class SupervisorAgent(BaseAgent):
    """Agent responsible for validating plans and detecting inefficiencies."""
    
    def __init__(self):
        super().__init__("supervisor")
        config = LLMConfig.get_config("supervisor")
        self.llm = ChatOpenAI(**config)
    
    def invoke(self, state: Dict[str, Any]) -> Command[Literal['executor', 'planner']]:
        """
        Validate the current plan and decide if replanning is needed.
        
        Returns:
            Command to either proceed to executor or replan
        """
        start_time = time.time()
        self.log_entry()
        self.log_state(state)
        
        plan = state.get("plan", {})
        user_query = state.get("user_query", "")
        current_step = state.get("current_step", 1)
        replan_attempts = state.get("replan_attempts", {})
        enabled_agents = state.get("enabled_agents", [])
        
        # Quick analysis
        web_research_count = sum(1 for step in plan.values() if step.get("agent") == "web_researcher")
        total_replans = replan_attempts.get("total", 0)
        
        self.logger.info("[SUPERVISOR] Plan: %d steps, %d web research, %d replans so far", 
                        len(plan), web_research_count, total_replans)
        
        # Build analysis prompt
        prompt = build_supervisor_prompt(
            user_query=user_query,
            plan=plan,
            enabled_agents=enabled_agents,
            replan_attempts=replan_attempts
        )
        
        # Invoke LLM for plan analysis
        self.logger.info("[SUPERVISOR] Invoking LLM for plan validation...")
        llm_reply = self.llm.invoke([prompt])
        self.logger.info("[SUPERVISOR] LLM response received in %.2f seconds", time.time() - start_time)
        self.logger.info("[SUPERVISOR] LLM reply: %s", llm_reply.content)
        
        print("\n" + "=" * 50)
        print("SUPERVISOR ANALYSIS:")
        print("=" * 50)
        print(llm_reply.content)
        print("=" * 50 + "\n")
        
        # Parse supervisor decision
        try:
            import json
            content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
            parsed = json.loads(content_str)
            
            needs_replan = parsed.get("needs_replan", False)
            reason = parsed.get("reason", "")
            issues = parsed.get("issues", [])
            suggestions = parsed.get("suggestions", [])
            
            self.logger.info("[SUPERVISOR] Needs replan: %s", needs_replan)
            self.logger.info("[SUPERVISOR] Reason: %s", reason)
            self.logger.info("[SUPERVISOR] Issues found: %s", issues)
            self.logger.info("[SUPERVISOR] Suggestions: %s", suggestions)
            
        except Exception as exc:
            self.logger.error("[SUPERVISOR] Invalid JSON: %s", llm_reply.content)
            self.logger.warning("[SUPERVISOR] Defaulting to proceed with plan")
            needs_replan = False
            reason = "Could not parse supervisor response"
        
        # Check if we've exceeded max replans - FORCE APPROVE to prevent infinite loops
        replan_attempts = state.get("replan_attempts", {}) or {}
        total_replans = replan_attempts.get("total", 0)
        
        if total_replans >= 2:
            self.logger.warning("[SUPERVISOR] Max replans reached (%d). FORCE APPROVING to prevent loop.", total_replans)
            needs_replan = False
            reason = "Force approved after 2 replan attempts"
        
        # Build command
        if needs_replan:
            self.logger.info("[SUPERVISOR] Triggering replan due to: %s", reason)
            command = Command(
                update={
                    "messages": [HumanMessage(content=llm_reply.content, name="supervisor")],
                    "replan_flag": True,
                    "supervisor_feedback": {
                        "reason": reason,
                        "issues": issues,
                        "suggestions": suggestions
                    }
                },
                goto="planner",
            )
        else:
            self.logger.info("[SUPERVISOR] Plan approved. Proceeding to executor.")
            command = Command(
                update={
                    "messages": [HumanMessage(content=llm_reply.content, name="supervisor")],
                    "supervisor_approved": True,
                    "supervisor_feedback": {
                        "reason": "Plan approved - efficient and well-structured",
                        "issues": [],
                        "suggestions": []
                    }
                },
                goto="executor",
            )
        
        self.log_command(command)
        self.log_exit()
        return command
