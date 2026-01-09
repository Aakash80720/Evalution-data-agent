"""Planner agent for creating execution plans."""
import json
import time
from typing import Any, Dict
from langgraph.types import Command
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from prompts import build_plan_prompt
from config import LLMConfig


class PlannerAgent(BaseAgent):
    """Agent responsible for creating and updating execution plans."""
    
    def __init__(self):
        super().__init__("planner")
        config = LLMConfig.get_config("planner")
        self.llm = ChatOpenAI(**config)
    
    def invoke(self, state: Dict[str, Any]) -> Command:
        """Create or update the execution plan."""
        start_time = time.time()
        self.log_entry()
        self.log_state(state)
        
        # Invoke LLM
        self.logger.info("[PLANNER] Invoking LLM with plan_prompt...")
        
        # Get user query safely
        user_query = state.get("user_query")
        if not user_query:
            # Fallback to messages if user_query not provided
            messages = state.get("messages", [])
            if messages:
                user_query = messages[0].content
            else:
                raise ValueError("No user query found in state")
        
        # Get replan reason from supervisor feedback or executor
        replan_reason = ""
        supervisor_feedback = state.get("supervisor_feedback", {})
        if supervisor_feedback:
            replan_reason = supervisor_feedback.get("reason", "")
            # Add issues and suggestions to the reason
            issues = supervisor_feedback.get("issues", [])
            suggestions = supervisor_feedback.get("suggestions", [])
            if issues:
                replan_reason += f"\n\nIssues identified: {', '.join(issues)}"
            if suggestions:
                replan_reason += f"\n\nSuggestions: {', '.join(suggestions)}"
        else:
            replan_reason = state.get("last_reason", "")
        
        # Build the prompt
        prompt = build_plan_prompt(
            user_query=user_query,
            replan_flag=state.get("replan_flag", False),
            prior_plan=state.get("plan"),
            replan_reason=replan_reason,
            enabled_agents=state.get("enabled_agents")
        )
        
        llm_reply = self.llm.invoke([prompt])
        self.logger.info("[PLANNER] LLM response received in %.2f seconds", time.time() - start_time)
        self.logger.info("[PLANNER] LLM reply: %s", llm_reply.content)
        
        print("\n" + "=" * 50)
        print("PLANNER RESPONSE:")
        print("=" * 50)
        print(llm_reply.content)
        print("=" * 50 + "\n")
        
        # Parse the plan
        try:
            content_str = llm_reply.content if isinstance(
                llm_reply.content, str) else str(llm_reply.content)
            parsed_plan = json.loads(content_str)
            self.logger.info("[PLANNER] Successfully parsed plan JSON")
            self.logger.info("[PLANNER] Parsed plan: %s", json.dumps(parsed_plan, indent=2))
        except json.JSONDecodeError as e:
            self.logger.error("[PLANNER] Invalid JSON from planner: %s", llm_reply.content)
            self.logger.error("[PLANNER] JSON decode error: %s", str(e))
            raise ValueError(f"Planner returned invalid JSON:\n{llm_reply.content}")
        
        replan = state.get("replan_flag", False)
        self.logger.info("[PLANNER] Is this a replan? %s", replan)
        
        # Track replan attempts
        replan_attempts = state.get("replan_attempts", {}) or {}
        if replan:
            # Increment global replan counter
            replan_attempts["total"] = replan_attempts.get("total", 0) + 1
            self.logger.info("[PLANNER] Replan attempt #%d", replan_attempts["total"])
        
        # Create command
        command = Command(
            update={
                "plan": parsed_plan,
                "messages": [HumanMessage(
                    content=llm_reply.content,
                    name="replan" if replan else "initial_plan")],
                "user_query": state.get("user_query", state["messages"][0].content),
                "current_step": 1 if not replan else state["current_step"],
                "replan_flag": False,  # Reset replan flag after planning
                "replan_attempts": replan_attempts,  # Track attempts
                "supervisor_feedback": {},  # Clear supervisor feedback
                "supervisor_approved": False,  # Reset approval status
                "enabled_agents": state.get("enabled_agents"),
            },
            goto="supervisor",  # Always route to supervisor for validation
        )
        
        self.log_command(command)
        self.log_exit()
        return command
