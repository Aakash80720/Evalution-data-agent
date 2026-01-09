"""Executor agent for routing to appropriate agents."""
import json
import time
from typing import Any, Dict, Literal
from langgraph.types import Command
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from prompts import build_executor_prompt
from config import LLMConfig, MAX_REPLANS


class ExecutorAgent(BaseAgent):
    """Agent responsible for executing the plan and routing to other agents."""
    
    def __init__(self):
        super().__init__("executor")
        config = LLMConfig.get_config("executor")
        self.llm = ChatOpenAI(**config)
    
    def invoke(self, state: Dict[str, Any]) -> Command[Literal['planner', 'web_researcher', 'chart_generator', 'chart_summarizer', 'synthesizer']]:
        """Execute the plan and route to the next agent."""
        start_time = time.time()
        self.log_entry()
        self.log_state(state)
        
        plan: Dict[str, Any] = state.get("plan", {})
        step: int = state.get("current_step", 1)
        
        self.logger.info("[EXECUTOR] Current plan: %s", json.dumps(plan, indent=2))
        self.logger.info("[EXECUTOR] Current step: %d", step)
        self.logger.info("[EXECUTOR] Replan flag: %s", state.get("replan_flag"))
        
        # Check if replan flag is set
        if state.get("replan_flag"):
            planned_agent = plan.get(str(step), {}).get("agent")
            self.logger.info("[EXECUTOR] REPLAN MODE - Routing to planned agent: %s", planned_agent)
            
            command = Command(
                update={
                    "replan_flag": False,
                    "current_step": step + 1,
                },
                goto=planned_agent,
            )
            self.log_command(command)
            self.log_exit()
            return command
        
        # Normal execution - invoke LLM
        self.logger.info("[EXECUTOR] NORMAL MODE - Building executor prompt...")
        
        # Build the prompt
        prompt = build_executor_prompt(
            user_query=state.get("user_query", state.get("messages", [{}])[0].content if state.get("messages") else ""),
            current_step=step,
            plan=plan,
            replan_flag=state.get("replan_flag", False),
            replan_attempts=state.get("replan_attempts", {}),
            recent_messages=state.get("messages", []),
            enabled_agents=state.get("enabled_agents")
        )
        
        llm_reply = self.llm.invoke([prompt])
        self.logger.info("[EXECUTOR] LLM response received in %.2f seconds", time.time() - start_time)
        self.logger.info("[EXECUTOR] LLM reply: %s", llm_reply.content)
        
        print("\n" + "=" * 50)
        print("EXECUTOR RESPONSE:")
        print("=" * 50)
        print(llm_reply.content)
        print("=" * 50 + "\n")
        
        # Parse executor decision
        try:
            content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
            parsed = json.loads(content_str)
            replan: bool = parsed["replan"]
            goto: str = parsed["goto"]
            reason: str = parsed["reason"]
            query: str = parsed["query"]
            
            self.logger.info("[EXECUTOR] Parsed decision:")
            self.logger.info("  - replan: %s", replan)
            self.logger.info("  - goto: %s", goto)
            self.logger.info("  - reason: %s", reason)
            self.logger.info("  - query: %s", query)
            
        except Exception as exc:
            self.logger.error("[EXECUTOR] Invalid JSON: %s", llm_reply.content)
            raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from exc
        
        # Prepare updates
        updates: Dict[str, Any] = {
            "messages": [HumanMessage(content=llm_reply.content, name="executor")],
            "last_reason": reason,
            "agent_query": query,
        }
        
        # Handle replan logic
        replans: Dict[int, int] = state.get("replan_attempts", {}) or {}
        step_replans = replans.get(step, 0)
        
        self.logger.info("[EXECUTOR] Replan tracking:")
        self.logger.info("  - All replans: %s", replans)
        self.logger.info("  - Step %d replans: %d (max: %d)", step, step_replans, MAX_REPLANS)
        
        if replan:
            self.logger.info("[EXECUTOR] REPLAN REQUESTED")
            if step_replans < MAX_REPLANS:
                replans[step] = step_replans + 1
                updates.update({
                    "replan_attempts": replans,
                    "replan_flag": True,
                    "current_step": step,
                })
                self.logger.info("[EXECUTOR] Replanning allowed. Updated replans: %s", replans)
                command = Command(update=updates, goto="planner")
            else:
                next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer")
                updates["current_step"] = step + 1
                self.logger.warning("[EXECUTOR] MAX REPLANS REACHED. Moving to: %s", next_agent)
                command = Command(update=updates, goto=next_agent)
        else:
            planned_agent = plan.get(str(step), {}).get("agent")
            self.logger.info("[EXECUTOR] HAPPY PATH")
            self.logger.info("  - Planned agent: %s", planned_agent)
            self.logger.info("  - Goto: %s", goto)
            
            if goto == planned_agent:
                updates["current_step"] = step + 1
                self.logger.info("[EXECUTOR] Advancing to step %d", step + 1)
            else:
                self.logger.warning("[EXECUTOR] Goto mismatch. Keeping step at %d", step)
            
            updates["replan_flag"] = False
            command = Command(update=updates, goto=goto)
        
        self.log_command(command)
        self.log_exit()
        return command
