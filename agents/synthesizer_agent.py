"""Synthesizer agent for creating final answers."""
import time
from typing import Any, Dict, Literal
from langgraph.types import Command
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from agents.base_agent import BaseAgent
from config import LLMConfig
from prompts import SYNTHESIZER_INSTRUCTIONS


class SynthesizerAgent(BaseAgent):
    """Agent responsible for synthesizing final answers."""
    
    def __init__(self):
        super().__init__("synthesizer")
        config = LLMConfig.get_config("synthesizer")
        self.llm = ChatOpenAI(**config)
    
    def invoke(self, state: Dict[str, Any]) -> Command[Literal["__end__"]]:
        """Create a concise final answer from all agent outputs."""
        start_time = time.time()
        self.log_entry()
        self.log_state(state)
        
        # Get agent outputs directly from state dictionary
        agent_outputs = state.get("agent_outputs", {}) or {}
        
        # Build context from agent outputs
        context_parts = []
        for agent_name in ["web_researcher", "chart_generator", "chart_summarizer"]:
            if agent_name in agent_outputs and agent_outputs[agent_name]:
                context_parts.append(f"[{agent_name}]:\n{agent_outputs[agent_name]}")
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No agent data available."
        
        self.logger.info("[SYNTHESIZER] Using %d agent outputs", len(context_parts))
        
        user_question = state.get("user_query", "")
        self.logger.info("[SYNTHESIZER] User question: %s", user_question)
        
        summary_prompt = [
            HumanMessage(content=(
                f"User question: {user_question}\n\n"
                f"{SYNTHESIZER_INSTRUCTIONS}\n\n"
                f"Context (from agents):\n\n{context}"
            ))
        ]
        
        # Invoke LLM
        self.logger.info("[SYNTHESIZER] Invoking LLM...")
        try:
            llm_reply = self.llm.invoke(summary_prompt)
            answer = llm_reply.content.strip()
            
            self.logger.info("[SYNTHESIZER] Completed in %.2f seconds", time.time() - start_time)
            
            print("\n" + "=" * 50)
            print("SYNTHESIZER FINAL ANSWER:")
            print("=" * 50)
            print(answer)
            print("=" * 50 + "\n")
        except Exception as e:
            self.logger.error("[SYNTHESIZER] Error during synthesis: %s", str(e))
            answer = f"Error generating final answer: {str(e)}"
        
        command = Command(
            update={
                "final_answer": answer,
                "messages": [HumanMessage(content=answer, name="synthesizer")],
            },
            goto=END,
        )
        
        self.log_command(command)
        self.log_exit()
        return command
