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
        
        # Gather relevant messages - limit to avoid token overflow
        relevant_msgs = []
        for m in state.get("messages", []):
            name = getattr(m, "name", None)
            if name in ("web_researcher", "chart_generator", "chart_summarizer"):
                # Truncate very long messages to avoid hitting token limits
                content = m.content
                if len(content) > 2000:
                    content = content[:2000] + "\n...[truncated for brevity]"
                relevant_msgs.append(f"[{name}]: {content}")
        
        # Limit to last 5 messages to avoid token overflow
        if len(relevant_msgs) > 5:
            self.logger.info("[SYNTHESIZER] Limiting to last 5 messages (had %d)", len(relevant_msgs))
            relevant_msgs = relevant_msgs[-5:]
        
        self.logger.info("[SYNTHESIZER] Found %d relevant messages", len(relevant_msgs))
        for i, msg in enumerate(relevant_msgs):
            self.logger.info("[SYNTHESIZER] Message %d (truncated): %s", i+1, msg[:200])
        
        user_question = state.get("user_query", 
                                 state.get("messages", [{}])[0].content if state.get("messages") else "")
        self.logger.info("[SYNTHESIZER] User question: %s", user_question)
        
        summary_prompt = [
            HumanMessage(content=(
                f"User question: {user_question}\n\n"
                f"{SYNTHESIZER_INSTRUCTIONS}\n\n"
                f"Context (from agents):\n\n" + "\n\n---\n\n".join(relevant_msgs)
            ))
        ]
        
        # Invoke LLM with error handling
        self.logger.info("[SYNTHESIZER] Invoking LLM...")
        try:
            llm_reply = self.llm.invoke(summary_prompt)
            answer = llm_reply.content.strip()
            
            self.logger.info("[SYNTHESIZER] Completed in %.2f seconds", time.time() - start_time)
            self.logger.info("[SYNTHESIZER] Final answer: %s", answer)
            
            print("\n" + "=" * 50)
            print("SYNTHESIZER FINAL ANSWER:")
            print("=" * 50)
            print(answer)
            print("=" * 50 + "\n")
        except Exception as e:
            self.logger.error("[SYNTHESIZER] Error during synthesis: %s", str(e))
            answer = f"Error generating final answer: {str(e)}\n\nContext was based on {len(relevant_msgs)} agent messages."
        
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
