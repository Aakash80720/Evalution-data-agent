"""Chart summarizer agent for creating chart descriptions."""
import time
from typing import Any, Dict, Literal
from langgraph.types import Command
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agents.base_agent import BaseAgent
from config import LLMConfig
from prompts import agent_system_prompt, CHART_SUMMARIZER_PROMPT


class ChartSummarizerAgent(BaseAgent):
    """Agent responsible for summarizing charts."""
    
    def __init__(self):
        super().__init__("chart_summarizer")
        config = LLMConfig.get_config("chart_summarizer")
        llm = ChatOpenAI(**config)
        
        self.agent = create_react_agent(
            llm,
            tools=[],
            prompt=agent_system_prompt(CHART_SUMMARIZER_PROMPT),
        )
    
    def invoke(self, state: Dict[str, Any]) -> Command[Literal["executor"]]:
        """Summarize the generated chart."""
        start_time = time.time()
        self.log_entry()
        self.log_state(state)
        
        # Extract only the chart generator's message
        chart_generator_msg = None
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "name") and msg.name == "chart_generator":
                chart_generator_msg = msg.content
                break
        
        if not chart_generator_msg:
            self.logger.warning("[CHART_SUMMARIZER] No chart generator message found!")
            chart_generator_msg = "No chart information available."
        
        # Create a minimal state with only the chart info
        minimal_state = {
            "messages": [HumanMessage(content=chart_generator_msg, name="chart_generator")]
        }
        
        # Invoke the agent with minimal context
        self.logger.info("[CHART_SUMMARIZER] Invoking agent with minimal state...")
        try:
            result = self.agent.invoke(minimal_state)
            self.logger.info("[CHART_SUMMARIZER] Completed in %.2f seconds", time.time() - start_time)
            
            if result.get("messages"):
                last_msg = result["messages"][-1]
                self.logger.info("[CHART_SUMMARIZER] Summary: %s", last_msg.content)
                
                print("\n" + "=" * 50)
                print("CHART SUMMARIZER RESULT:")
                print("=" * 50)
                print(last_msg.content)
                print("=" * 50 + "\n")
                
        except Exception as e:
            self.logger.error("[CHART_SUMMARIZER] Error: %s", str(e))
            raise
        
        # Get the summary result
        summary_result = result["messages"][-1].content
        
        # Add the summary to the state messages
        state["messages"].append(HumanMessage(
            content=summary_result,
            name="chart_summarizer"
        ))
        
        # Store in agent_outputs for reliable synthesis
        agent_outputs = state.get("agent_outputs", {}) or {}
        agent_outputs["chart_summarizer"] = summary_result
        
        command = Command(
            update={
                "messages": state["messages"],
                "agent_outputs": agent_outputs,
            },
            goto="executor",
        )
        
        self.log_command(command)
        self.log_exit()
        return command
