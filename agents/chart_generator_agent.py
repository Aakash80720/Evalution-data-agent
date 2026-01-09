"""Chart generator agent for creating visualizations."""
import time
from typing import Any, Dict, Literal
from langgraph.types import Command
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agents.base_agent import BaseAgent
from tools import python_repl_tool
from config import LLMConfig
from prompts import agent_system_prompt, CHART_GENERATOR_PROMPT


class ChartGeneratorAgent(BaseAgent):
    """Agent responsible for generating charts and visualizations."""
    
    def __init__(self):
        super().__init__("chart_generator")
        config = LLMConfig.get_config("chart_generator")
        llm = ChatOpenAI(**config)
        
        self.agent = create_react_agent(
            llm,
            [python_repl_tool],
            prompt=agent_system_prompt(CHART_GENERATOR_PROMPT),
        )
    
    def invoke(self, state: Dict[str, Any]) -> Command[Literal["executor"]]:
        """Generate a chart based on the query."""
        start_time = time.time()
        self.log_entry()
        self.log_state(state)
        
        # Invoke the agent
        self.logger.info("[CHART_GENERATOR] Invoking agent...")
        try:
            result = self.agent.invoke(state)
            self.logger.info("[CHART_GENERATOR] Completed in %.2f seconds", time.time() - start_time)
            self.logger.info("[CHART_GENERATOR] Message count: %d", len(result.get("messages", [])))
            
            if result.get("messages"):
                last_msg = result["messages"][-1]
                self.logger.info("[CHART_GENERATOR] Result: %s", last_msg.content)
                
                print("\n" + "=" * 50)
                print("CHART GENERATOR RESULT:")
                print("=" * 50)
                print(last_msg.content)
                print("=" * 50 + "\n")
                
                # Check if chart was saved
                if "CHART_PATH:" in last_msg.content:
                    chart_path = last_msg.content.split("CHART_PATH:")[1].split("\n")[0].strip()
                    self.logger.info("[CHART_GENERATOR] Chart saved at: %s", chart_path)
                else:
                    self.logger.warning("[CHART_GENERATOR] No CHART_PATH found in output!")
                    
        except Exception as e:
            self.logger.error("[CHART_GENERATOR] Error: %s", str(e))
            raise
        
        # Convert last message to HumanMessage
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content,
            name="chart_generator"
        )
        
        command = Command(
            update={"messages": result["messages"]},
            goto="executor",
        )
        
        self.log_command(command)
        self.log_exit()
        return command
