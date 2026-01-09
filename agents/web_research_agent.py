"""Web research agent for searching the web."""
import time
from typing import Any, Dict, Literal
from langgraph.types import Command
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agents.base_agent import BaseAgent
from tools import web_search_tool
from config import LLMConfig
from prompts import WEB_RESEARCH_PROMPT


class WebResearchAgent(BaseAgent):
    """Agent responsible for web research."""
    
    def __init__(self):
        super().__init__("web_researcher")
        config = LLMConfig.get_config("researcher")
        llm = ChatOpenAI(**config)
        
        self.agent = create_react_agent(
            llm,
            tools=[web_search_tool],
            prompt=WEB_RESEARCH_PROMPT,
        )
    
    def invoke(self, state: Dict[str, Any]) -> Command[Literal["executor"]]:
        """Perform web research."""
        start_time = time.time()
        self.log_entry()
        self.log_state(state)
        
        agent_query = state.get("agent_query")
        self.logger.info("[WEB_RESEARCHER] Query: %s", agent_query)
        
        # Invoke the agent
        self.logger.info("[WEB_RESEARCHER] Invoking agent...")
        try:
            result = self.agent.invoke({"messages": agent_query})
            self.logger.info("[WEB_RESEARCHER] Completed in %.2f seconds", time.time() - start_time)
            self.logger.info("[WEB_RESEARCHER] Message count: %d", len(result.get("messages", [])))
            
            if result.get("messages"):
                last_msg = result["messages"][-1]
                self.logger.info("[WEB_RESEARCHER] Result (truncated): %s", str(last_msg.content)[:500])
                
                print("\n" + "=" * 50)
                print("WEB RESEARCHER RESULT:")
                print("=" * 50)
                print(last_msg.content[:1000] if len(last_msg.content) > 1000 else last_msg.content)
                print("=" * 50 + "\n")
                
        except Exception as e:
            self.logger.error("[WEB_RESEARCHER] Error: %s", str(e))
            raise
        
        # Convert last message to HumanMessage
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, 
            name="web_researcher"
        )
        
        command = Command(
            update={"messages": result["messages"]},
            goto="executor",
        )
        
        self.log_command(command)
        self.log_exit()
        return command
