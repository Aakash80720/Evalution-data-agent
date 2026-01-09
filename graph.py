"""Graph builder for the multi-agent system."""
import logging
from langgraph.graph import StateGraph, START

from state import MessageContext
from agents import (
    PlannerAgent,
    ExecutorAgent,
    WebResearchAgent,
    ChartGeneratorAgent,
    ChartSummarizerAgent,
    SynthesizerAgent,
    SupervisorAgent,
)

logger = logging.getLogger(__name__)


def build_graph():
    """Build and compile the agent graph."""
    logger.info("=" * 80)
    logger.info("BUILDING AGENT GRAPH")
    logger.info("=" * 80)
    
    # Initialize agents
    planner = PlannerAgent()
    supervisor = SupervisorAgent()
    executor = ExecutorAgent()
    web_researcher = WebResearchAgent()
    chart_generator = ChartGeneratorAgent()
    chart_summarizer = ChartSummarizerAgent()
    synthesizer = SynthesizerAgent()
    
    # Build the graph
    flow = StateGraph(MessageContext)
    
    # Add nodes
    flow.add_node("planner", planner.invoke)
    flow.add_node("supervisor", supervisor.invoke)
    flow.add_node("executor", executor.invoke)
    flow.add_node("web_researcher", web_researcher.invoke)
    flow.add_node("chart_generator", chart_generator.invoke)
    flow.add_node("chart_summarizer", chart_summarizer.invoke)
    flow.add_node("synthesizer", synthesizer.invoke)
    
    # Add edges
    # Start -> planner (creates the initial plan)
    flow.add_edge(START, "planner")
    
    # Planner -> supervisor (validates the plan before execution)
    flow.add_edge("planner", "supervisor")
    
    # Supervisor routes to either:
    # - executor (if plan approved)
    # - planner (if replanning needed)
    # The routing is handled by Command returns in supervisor.invoke()
    
    logger.info("Nodes added: planner, supervisor, executor, web_researcher, chart_generator, chart_summarizer, synthesizer")
    logger.info("Edges: START -> planner -> supervisor -> [executor | planner]")
    
    # Compile the graph
    graph = flow.compile()
    logger.info("Graph compiled successfully")
    
    return graph
