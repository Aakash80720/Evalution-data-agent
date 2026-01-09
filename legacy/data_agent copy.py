from typing import Optional, List, Dict, Any, Literal
from legacy.prompt import plan_prompt, executor_prompt, MAX_REPLANS, agent_system_prompt
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, MessagesState, END
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from tools import python_repl_tool, web_search_tool
from langchain.tools import StructuredTool
import json
from pydantic import BaseModel, Field
import logging


# Load environment variables from .env file
from dotenv import load_dotenv
_ = load_dotenv(override=True)

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MessageContext(MessagesState):
    user_query: Optional[str]
    enabled_agents: Optional[List[str]]
    plan: Optional[List[Dict[int, Dict[str, Any]]]]
    current_step: int
    agent_query: Optional[str]
    last_reason: Optional[str]
    replan_flag: Optional[bool]
    replan_attempts: Optional[Dict[int, Dict[int, int]]]

llm = ChatOpenAI(
    model="o3",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def planner_node(state: MessageContext):
    """
    Runs the planning LLM and stores the resulting plan in state.
    """
    logger.info("[PLANNER] Starting planner_node with state: %s", state)
    llm_reply = llm.invoke([plan_prompt(state)])
    logger.info("[PLANNER] LLM reply: %s", llm_reply.content)
    print("Planner response:")
    print("--------------------------------")
    print(llm_reply.content)
    print()
    try:
        content_str = llm_reply.content if isinstance(
            llm_reply.content, str) else str(llm_reply.content)
        parsed_plan = json.loads(content_str)
    except json.JSONDecodeError:
        logger.error("[PLANNER] Invalid JSON from planner: %s", llm_reply.content)
        raise ValueError(
            f"Planner returned invalid JSON:\n{llm_reply.content}"
            )
    replan = state.get("replan_flag", False)
    updated_plan: Dict[str, Any] = parsed_plan
    logger.info("[PLANNER] Updated plan: %s", updated_plan)
    return Command(
        update={
            "plan": updated_plan,
            "messages": [HumanMessage(
                content=llm_reply.content,
                name="replan" if replan else "initial_plan")],
            "user_query": state.get("user_query", state["messages"][0].content),
            "current_step": 1 if not replan else state["current_step"],
            "replan_flag": state.get("replan_flag", False),
            "last_reason": "",
            "enabled_agents": state.get("enabled_agents"),
        },
        goto="executor",
    )

def executor_node(
    state: MessageContext,
) -> Command[Literal['planner', 'web_researcher', 'chart_generator', 'chart_summarizer', 'synthesizer']]:
    logger.info("[EXECUTOR] Starting executor_node with state: %s", state)
    plan: Dict[str, Any] = state.get("plan", {})
    step: int = state.get("current_step", 1)
    logger.info("[EXECUTOR] Plan: %s, Step: %d", plan, step)
    if state.get("replan_flag"):
        planned_agent = plan.get(str(step), {}).get("agent")
        logger.info("[EXECUTOR] Replan flag set. Planned agent: %s", planned_agent)
        return Command(
            update={
                "replan_flag": False,
                "current_step": step + 1,  # advance because we executed the planned agent
            },
            goto=planned_agent,
        )
    # 1) Build prompt & call LLM
    logger.info("[EXECUTOR] Building executor prompt and invoking LLM.")
    llm_reply = llm.invoke([executor_prompt(state)])
    logger.info("[EXECUTOR] LLM reply: %s", llm_reply.content)
    print("Executor response:")
    print("--------------------------------")
    print(llm_reply.content)
    print()
    try:
        content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
        parsed = json.loads(content_str)
        replan: bool = parsed["replan"]
        goto: str   = parsed["goto"]
        reason: str = parsed["reason"]
        query: str  = parsed["query"]
    except Exception as exc:
        logger.error("[EXECUTOR] Invalid executor JSON: %s", llm_reply.content)
        raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from exc
    updates: Dict[str, Any] = {
        "messages": [HumanMessage(content=llm_reply.content, name="executor")],
        "last_reason": reason,
        "agent_query": query,
    }
    replans: Dict[int, int] = state.get("replan_attempts", {}) or {}
    logger.info("[EXECUTOR] Replans so far: %s", replans)
    step_replans = replans.get(step, 0)
    logger.info("[EXECUTOR] Replans for step %d: %d", step, step_replans)
    if replan:
        if step_replans < MAX_REPLANS:
            replans[step] = step_replans + 1
            updates.update({
                "replan_attempts": replans,
                "replan_flag": True,     # ensure next turn executes the planned agent once
                "current_step": step,    # stay on same step for the new plan
            })
            logger.info("[EXECUTOR] Replanning. Updated replans: %s", replans)
            return Command(update=updates, goto="planner")
        else:
            next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer")
            updates["current_step"] = step + 1
            logger.info("[EXECUTOR] Max replans reached. Moving to next agent: %s", next_agent)
            return Command(update=updates, goto=next_agent)
    planned_agent = plan.get(str(step), {}).get("agent")
    updates["current_step"] = step + 1 if goto == planned_agent else step
    updates["replan_flag"] = False
    logger.info("[EXECUTOR] Happy path. Goto: %s, Updates: %s", goto, updates)
    return Command(update=updates, goto=goto)

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query for the web")

# 2. Define function with signature matching schema
def web_search(query: str) -> str:
    return TavilySearch(max_results=2).invoke(query)

# 3. Make strict tool
web_research = StructuredTool.from_function(
    func=web_search,
    name="web_research",
    description="Useful for searching the web for information",
    args_schema=WebSearchInput,
)


llm_researcher = ChatOpenAI(model="gpt-4.1")

web_search_agent = create_react_agent(
    llm_researcher,
    tools=[web_research],
    prompt="""
        You are the Researcher. You can ONLY perform research 
        by using the provided search tool (web_research). 
        When you have found the necessary information, end your output.  
        Do NOT attempt to take further actions.
    """,
)


def web_research_node(
    state: MessageContext,
) -> Command[Literal["executor"]]:
    agent_query = state.get("agent_query")
    logger.info("[WEB_RESEARCH] Web researcher query: %s", agent_query)
    result = web_search_agent.invoke({"messages":agent_query})
    logger.info("[WEB_RESEARCH] Agent result: %s", result)
    goto = "executor"
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="web_researcher"
    )
    logger.info("[WEB_RESEARCH] Updated messages: %s", result["messages"])
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

chart_agent = create_react_agent(
    llm_researcher,
    [python_repl_tool],
    prompt=agent_system_prompt(
        """
        You can only generate charts. You are working with a researcher 
        colleague.
        1) Use matplotlib to generate charts add this import matplotlib matplotlib.use('TkAgg').
        2) Dont, Print the chart, Ignore plt.show() .
        3) Save the chart to a file in the current working directory plt.savefig('filename.png').
        4) At the very end of your message, output EXACTLY two lines
        so the summarizer can find them:
           CHART_PATH: <relative_path_to_chart_file>
           CHART_NOTES: <one concise sentence summarizing the main insight in the chart>
        Do not include any other trailing text after these two lines.
        """
    ),
)

def chart_node(state: MessageContext) -> Command[Literal["chart_summarizer"]]:
    logger.info("[CHART] Starting chart_node with state: %s", state)
    result = chart_agent.invoke(state)
    logger.info("[CHART] Chart generator answer: %s", result['messages'][-1].content)
    print(f"Chart generator answer: {result['messages'][-1].content}")
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    goto="chart_summarizer"
    logger.info("[CHART] Updated messages: %s", result["messages"])
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


chart_summary_agent = create_react_agent(
    llm_researcher,
    tools=[],  # Add image processing tools if available/needed.
    prompt=agent_system_prompt(
        "You can only generate image captions. You are working with a researcher colleague and a chart generator colleague. "
        + "Your task is to generate a standalone, concise summary for the provided chart image saved at a local PATH, where the PATH should be and only be provided by your chart generator colleague. The summary should be no more than 3 sentences and should not mention the chart itself."
    ),
)

def chart_summary_node(
    state: MessageContext,
) -> Command[Literal[END]]:
    logger.info("[CHART_SUMMARY] Starting chart_summary_node with state: %s", state)
    result = chart_summary_agent.invoke(state)
    logger.info("[CHART_SUMMARY] Chart summarizer answer: %s", result['messages'][-1].content)
    print(f"Chart summarizer answer: {result['messages'][-1].content}")
    goto = END
    logger.info("[CHART_SUMMARY] Updated messages: %s", result["messages"])
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
            "final_answer": result["messages"][-1].content,
        },
        goto=goto,
    )

def synthesizer_node(state: MessageContext) -> Command[Literal[END]]:
    """
    Creates a concise, human‑readable summary of the entire interaction,
    **purely in prose**.

    It ignores structured tables or chart IDs and instead rewrites the
    relevant agent messages (research results, chart commentary, etc.)
    into a short final answer.
    """
    # Gather informative messages for final synthesis
    relevant_msgs = [
        m.content for m in state.get("messages", [])
        if getattr(m, "name", None) in ("web_researcher", 
                                        "chart_generator", 
                                        "chart_summarizer")
    ]
    user_question = state.get("user_query", state.get("messages", [{}])[0].content if state.get("messages") else "")
    synthesis_instructions = (
        """
        You are the Synthesizer. Use the context below to directly 
        answer the user's question. Perform any lightweight calculations, 
        comparisons, or inferences required. Do not invent facts not 
        supported by the context. If data is missing, say what's missing
        and, if helpful, offer a clearly labeled best-effort estimate 
        with assumptions.\n\n
        Produce a concise response that fully answers the question, with 
        the following guidance:\n
        - Start with the direct answer (one short paragraph or a tight bullet list).\n
        - Include key figures from any 'Results:' tables (e.g., totals, top items).\n
        - If any message contains citations, include them as a brief 'Citations: [...]' line.\n
        - Keep the output crisp; avoid meta commentary or tool instructions.
        """
        )
    summary_prompt = [
        HumanMessage(content=(
            f"User question: {user_question}\n\n"
            f"{synthesis_instructions}\n\n"
            f"Context:\n\n" + "\n\n---\n\n".join(relevant_msgs)
        ))
    ]
    logger.info("[SYNTHESIZER] Synthesis prompt: %s", summary_prompt)
    llm_reply = llm_researcher.invoke(summary_prompt)
    answer = llm_reply.content.strip()
    logger.info("[SYNTHESIZER] Synthesizer answer: %s", answer)
    print(f"Synthesizer answer: {answer}")
    return Command(
        update={
            "final_answer": answer,
            "messages": [HumanMessage(content=answer, name="synthesizer")],
        },
        goto=END,          
    )

flow = StateGraph(MessageContext)
flow.add_node("planner", planner_node)
flow.add_node("executor", executor_node)
flow.add_node("web_researcher", web_research_node)
flow.add_node("synthesizer", synthesizer_node)
flow.add_node("chart_generator", chart_node)
flow.add_node("chart_summarizer", chart_summary_node)
flow.add_edge(START, "planner")

graph = flow.compile()
query = "Chart the current market capitalization of the top 5 banks in the US and Report that?"
print(f"Query: {query}")

state = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "enabled_agents": ["web_researcher", "chart_generator", 
                               "chart_summarizer", "synthesizer"],
        }
graph.invoke(state)
H = "Aakash"
