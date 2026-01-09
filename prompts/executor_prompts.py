"""Executor prompt templates."""
from typing import Dict, Any
from langchain.schema import HumanMessage

from prompts.agent_descriptions import get_agent_descriptions, get_enabled_agents
from config import MAX_REPLANS


def format_agent_guidelines_for_executor(enabled_list: list = None) -> str:
    """
    Format agent usage guidelines for the executor prompt.
    
    Args:
        enabled_list: List of enabled agent names
    
    Returns:
        Formatted string of agent guidelines
    """
    descriptions = get_agent_descriptions()
    enabled = get_enabled_agents(enabled_list)
    guidelines = []
    
    if "web_researcher" in enabled:
        web_desc = descriptions['web_researcher']
        guidelines.append(f"- Use `\"web_researcher\"` when {web_desc['use_when'].lower()}.")
    if "cortex_researcher" in enabled:
        cortex_desc = descriptions['cortex_researcher']
        guidelines.append(f"- Use `\"cortex_researcher\"` for {cortex_desc['use_when'].lower()}.")
    
    return "\n".join(guidelines)


def build_executor_prompt(
    user_query: str,
    current_step: int,
    plan: Dict[str, Any],
    replan_flag: bool = False,
    replan_attempts: Dict[int, int] = None,
    recent_messages: list = None,
    enabled_agents: list = None
) -> HumanMessage:
    """
    Build the single‑turn JSON prompt that drives the executor LLM.
    
    Args:
        user_query: The user's original query
        current_step: Current step in the plan
        plan: The execution plan
        replan_flag: Whether we just replanned
        replan_attempts: Dictionary tracking replan attempts per step
        recent_messages: Recent messages for context
        enabled_agents: List of enabled agent names
    
    Returns:
        HumanMessage with the executor prompt
    """
    plan_block: Dict[str, Any] = plan.get(str(current_step), {})
    attempts = (replan_attempts or {}).get(current_step, 0)
    
    # Get agent guidelines dynamically
    executor_guidelines = format_agent_guidelines_for_executor(enabled_agents)
    plan_agent = plan_block.get("agent", "web_researcher")
    
    enabled = get_enabled_agents(enabled_agents)
    enabled_for_executor = [a for a in enabled if a in ['web_researcher', 'cortex_researcher', 'chart_generator', 'chart_summarizer', 'synthesizer']]
    agent_enum = '|'.join(sorted(set(enabled_for_executor + ['planner'])))
    agent_list = '`, `'.join(sorted(set(enabled_for_executor + ['planner'])))
    
    executor_prompt = f"""
You are the **executor** in a multi‑agent system with these agents:
`{agent_list}`.

**Tasks**
1. Decide if the current plan needs revision. → `"replan": true|false`
2. Decide which agent to run next.           → `"goto": "<agent_name>"`
3. Give one‑sentence justification.          → `"reason": "<text>"`
4. Write the exact question that the chosen agent should answer
                                             → `"query": "<text>"`

**Guidelines**
{executor_guidelines}
- After **{MAX_REPLANS}** failed replans for the same step, move on.
- If you *just replanned* (replan_flag is true) let the assigned agent try before
  requesting another replan.

Respond **only** with valid JSON (no additional text):

{{
  "replan": <true|false>,
  "goto": "<{agent_enum}>",
  "reason": "<1 sentence>",
  "query": "<text>"
}}

**PRIORITIZE FORWARD PROGRESS:** Only replan if the current step is completely blocked.
1. If any reasonable data was obtained that addresses the step's core goal, set `"replan": false` and proceed.
2. Set `"replan": true` **only if** ALL of these conditions are met:
   • The step has produced zero useful information
   • The missing information cannot be approximated or obtained by remaining steps
   • `attempts < {MAX_REPLANS}`
3. When `attempts == {MAX_REPLANS}`, always move forward (`"replan": false`).

### Decide `"goto"`
- If `"replan": true` → `"goto": "planner"`.
- If current step has made reasonable progress → move to next step's agent.
- Otherwise execute the current step's assigned agent (`{plan_agent}`).

### Build `"query"`
Write a clear, standalone instruction for the chosen agent. If the chosen agent 
is `web_researcher` or `cortex_researcher`, the query should be a standalone question, 
written in plain english, and answerable by the agent.

Ensure that the query uses consistent language as the user's query.

Context you can rely on:
- User query ..............: {user_query}
- Current step index ......: {current_step}
- Current plan step .......: {plan_block}
- Just‑replanned flag .....: {replan_flag}
- Previous messages .......: {recent_messages[-4:] if recent_messages else []}

Respond **only** with JSON, no extra text.
"""
    
    return HumanMessage(content=executor_prompt)
