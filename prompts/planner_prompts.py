"""Planner prompt templates."""
import json
from typing import Dict, Any
from langchain.schema import HumanMessage

from prompts.agent_descriptions import get_agent_descriptions, get_enabled_agents
from config import MAX_REPLANS


def format_agent_list_for_planning(enabled_list: list = None) -> str:
    """
    Format agent descriptions for the planning prompt.
    
    Args:
        enabled_list: List of enabled agent names
    
    Returns:
        Formatted string of agent descriptions
    """
    descriptions = get_agent_descriptions()
    enabled = get_enabled_agents(enabled_list)
    agent_list = []
    
    for agent_key, details in descriptions.items():
        if agent_key not in enabled:
            continue
        agent_list.append(f"  • `{agent_key}` – {details['capability']}")
    
    return "\n".join(agent_list)


def format_agent_guidelines_for_planning(enabled_list: list = None) -> str:
    """
    Format agent usage guidelines for the planning prompt.
    
    Args:
        enabled_list: List of enabled agent names
    
    Returns:
        Formatted string of agent guidelines
    """
    descriptions = get_agent_descriptions()
    enabled = set(get_enabled_agents(enabled_list))
    guidelines = []
    
    # Cortex vs Web researcher (only include guidance for enabled agents)
    if "cortex_researcher" in enabled:
        guidelines.append(f"- Use `cortex_researcher` when {descriptions['cortex_researcher']['use_when'].lower()}.")
    if "web_researcher" in enabled:
        guidelines.append(f"- Use `web_researcher` for {descriptions['web_researcher']['use_when'].lower()}.")
    
    # Chart generator specific rules
    if "chart_generator" in enabled:
        chart_desc = descriptions['chart_generator']
        if "chart_summarizer" in enabled:
            guidelines.append(f"- **Include `chart_generator` _only_ if {chart_desc['use_when'].lower()}**. If included, `chart_generator` must be {chart_desc['position_requirement'].lower()}. After chart generation, you can optionally use `chart_summarizer` to describe the chart, OR go directly to `synthesizer` for the final answer.")
        else:
            guidelines.append(f"- **Include `chart_generator` _only_ if {chart_desc['use_when'].lower()}**. If included, `chart_generator` must be {chart_desc['position_requirement'].lower()}.")
    
    # Synthesizer default
    if "synthesizer" in enabled:
        synth_desc = descriptions['synthesizer'] 
        guidelines.append(f"  – Use `synthesizer` as {synth_desc['position_requirement'].lower()} to create the final comprehensive answer from all gathered data.")
    
    return "\n".join(guidelines)


def build_plan_prompt(
    user_query: str,
    replan_flag: bool = False,
    prior_plan: Dict[str, Any] = None,
    replan_reason: str = "",
    enabled_agents: list = None
) -> HumanMessage:
    """
    Build the prompt that instructs the LLM to return a high‑level plan.
    
    Args:
        user_query: The user's query
        replan_flag: Whether this is a replanning request
        prior_plan: The previous plan (for replanning)
        replan_reason: Reason for replanning
        enabled_agents: List of enabled agent names
    
    Returns:
        HumanMessage with the planning prompt
    """
    # Get agent descriptions dynamically
    agent_list = format_agent_list_for_planning(enabled_agents)
    agent_guidelines = format_agent_guidelines_for_planning(enabled_agents)
    
    enabled = get_enabled_agents(enabled_agents)
    
    # Build planner agent enum based on enabled agents
    enabled_for_planner = [
        a for a in enabled
        if a in ("web_researcher", "cortex_researcher", "chart_generator", "synthesizer")
    ]
    planner_agent_enum = " | ".join(enabled_for_planner) or "web_researcher | chart_generator | synthesizer"
    
    prompt = f"""
You are the **Planner**. Create an EFFICIENT execution plan with MINIMAL steps.

**CRITICAL RULES:**
1. Maximum 3 web_researcher steps (combine related queries into ONE comprehensive search)
2. ALWAYS end with synthesizer
3. Add chart_generator ONLY if user explicitly asks for chart/visualization

**Available Agents:**
{agent_list}

**Output Format (JSON only):**
{{
  "1": {{"agent": "web_researcher", "action": "comprehensive search query covering all needed data"}},
  "2": {{"agent": "chart_generator", "action": "create visualization"}},  // Only if chart requested
  "3": {{"agent": "synthesizer", "action": "create final answer"}}
}}

**Example - CORRECT (efficient):**
Query: "Compare revenue and market cap of Apple, Google, Microsoft"
{{
  "1": {{"agent": "web_researcher", "action": "Apple Google Microsoft revenue market cap 2023 2024 comparison"}},
  "2": {{"agent": "synthesizer", "action": "synthesize comparison"}}
}}

**Example - WRONG (too many steps):**
{{
  "1": {{"agent": "web_researcher", "action": "Apple revenue"}},
  "2": {{"agent": "web_researcher", "action": "Google revenue"}},
  "3": {{"agent": "web_researcher", "action": "Microsoft revenue"}},
  "4": {{"agent": "web_researcher", "action": "Apple market cap"}},
  ...
}}

Guidelines:
{agent_guidelines}
"""
    
    if replan_flag:
        prompt += f"""

**REPLAN REQUIRED:** {replan_reason}

Previous plan (rejected):
{json.dumps(prior_plan or {{}}, indent=2)}

**FIX:** Reduce web_researcher steps to maximum 3 by combining searches.
"""
    
    prompt += f'\n\nUser query: "{user_query}"\n\nReturn JSON only:'
    
    return HumanMessage(content=prompt)
