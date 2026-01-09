"""Supervisor prompt templates."""
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage


def build_supervisor_prompt(
    user_query: str,
    plan: Dict[str, Any],
    enabled_agents: List[str] = None,
    replan_attempts: Dict[int, int] = None
) -> HumanMessage:
    """
    Build the prompt for the supervisor to validate a plan.
    
    Args:
        user_query: The user's original query
        plan: The execution plan to validate
        enabled_agents: List of enabled agent names
        replan_attempts: Dictionary tracking replan attempts per step
    
    Returns:
        HumanMessage with the supervisor prompt
    """
    # Analyze the plan structure
    total_steps = len(plan)
    web_research_count = sum(1 for step in plan.values() if step.get("agent") == "web_researcher")
    chart_count = sum(1 for step in plan.values() if step.get("agent") == "chart_generator")
    
    enabled_str = ", ".join(enabled_agents) if enabled_agents else "unknown"
    total_replans = sum(replan_attempts.values()) if replan_attempts else 0
    
    prompt = f"""
You are the Plan Supervisor. Validate this execution plan.

**Query:** {user_query}

**Plan:** 
{json.dumps(plan, indent=2)}

**Stats:** {total_steps} steps, {web_research_count} web research, {chart_count} charts, {total_replans} prior replans

**APPROVE if:**
- Web research ≤ 3 steps
- Ends with synthesizer
- Chart included only if requested
- Logical flow (research → chart → synthesis)

**REJECT if:**
- Web research > 3 steps
- Missing synthesizer at end
- Chart requested but missing
- Illogical order

**After 2+ replans:** APPROVE to prevent loops (prioritize progress over perfection)

**Respond with JSON only:**
{{"needs_replan": true/false, "reason": "brief explanation", "issues": [], "suggestions": []}}

Example APPROVE: {{"needs_replan": false, "reason": "Efficient plan with 2 research steps", "issues": [], "suggestions": []}}
Example REJECT: {{"needs_replan": true, "reason": "5 web research steps exceeds limit of 3", "issues": ["Too many searches"], "suggestions": ["Combine into 2-3 comprehensive queries"]}}
"""
    
    return HumanMessage(content=prompt)
