"""Agent descriptions and capabilities."""
from typing import Dict, Any, List


def get_agent_descriptions() -> Dict[str, Dict[str, Any]]:
    """
    Return structured agent descriptions with capabilities and guidelines.
    Edit this function to change how the planner/executor reason about agents.
    """
    return {
        "web_researcher": {
            "name": "Web Researcher",
            "capability": "Fetch public data via Tavily web search",
            "use_when": "Public information, news, current events, or external facts are needed",
            "limitations": "Cannot access private/internal company data, Don't split many recursive queries",
            "output_format": "Raw research data and findings from public sources",
        },
        "cortex_researcher": {
            "name": "Cortex Researcher",
            "capability": "Query private/company data in Snowflake, including structured deal records (company name, deal value, sales rep, close date, deal status, product line) and unstructured sales meeting notes, via Snowflake Cortex Agents.",
            "use_when": "Internal documents, company databases, or private data access is required",
            "limitations": "Cannot access public web data",
            "output_format": "For structured requests, return the exact fields and include SQL when applicable; for unstructured, return concise relevant excerpts with citations.",
        },
        "chart_generator": {
            "name": "Chart Generator",
            "capability": "Build visualizations from structured data",
            "use_when": "User explicitly requests charts, graphs, plots, visualizations (keywords: chart, graph, plot, visualise, bar-chart, line-chart, histogram, etc.)",
            "limitations": "Requires structured data input from previous steps",
            "output_format": "Visual charts and graphs",
            "position_requirement": "Must be used as final step after data gathering is complete",
        },
        "chart_summarizer": {
            "name": "Chart Summarizer",
            "capability": "Summarize and explain chart visualizations",
            "use_when": "After chart_generator has created a visualization",
            "limitations": "Requires a chart as input",
            "output_format": "Written summary and analysis of chart content",
        },
        "synthesizer": {
            "name": "Synthesizer",
            "capability": "Write comprehensive prose summaries of findings",
            "use_when": "Final step when no visualization is requested - combines all previous research",
            "limitations": "Requires research data from previous steps",
            "output_format": "Coherent written summary incorporating all findings",
            "position_requirement": "Should be used as final step when no chart is needed",
        },
    }


def get_enabled_agents(enabled_list: List[str] = None) -> List[str]:
    """
    Get the list of enabled agents.
    
    Args:
        enabled_list: Optional list of enabled agents. If None, returns baseline.
    
    Returns:
        List of enabled agent names
    """
    baseline = ["web_researcher", "chart_generator", "chart_summarizer", "synthesizer"]
    
    if not enabled_list:
        return baseline
    
    allowed = {"web_researcher", "cortex_researcher", "chart_generator", "chart_summarizer", "synthesizer"}
    filtered = [a for a in enabled_list if a in allowed]
    return filtered if filtered else baseline
