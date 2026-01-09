from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool, StructuredTool
from typing import Annotated
from langchain_tavily import TavilySearch
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. You will be used to execute python code
    that generates charts. Charts should be saved to the 'outputs' directory.
    This is visible to the user."""
    try:
        os.makedirs("outputs", exist_ok=True)
        # Inject matplotlib backend setting for thread safety
        setup_code = "import matplotlib\nmatplotlib.use('Agg')\n"
        result = repl.run(setup_code + code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    )
    return (
        result_str
        + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


def web_search(query: str) -> str:
        """Use this to search the web for information."""
        return TavilySearch(max_results=5).invoke(query)

web_search_tool = StructuredTool.from_function(
    web_search, name="web_research", 
    description="Useful for when you need to search the web for information",
    strict=True
)
