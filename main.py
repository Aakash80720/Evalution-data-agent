
"""Main entry point for the multi-agent system."""
from dotenv import load_dotenv
from langchain.schema import HumanMessage

from graph import build_graph
from config import ENABLED_AGENTS
from output_manager import OutputManager

load_dotenv(override=True)


def main():
    """Run the multi-agent system."""
    output_mgr = OutputManager(output_dir="outputs")
    graph = build_graph()
    
    query = "Chart the current market capitalization of the top 5 banks in the US and Report that?"
    
    print(f"\nExecuting query: {query}\n")
    
    state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "enabled_agents": ENABLED_AGENTS,
    }
    
    try:
        final_state = graph.invoke(state)
        
        final_answer = final_state.get("final_answer", "No final answer generated")
        if final_answer:
            print(f"\nFinal Answer:\n{final_answer}\n")
        
        # Extract chart information from messages
        chart_path = None
        chart_notes = None
        
        for msg in final_state.get("messages", []):
            if hasattr(msg, "name") and msg.name == "chart_generator":
                chart_path, chart_notes = output_mgr.extract_chart_info(msg.content)
                if chart_path:
                    chart_path = output_mgr.copy_chart_to_outputs(chart_path)
                break
        
        # Create metadata
        metadata = {
            "enabled_agents": final_state.get("enabled_agents", []),
            "total_steps": final_state.get("current_step", 0),
            "chart_generated": chart_path is not None,
        }
        
        # Save report
        report_path = output_mgr.save_markdown_report(
            query=query,
            final_answer=final_answer,
            chart_path=chart_path,
            chart_notes=chart_notes,
            metadata=metadata
        )
        
        print(f"Report saved: {report_path}")
        if chart_path:
            print(f"Chart saved: {chart_path}")
        
        return final_state
        
        
    except Exception as e:
        print(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
