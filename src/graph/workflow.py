from typing import Annotated, Literal, TypedDict
import operator
from langgraph.graph import StateGraph, START, END

# Import the nodes
from src.agents.supervisor import supervisor_node
from src.agents.document_agent import document_node
from src.agents.researcher import researcher_node

# Define the overall state for the graph
class GraphState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str # Used by supervisor to route
    model_choice: str # The user selected LLM
    gemini_key: str
    openai_key: str


def router(state: GraphState) -> Literal["document_agent", "researcher_agent"]:
    """Routing function that reads the supervisor's decision."""
    return state["next_agent"]

def create_workflow():
    """Builds and compiles the LangGraph workflow."""
    
    # Initialize the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("document_agent", document_node)
    workflow.add_node("researcher_agent", researcher_node)
    
    # Define the edges
    # Start at the supervisor
    workflow.add_edge(START, "supervisor")
    
    # Add conditional edges from supervisor to the workers based on router function
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "document_agent": "document_agent",
            "researcher_agent": "researcher_agent"
        }
    )
    
    # Both workers lead to the END
    workflow.add_edge("document_agent", END)
    workflow.add_edge("researcher_agent", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app
