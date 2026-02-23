from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.utils.llm_factory import get_llm

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    model_choice: str
    gemini_key: str
    openai_key: str


# Pydantic model for structured output routing
class RouteSchema(BaseModel):
    next_node: Literal["document_agent", "researcher_agent"] = Field(
        description="The next agent to route the query to."
    )

def supervisor_node(state: AgentState):
    """Analyzes the user request and routes it to the appropriate worker agent."""
    messages = state["messages"]
    model_choice = state.get("model_choice", "Gemini 2.5 Flash")
    gemini_key = state.get("gemini_key")
    openai_key = state.get("openai_key")
    
    # Initialize LLM inside the node so it matches the state's choice and keys
    llm = get_llm(model_choice, temperature=0.0, google_api_key=gemini_key, openai_api_key=openai_key)
    router_llm = llm.with_structured_output(RouteSchema)
    
    question = messages[-1].content
    
    history_msgs = messages[-5:-1] if len(messages) > 1 else []
    conversation_history = ""
    if history_msgs:
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])
        conversation_history = f"Recent Conversation:\n{history_str}\n"

    prompt = f"""You are the supervisor of a research assistant system. Your job is to route the user's question to the correct specialist.
    
    Available specialists:
    1. 'document_agent': Use this IF the user is asking about a specific PDF they uploaded, a document provided to you, or asking to summarize "my document", "the given context", or "the text".
    2. 'researcher_agent': Use this IF the question requires up-to-date information, facts from the internet, current events, or general knowledge not contained in a specific local document.
    
    {conversation_history}
    User Query: {question}
    """
    
    # Get the routing decision using structured output
    decision = router_llm.invoke(prompt)
    
    # Store the decision in the state so the graph can route
    return {"next_agent": decision.next_node}
