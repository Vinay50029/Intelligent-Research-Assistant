from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag.retrieve import get_retriever

from src.utils.llm_factory import get_llm

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    model_choice: str
    gemini_key: str
    openai_key: str

def document_node(state: AgentState):
    """Answers questions based on retrieved local PDF documents."""
    messages = state["messages"]
    question = messages[-1].content
    
    retriever = get_retriever()
    
    if retriever:
        # Fetch relevant documents
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        context = "No documents have been loaded into the database yet."
        
    history_msgs = messages[-5:-1] if len(messages) > 1 else []
    conversation_history = ""
    if history_msgs:
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])
        conversation_history = f"Recent Conversation:\n{history_str}\n"

    prompt = f"""You are a helpful research assistant. Answer the user's question based strictly on the provided context. 
    If the context doesn't contain the answer, say that you don't know based on the provided documents.
    
    Context:
    {context}
    
    {conversation_history}
    Question: {question}
    """
    
    model_choice = state.get("model_choice", "Gemini 2.5 Flash")
    gemini_key = state.get("gemini_key")
    openai_key = state.get("openai_key")
    
    llm = get_llm(model_choice, temperature=0.2, google_api_key=gemini_key, openai_api_key=openai_key)
    response = llm.invoke(prompt)
    return {"messages": [response]}
