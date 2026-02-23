from typing import Annotated
import operator
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup

from src.utils.llm_factory import get_llm

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    model_choice: str
    gemini_key: str
    openai_key: str

search = DuckDuckGoSearchRun()

@tool
def scrape_website(url: str) -> str:
    """
    This is a custom 'tool' that our AI can use. When the AI realizes it needs to 
    read a specific website, it calls this function with the website's URL.
    It returns the text content found on that webpage!
    """
    
    # --- SPECIAL CASE: LEETCODE PROFILES ---
    # LeetCode blocks normal bots, so we use their secret 'GraphQL API' to get the data instead.
    if "leetcode.com/u/" in url:
        try:
            username = url.rstrip('/').split('/')[-1]
            graphql_url = "https://leetcode.com/graphql/"
            payload = {
                "query": """
                query leetcodeProfileInfo($username: String!) {
                  matchedUser(username: $username) {
                    profile {
                      ranking
                      reputation
                    }
                    submitStatsGlobal {
                      acSubmissionNum {
                        difficulty
                        count
                      }
                    }
                  }
                }
                """,
                "variables": {"username": username},
                "operationName": "leetcodeProfileInfo"
            }
            
            headers = {"Content-Type": "application/json"}
            res = requests.post(graphql_url, json=payload, headers=headers).json()
            
            user_data = res.get('data', {}).get('matchedUser', {})
            profile = user_data.get('profile', {}) if user_data else {}
            stats = user_data.get('submitStatsGlobal', {}).get('acSubmissionNum', []) if user_data else []
            
            result = f"LeetCode Profile Data for {username}:\n"
            result += f"Ranking: {profile.get('ranking')}\n"
            result += f"Reputation: {profile.get('reputation')}\n"
            for stat in stats:
                result += f"{stat.get('difficulty')} Problems Solved: {stat.get('count')}\n"
            return result
        except Exception as e:
            return f"Failed to fetch LeetCode profile: {str(e)}"
            
    # --- NORMAL WEBSITES ---
    try:
        # We pretend to be a normal web browser (Chrome) so websites don't block us for being a bot
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        # Go to the URL and download the webpage content
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # If the website is down or gives a 404 error, this stops the process
        
        # BeautifulSoup is a library that helps us read the messy HTML code of a webpage easily
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # We don't want Javascript code or CSS styles in our text, so we remove those tags
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ')
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Truncate to avoid massive token usage
        return text[:8000]
    except Exception as e:
        return f"Failed to scrape the website: {str(e)}"

# We use create_react_agent to let the LLM decide which tools to use
from langgraph.prebuilt import create_react_agent

def researcher_node(state: AgentState):
    """
    This is the 'brain' of the Research Agent. 
    It takes the user's question, decides whether it needs to search the web or read a link, 
    and then generates an answer!
    """
    # Get the history of messages and user's API keys
    messages = state["messages"]
    model_choice = state.get("model_choice", "Gemini 2.5 Flash")
    gemini_key = state.get("gemini_key")
    openai_key = state.get("openai_key")
    
    # 1. Get the requested AI model (like Gemini or GPT)
    llm = get_llm(model_choice, temperature=0.7, google_api_key=gemini_key, openai_api_key=openai_key)
    
    # 2. Create the Agent - we give it a 'brain' (the LLM) and 'tools' (search the web, read a website).
    # 'create_react_agent' automatically loops through: Thought -> Action (Tool) -> Observation -> Repeat!
    agent_executor = create_react_agent(llm, tools=[search, scrape_website])
    
    # Get the latest question the user asked
    question = messages[-1].content
    
    history_msgs = messages[-5:-1] if len(messages) > 1 else []
    conversation_history = ""
    if history_msgs:
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])
        conversation_history = f"Recent Conversation:\n{history_str}\n"

    system_prompt = SystemMessage(content=f"""You are a helpful AI research assistant.
You have tools to search the web (DuckDuckGo) and scrape specific URLs. 
If the user provides a specific link, you should ALWAYS use the scrape_website tool to read it first before answering.
If search results are missing, failed, or irrelevant, fallback to your own general knowledge.

{conversation_history}""")
    
    # Run the react agent
    inputs = {"messages": [system_prompt, messages[-1]]}
    result = agent_executor.invoke(inputs)
    
    # Return the final message
    return {"messages": [result["messages"][-1]]}
