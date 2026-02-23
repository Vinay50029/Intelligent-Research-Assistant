from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

def get_llm(model_choice="Gemini 2.5 Flash", temperature=0.0, google_api_key=None, openai_api_key=None):
    """
    Returns the appropriate LLM (Large Language Model) instance based on the user's choice and provided keys.
    Think of this as a 'vending machine' for AI models - you tell it which model you want, and it hands you the right one!
    """
    # Check if the user selected the smaller, faster OpenAI model
    if model_choice == "GPT-4o Mini":
        # Return the GPT-4o Mini model, passing in the user's OpenAI API key
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=openai_api_key)
    
    # Check if the user selected the more powerful OpenAI model
    elif model_choice == "GPT-4o":
        # Return the standard GPT-4o model
        return ChatOpenAI(model="gpt-4o", temperature=temperature, api_key=openai_api_key)
    
    # If neither OpenAI model was chosen, we default to using Google's Gemini model
    else:
        # Return the Gemini 2.5 Flash model, passing in the user's Google API key
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature, google_api_key=google_api_key)
