# --- 1. IMPORTING NECESSARY LIBRARIES ---
# Think of this like gathering ingredients before cooking.
import streamlit as st # Streamlit is the framework we use to easily build the web interface
from dotenv import load_dotenv # Used to load secret keys from a hidden '.env' file safely
import os # Helps interact with the computer's operating system (like checking files)

# This line officially loads the variables from the .env file so we can read them
load_dotenv()

from langchain_core.messages import HumanMessage # Represents a message typed by the user
from src.graph.workflow import create_workflow # Imports our custom AI thinking process
from src.rag.ingest import ingest_documents # Imports our function to read and understand PDFs
    
# --- 2. SETTING UP THE WEB PAGE VISUALS ---
from PIL import Image

# Load a robot picture to use as the website's favicon (the small icon at the top of the browser tab)
robo_icon = Image.open("IRA_pic.png")
# Configure the page title and icon
st.set_page_config(page_title="Intelligent Research Assistant", page_icon=robo_icon)

# Display the main large title on the screen
st.title("🤖 Intelligent Research Assistant")
st.markdown("Ask questions about your documents (RAG) or search the web for real-time information. A **Supervisor Agent** will route your request automatically.")

# --- 3. THE SIDEBAR SETTINGS ---
# The sidebar is the collapsible menu on the left side of the screen
with st.sidebar:
    st.header("AI Model Settings")
    selected_model = st.selectbox(
        "Choose AI Model", 
        ["Gemini 2.5 Flash", "GPT-4o Mini", "GPT-4o"]
    )
    
    with st.expander("🔑 Secure-BYOK (Bring-Your-Own-Key)", expanded=False):
        st.markdown("Use your own API keys. Keys are not saved and clear when the session ends.")
        user_gemini_key = st.text_input(
            "Google Gemini API Key", 
            type="password", 
            help="""**How to get a Gemini API Key:**

1. Go to Google AI Studio: Open this link in your browser: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account.
3. Click the blue button that says "Create API key".
4. You can select an existing project or create a new one, then click "Create API key in new project".
5. When the key is generated, click the "Copy" button (it will look like a long string of random letters and numbers starting with AIzaSy...).
6. Make sure to **set up billing** before using the API key (otherwise the model will not work).
"""
        )
        
        user_openai_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            help="""**How to get an OpenAI API Key:**

1. Go to: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in
3. Go to API Keys
4. Click "Create new secret key"
5. Copy the key (it looks like: sk-xxxx...)
6. Make sure to **add funds / set up billing** on the "Billing" page before using the API key (otherwise the model will not work).

Overrides the .env key"""
        )
        
    gemini_key = user_gemini_key if user_gemini_key else os.environ.get("GOOGLE_API_KEY")
    openai_key = user_openai_key if user_openai_key else os.environ.get("OPENAI_API_KEY")
    
    if selected_model.startswith("GPT"):
        if not openai_key or openai_key == "your_openai_api_key_here":
            st.warning("⚠️ Please enter your OpenAI API Key above")
            st.stop()
    else:
        if not gemini_key or gemini_key == "your_google_api_key_here":
            st.warning("⚠️ Please enter your Google API Key above")
            st.stop()
            
    st.divider()

    # --- 4. DOCUMENT UPLOAD LOGIC ---
    st.header("Document Management")
    st.markdown("Upload a PDF to add it to your research database. (Max 2 uploads, 200MB limit)")
    
    # 'session_state' is Streamlit's way of remembering things across page reloads.
    # Here we remember how many files the user has uploaded so far.
    if "upload_count" not in st.session_state:
        st.session_state.upload_count = 0

    # Stop them if they've uploaded too much
    if st.session_state.upload_count >= 2:
        st.warning("🛑 Upload limit reached(2 max).")
        uploaded_file = None
    else:
        # Show a file upload box specifically for PDFs
        uploaded_file = st.file_uploader(f"Upload a new PDF ({st.session_state.upload_count}/2 used)", type="pdf")
    
    # If the user successfully selected a file...
    if uploaded_file is not None:
        file_size_bytes = uploaded_file.size
        if file_size_bytes > 200 * 1024 * 1024:
            st.error("File exceeds the 200MB limit. Please upload a smaller file.")
        else:
            os.makedirs("data", exist_ok=True)
            save_path = os.path.join("data", uploaded_file.name)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_size = os.path.getsize(save_path)
            if file_size > 0:
                st.success(f"Saved {uploaded_file.name} to disk! ({file_size / (1024*1024):.1f} MB)")
                st.session_state.upload_count += 1
            else:
                st.error("File saved but it is empty! Something went wrong.")
            
    # --- 5. PROCESSING (INGESTING) THE PDF ---
    # When the user clicks the "Ingest Document" button:
    if st.button("Ingest Document"):
        # First check: Did they actually upload something into the 'data' folder yet?
        if not os.path.exists("data") or not any(f.endswith('.pdf') for f in os.listdir("data")):
            st.warning("⚠️ Please upload a PDF file using the box above FIRST, before clicking Ingest!")
        else:
            # Show a spinning loading icon while the computer reads the file
            with st.spinner("Processing into Vector DB... (This can take 1-2 mins to download the embeddings model the first time)"):
                # Call our specialized function that turns the PDF into searchable math numbers (vectors)
                result = ingest_documents()
                
                # If successful...
                if result is not None:
                    st.success("Documents Ingested Successfully! You can now ask questions.")
                    
                    # Cleanup: Delete the PDF from the 'data' folder to save computer space
                    try:
                        for filename in os.listdir("data"):
                            file_path = os.path.join("data", filename)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        st.info("🧹 Cleared uploaded files from disk to save space.")
                    except Exception as e:
                        print(f"Failed to clear data folder: {e}")
                else:
                    st.warning("No documents found to ingest.")

# --- 6. CHAT HISTORY AND SYSTEM MEMORY ---
# Prepare the app to remember the continuing conversation and AI setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "graph_app" not in st.session_state:
    st.session_state.graph_app = create_workflow()

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

is_rate_limited = False
if not user_gemini_key and not user_openai_key: 
    if st.session_state.question_count >= 5:
        st.warning("🛑 You have reached the free limit of 5 questions. Please enter your own API Key in the sidebar to continue.")
        is_rate_limited = True

# --- 7. HANDLING USER QUESTIONS ---
# This shows the text box at the bottom. If the user types a prompt and hits enter...
if prompt := st.chat_input("Ask a question (e.g., 'What is in my document?' or 'What is the news today?'):", disabled=is_rate_limited):
    st.session_state.question_count += 1 # Track how many questions they've asked
    
    # Save what the user said to the chat memory
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    # Display the user's message bubble on screen
    with st.chat_message("user"):
        st.markdown(prompt)

    # Now it's the AI's turn to respond...
    with st.chat_message("assistant"):
        with st.spinner("Thinking (Routing via LangGraph)..."):
            # We bundle up everything the AI needs to know to answer the question
            inputs = {
                "messages": st.session_state.messages,
                "model_choice": selected_model,
                "gemini_key": gemini_key, # Sending the keys over
                "openai_key": openai_key
            }
            
            try:
                # We send the inputs to our 'graph_app' (the brain that routes to Researcher or Document Reader)
                # 'invoke' means "run it now"
                final_state = st.session_state.graph_app.invoke(inputs)
                
                # The AI's final answer will be the very last message in the returned list
                ai_response = final_state["messages"][-1]
                
                # Print the AI's answer nicely on screen
                st.markdown(ai_response.content)
                # Save the AI's answer into memory so it remembers for next time
                st.session_state.messages.append(ai_response)
                
            except Exception as e:
                # If anything fails (like a bad API key), show an error safely
                st.error(f"An error occurred: {e}")
