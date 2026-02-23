import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "chroma_db"

def get_retriever():
    """Initializes and returns the Chroma document retriever."""
    if not os.path.exists(DB_DIR):
        print("Creating an empty Chroma DB. Please run ingest.py later.")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Return a retriever that fetches the top 3 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})
