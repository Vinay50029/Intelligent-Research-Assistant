import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
DB_DIR = "chroma_db"

def ingest_documents():
    """Loads PDFs from the data directory, splits them, and stores them in ChromaDB."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} directory. Please add some PDFs.")
        return None

    print(f"Loading PDFs from {DATA_DIR}...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    if not documents:
        print(f"No documents found in {DATA_DIR}.")
        return None

    print(f"Loaded {len(documents)} document pages.")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks.")

    # Use free HuggingFace embeddings
    print("Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print("Ingestion complete!")
    return vectorstore

if __name__ == "__main__":
    ingest_documents()
