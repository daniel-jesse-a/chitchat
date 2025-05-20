import os
from dotenv import load_dotenv
import platform

# Load environment variables from .env file
load_dotenv()

# Get username for Windows path
USERNAME = os.getenv('USERNAME') or os.getenv('USER')

# Configuration settings
class Config:
    # Path to knowledge base - handle both local and Streamlit Cloud environments
    if os.path.exists(f"C:\\Users\\{USERNAME}\\Desktop\\ai_assignment\\knowledge_base"):
        KNOWLEDGE_BASE_PATH = f"C:\\Users\\{USERNAME}\\Desktop\\ai_assignment\\knowledge_base"
    elif os.path.exists("./knowledge_base"):
        KNOWLEDGE_BASE_PATH = "./knowledge_base"
    else:
        KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
    
    # Model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")  # OpenAI model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Sentence transformer model
    
    # Vector database settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    
    # RAG settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Response for out-of-scope questions
    OUT_OF_SCOPE_RESPONSE = os.getenv("OUT_OF_SCOPE_RESPONSE", 
        "I'm sorry, but I can only answer questions related to the information in my knowledge base.")
