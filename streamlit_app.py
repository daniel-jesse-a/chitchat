import streamlit as st
import os
from llama_rag_system import LlamaRAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Pregnancy Companion V0",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to initialize RAG system
@st.cache_resource
def initialize_rag_system():
    try:
        rag_system = LlamaRAGSystem()
        return rag_system, None
    except Exception as e:
        return None, str(e)

# Main app header
st.title("📚 Pregnancy Companion V0")
st.markdown("Ask questions about your knowledge base and get answers based solely on that information.")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses Retrieval-Augmented Generation (RAG) to answer questions 
    based only on the information in your knowledge base.
    
    Questions outside the scope of the knowledge base will receive a standard response.
    """)
    
    st.header("Knowledge Base Info")
    knowledge_base_path = "./knowledge_base"
    if os.path.exists(knowledge_base_path):
        file_count = sum(len(files) for _, _, files in os.walk(knowledge_base_path))
        st.success(f"✅ Knowledge base found with {file_count} files")
    else:
        st.error(f"❌ Knowledge base not found at {knowledge_base_path}")
    
    index_path = "./vector_db/llama_index"
    if os.path.exists(index_path):
        st.success(f"✅ Vector database found")
    else:
        st.error(f"❌ Vector database not found. Please index your knowledge base first.")
        
        if st.button("Index Knowledge Base"):
            with st.spinner("Indexing knowledge base... This may take a few minutes."):
                try:
                    rag_system, _ = initialize_rag_system()
                    rag_system.process_knowledge_base()
                    st.success("✅ Knowledge base indexed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error indexing knowledge base: {str(e)}")

# Initialize RAG system
rag_system, error = initialize_rag_system()

if error:
    st.error(f"Error initializing RAG system: {error}")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"{i+1}. {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your knowledge base"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_system.answer_question(prompt)
                
                st.markdown(response["answer"])
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"]
                })
                
                # Show sources if available
                if response["sources"]:
                    with st.expander("Sources"):
                        for i, source in enumerate(response["sources"]):
                            st.markdown(f"{i+1}. {source}")
