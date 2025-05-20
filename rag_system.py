from typing import List, Dict, Any
import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from document_processor import DocumentProcessor
from config import Config

class RAGSystem:
    """Retrieval-Augmented Generation system for answering questions based on a knowledge base."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.processor = DocumentProcessor()
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        # Initialize language model
        try:
            if hasattr(Config, 'OPENAI_API_KEY') and Config.OPENAI_API_KEY:
                self.llm = ChatOpenAI(
                    model_name=Config.MODEL_NAME,
                    temperature=0.2,
                    api_key=Config.OPENAI_API_KEY
                )
                print("OpenAI language model initialized successfully.")
            else:
                # If no OpenAI API key, use a fallback approach
                print("OpenAI API key not found. Using a simple response generator instead.")
                self.llm = SimpleResponseGenerator()
        except Exception as e:
            print(f"Error initializing language model: {e}")
            # Fallback to a simple response generator
            self.llm = SimpleResponseGenerator()
        
        # Load vector store if it exists
        self._load_vector_store()
        
        # Create the prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are an AI assistant that only answers questions based on the provided context.
            
            Context:
            {context}
            
            Question: {question}
            
            Instructions:
            1. Answer the question using ONLY the information from the context provided above.
            2. If the context doesn't contain the information needed to answer the question, respond with:
               "{out_of_scope_response}"
            3. Do not use any prior knowledge or information not contained in the context.
            4. Provide detailed, accurate answers when the information is available in the context.
            5. Always maintain a helpful, informative tone.
            
            Answer:"""
        )
    
    def _load_vector_store(self):
        """Load the vector store from disk."""
        if os.path.exists(Config.VECTOR_DB_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    Config.VECTOR_DB_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Vector store loaded from {Config.VECTOR_DB_PATH}")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_store = None
        else:
            print(f"Vector store not found at {Config.VECTOR_DB_PATH}")
            self.vector_store = None
    
    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if not self.vector_store:
            raise ValueError("Vector store not loaded. Please index your knowledge base first.")
        
        k = k or Config.TOP_K_RESULTS
        documents = self.vector_store.similarity_search(query, k=k)
        return documents
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a context string."""
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Extract source information if available
            source = doc.metadata.get('source', 'Unknown source')
            
            # Format the document content with source
            doc_text = f"Document {i+1} (Source: {source}):\n{doc.page_content}\n"
            context_parts.append(doc_text)
        
        # Join all document texts with separators
        return "\n" + "-" * 50 + "\n".join(context_parts)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        try:
            # Retrieve relevant documents
            documents = self.retrieve_documents(question)
            
            if not documents:
                return {
                    "answer": Config.OUT_OF_SCOPE_RESPONSE,
                    "sources": [],
                    "success": True
                }
            
            # Format context from documents
            context = self.format_context(documents)
            
            # Generate answer using language model
            if hasattr(self.llm, "invoke"):
                # For newer LangChain models
                response = self.llm.invoke(self.prompt_template.format(
                    context=context,
                    question=question,
                    out_of_scope_response=Config.OUT_OF_SCOPE_RESPONSE
                ))
                answer = response.content
            else:
                # For older LLMChain approach
                response = self.llm(self.prompt_template.format(
                    context=context,
                    question=question,
                    out_of_scope_response=Config.OUT_OF_SCOPE_RESPONSE
                ))
                answer = response
            
            # Extract sources for citation
            sources = [doc.metadata.get('source', 'Unknown source') for doc in documents]
            
            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "success": False
            }

# Simple fallback response generator when OpenAI is not available
class SimpleResponseGenerator:
    """A simple response generator that doesn't require an API key."""
    
    def invoke(self, prompt):
        """Generate a simple response based on the context."""
        # This is a very basic implementation
        # In a real system, you might want to use a local model
        return type('obj', (object,), {'content': "I've found some relevant information in the knowledge base, but I need an OpenAI API key to generate a detailed response. Please add your API key to the .env file."})
    
    def __call__(self, prompt):
        """Support for older LLMChain approach."""
        return "I've found some relevant information in the knowledge base, but I need an OpenAI API key to generate a detailed response. Please add your API key to the .env file."
