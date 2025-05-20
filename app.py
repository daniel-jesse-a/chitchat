import os
import sys
from rag_system import RAGSystem
from document_processor import DocumentProcessor
from config import Config

def check_environment():
    """Check if the environment is properly set up."""
    # Check if knowledge base exists
    if not os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        print(f"Error: Knowledge base path not found: {Config.KNOWLEDGE_BASE_PATH}")
        print("Please make sure the 'ai_assignment' folder exists on your desktop.")
        return False
    
    # Check if vector store exists
    if not os.path.exists(Config.VECTOR_DB_PATH):
        print(f"Vector store not found at {Config.VECTOR_DB_PATH}")
        print("You need to index your knowledge base first.")
        return False
    
    # Check OpenAI API key if using OpenAI
    if not Config.OPENAI_API_KEY:
        print("Warning: OpenAI API key not found in .env file.")
        print("You'll need to set up a local model or add your API key.")
    
    return True

def index_knowledge_base():
    """Index the knowledge base."""
    print(f"Indexing knowledge base at: {Config.KNOWLEDGE_BASE_PATH}")
    
    processor = DocumentProcessor()
    try:
        processor.process_knowledge_base()
        print("Indexing completed successfully!")
        return True
    except Exception as e:
        print(f"Error indexing knowledge base: {e}")
        return False

def interactive_mode():
    """Run the RAG system in interactive mode."""
    print("\n" + "=" * 50)
    print("RAG AI System - Interactive Mode")
    print("=" * 50)
    print("Ask questions about your knowledge base.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 50 + "\n")
    
    try:
        rag_system = RAGSystem()
        
        while True:
            # Get user question
            question = input("\nQuestion: ")
            
            # Check for exit command
            if question.lower() in ['exit', 'quit']:
                print("Exiting interactive mode. Goodbye!")
                break
            
            # Skip empty questions
            if not question.strip():
                continue
            
            # Answer the question
            result = rag_system.answer_question(question)
            
            if result["success"]:
                print("\nAnswer:")
                print(result["answer"])
                
                if result["sources"]:
                    print("\nSources:")
                    for i, source in enumerate(result["sources"]):
                        print(f"  {i+1}. {source}")
            else:
                print(f"\nError: {result['answer']}")
            
            print("\n" + "-" * 50)
            
    except Exception as e:
        print(f"Error in interactive mode: {e}")

def main():
    """Main entry point for the application."""
    # Check if environment is set up
    if not check_environment():
        choice = input("Would you like to index your knowledge base now? (y/n): ")
        if choice.lower() == 'y':
            if index_knowledge_base():
                print("Knowledge base indexed successfully!")
            else:
                print("Failed to index knowledge base. Exiting.")
                return
        else:
            print("Exiting. Please set up your environment before running the application.")
            return
    
    # Run in interactive mode
    interactive_mode()

if __name__ == "__main__":
    main()
