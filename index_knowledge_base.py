from document_processor import DocumentProcessor
import os
from config import Config
import time

def main():
    """Index the knowledge base."""
    start_time = time.time()
    
    print(f"Indexing knowledge base at: {Config.KNOWLEDGE_BASE_PATH}")
    
    # Check if knowledge base exists
    if not os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        print(f"Error: Knowledge base path does not exist: {Config.KNOWLEDGE_BASE_PATH}")
        print("Please make sure the 'ai_assignment' folder exists on your desktop.")
        return
    
    # Process knowledge base
    processor = DocumentProcessor()
    
    try:
        # Process and index the knowledge base
        vector_store = processor.process_knowledge_base()
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        print(f"Indexing completed in {elapsed_time:.2f} seconds")
        
        # Test the vector store with a simple query
        query = "test query"
        results = vector_store.similarity_search(query, k=1)
        
        if results:
            print("\nVector store is working correctly!")
            print(f"Test query: '{query}'")
            print(f"Found {len(results)} results")
        else:
            print("\nWarning: Vector store returned no results for test query")
            
    except Exception as e:
        print(f"Error processing knowledge base: {e}")

if __name__ == "__main__":
    main()
