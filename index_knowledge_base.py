from llama_rag_system import LlamaRAGSystem

if __name__ == "__main__":
    print("Indexing knowledge base...")
    rag = LlamaRAGSystem()
    rag.process_knowledge_base()
    print("Indexing complete!")
