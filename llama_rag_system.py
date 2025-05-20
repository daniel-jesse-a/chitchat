import os
from typing import List, Dict, Any
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LlamaRAGSystem:
    """RAG system using LlamaIndex instead of LangChain."""
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        """Initialize the RAG system."""
        self.knowledge_base_path = knowledge_base_path
        
        # Configure embeddings
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        
        # Initialize OpenAI LLM if API key is available
        if os.getenv("OPENAI_API_KEY"):
            self.llm_predictor = LLMPredictor(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
            )
            print("OpenAI language model initialized successfully.")
        else:
            print("OpenAI API key not found. Using a simple response generator instead.")
            self.llm_predictor = None
        
        # Create service context
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=self.llm_predictor,
            embed_model=self.embed_model,
            node_parser=SimpleNodeParser.from_defaults(chunk_size=64, chunk_overlap=10)
        )
        
        # Load or create index
        self.index = self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        index_path = "./vector_db/llama_index"
        
        if os.path.exists(index_path):
            try:
                # Load existing index
                from llama_index import StorageContext, load_index_from_storage
                
                storage_context = StorageContext.from_defaults(persist_dir=index_path)
                index = load_index_from_storage(storage_context, service_context=self.service_context)
                print(f"Index loaded from {index_path}")
                return index
            except Exception as e:
                print(f"Error loading index: {e}")
        
        # Create new index
        return self._create_index()
    
    def _create_index(self):
        """Create a new index from the knowledge base."""
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(f"Knowledge base not found at {self.knowledge_base_path}")
        
        # Load documents
        documents = SimpleDirectoryReader(self.knowledge_base_path).load_data()
        print(f"Loaded {len(documents)} documents from knowledge base")
        
        # Create index
        index = GPTVectorStoreIndex.from_documents(
            documents, 
            service_context=self.service_context
        )
        
        # Save index
        index_path = "./vector_db/llama_index"
        os.makedirs(index_path, exist_ok=True)
        index.storage_context.persist(persist_dir=index_path)
        print(f"Index saved to {index_path}")
        
        return index
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        try:
            if not self.index:
                return {
                    "answer": "Error: Index not loaded. Please index your knowledge base first.",
                    "sources": [],
                    "success": False
                }
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=5
            )
            
            # Get response
            response = query_engine.query(question)
            
            # Extract sources
            source_nodes = response.source_nodes
            sources = [node.node.metadata.get('file_name', 'Unknown source') for node in source_nodes]
            
            return {
                "answer": str(response),
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def process_knowledge_base(self):
        """Process the knowledge base and create a new index."""
        self.index = self._create_index()
        return self.index
