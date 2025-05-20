from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from document_loader import DocumentLoader
from config import Config

class DocumentProcessor:
    """Class to process and index documents for RAG."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        self.vector_db_path = Config.VECTOR_DB_PATH
        self.embedding_model_name = Config.EMBEDDING_MODEL
        
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def create_vector_index(self, documents: List[Document], save: bool = True):
        """Create a vector index from documents."""
        # Split documents into chunks
        chunks = self.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the vector store if requested
        if save:
            vector_store.save_local(self.vector_db_path)
            print(f"Vector index saved to {self.vector_db_path}")
        
        return vector_store
    
    def load_vector_index(self):
        """Load the vector index from disk."""
        try:
            vector_store = FAISS.load_local(self.vector_db_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True)
            print(f"Vector index loaded from {self.vector_db_path}")
            return vector_store
        except Exception as e:
            print(f"Error loading vector index: {e}")
            return None
    
    def process_knowledge_base(self):
        """Process the entire knowledge base and create a vector index."""
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_documents()
        
        # Create vector index
        vector_store = self.create_vector_index(documents)
        
        return vector_store
