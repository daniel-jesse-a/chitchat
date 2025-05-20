import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    CSVLoader
)
from config import Config

class DocumentLoader:
    """Class to load documents from the knowledge base directory."""
    
    def __init__(self, knowledge_base_path: str = None):
        """Initialize with path to knowledge base."""
        self.knowledge_base_path = knowledge_base_path or Config.KNOWLEDGE_BASE_PATH
        
    def _get_file_loader(self, file_path: str):
        """Return the appropriate loader based on file extension."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension == '.txt':
            # Try different encodings for text files
            try:
                return TextLoader(file_path, encoding='utf-8')
            except:
                try:
                    return TextLoader(file_path, encoding='latin-1')
                except:
                    try:
                        return TextLoader(file_path, encoding='cp1252')
                    except Exception as e:
                        print(f"Failed to load with common encodings: {e}")
                        return None
        elif file_extension in ['.docx', '.doc']:
            return Docx2txtLoader(file_path)
        elif file_extension in ['.html', '.htm']:
            return UnstructuredHTMLLoader(file_path)
        elif file_extension == '.csv':
            return CSVLoader(file_path)
        else:
            # Skip unsupported file types
            print(f"Unsupported file type: {file_path}")
            return None
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the knowledge base directory."""
        documents = []
        
        # Check if the knowledge base path exists
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(f"Knowledge base path not found: {self.knowledge_base_path}")
        
        # Walk through all files in the directory
        for root, _, files in os.walk(self.knowledge_base_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip hidden files
                if file.startswith('.'):
                    continue
                
                # Get the appropriate loader
                loader = None
                try:
                    loader = self._get_file_loader(file_path)
                except Exception as e:
                    print(f"Error creating loader for {file_path}: {e}")
                    continue
                
                # If we have a loader, load the document
                if loader:
                    try:
                        print(f"Loading: {file_path}")
                        docs = loader.load()
                        documents.extend(docs)
                        print(f"Successfully loaded: {file_path}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(documents)} documents from knowledge base")
        return documents
