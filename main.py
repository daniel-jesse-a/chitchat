# main.py

import warnings
warnings.filterwarnings("ignore")

from llama_index.core.readers import SimpleDirectoryReader

# Load documents from the folder (supports .txt and .pdf if plugins installed)
reader = SimpleDirectoryReader(
    input_dir="./knowledge_base"  # Folder where your docs are
)

documents = reader.load_data()

# Print preview of loaded documents
for i, doc in enumerate(documents):
    print(f"\n📄 Document {i+1} Preview:\n{'-'*30}")
    print(doc.text[:300])  # Show first 300 characters



