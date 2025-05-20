from config import Config
import os

def test_environment():
    print("Testing RAG environment setup...")
    
    # Check knowledge base path
    print(f"Knowledge base path: {Config.KNOWLEDGE_BASE_PATH}")
    if os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        print("✓ Knowledge base path exists")
    else:
        print("✗ Knowledge base path does not exist")
        print(f"  Please ensure the folder exists at: {Config.KNOWLEDGE_BASE_PATH}")
    
    # Check API key (if using OpenAI)
    if Config.OPENAI_API_KEY:
        masked_key = Config.OPENAI_API_KEY[:4] + "..." + Config.OPENAI_API_KEY[-4:]
        print(f"✓ OpenAI API key found: {masked_key}")
    else:
        print("✗ OpenAI API key not found")
        print("  If you plan to use OpenAI models, add your API key to the .env file")
        print("  Otherwise, we'll set up a local model in the next steps")
    
    print("\nSetup verification complete!")

if __name__ == "__main__":
    test_environment()
