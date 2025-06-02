import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

def test_openai_embeddings():
    # Load environment variables
    load_dotenv()
    
    # Verify API key is loaded
    print("Current working directory:", os.getcwd())
    print("Checking if .env file exists:", os.path.exists('.env'))
    
    # Try to read the .env file directly
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
            print(".env file content:", env_content)
    except Exception as e:
        print(f"Error reading .env file: {e}")
    
    # Load environment variables
    load_dotenv(override=True)
    
    # Check if the API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY not found in environment variables')
    print('API key from os.getenv:', api_key[:10] + '...')  # Print only first 10 chars for security
    print('API key from os.environ:', os.environ.get('OPENAI_API_KEY', 'Not found')[:10] + '...')
    
    # Ensure we're using the correct key
    os.environ['OPENAI_API_KEY'] = api_key
    
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        print("OpenAI embeddings initialized successfully")
        
        # Test embedding a simple text
        text = "This is a test sentence for embeddings."
        result = embeddings.embed_query(text)
        print(f"Successfully embedded text. Vector length: {len(result)}")
        print(f"First 5 values: {result[:5]}")
        
        return True
    except Exception as e:
        print(f"Error testing OpenAI embeddings: {e}")
        return False

if __name__ == "__main__":
    success = test_openai_embeddings()
    print(f"Test {'succeeded' if success else 'failed'}")
