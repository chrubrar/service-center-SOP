from chatbot import SOPChatbot
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

def test_chatbot():
    # Load environment variables
    load_dotenv(override=True)  # Force override existing env vars
    
    # Verify API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY not found in environment variables')
    print('API key from .env:', api_key)
    print('Current env API key:', os.environ.get('OPENAI_API_KEY'))
    
    # Ensure we're using the correct key
    os.environ['OPENAI_API_KEY'] = api_key
    
    # Create a test document
    os.makedirs("localdata/SOPs", exist_ok=True)
    with open("localdata/SOPs/test.txt", "w") as f:
        f.write("""
        BAA Processing Guide
        
        1. When receiving a BAA:
           - Check if it's signed
           - Verify all fields are complete
           
        2. For signed BAAs:
           - Add to Epic system
           - Update customer record
           
        3. For unsigned BAAs:
           - Send reminder to client
           - Follow up in 48 hours
        """)
    
    # Load and process the test document
    loader = TextLoader("localdata/SOPs/test.txt")
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Create vector store with standard OpenAI settings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # Test various queries
    print("Vector store created successfully!")
    
    def test_query(query):
        print(f"\nQuery: {query}")
        docs = vector_store.similarity_search(query, k=2)  # Get top 2 results
        print("Top 2 relevant documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\nResult {i}:")
            print(doc.page_content)
            print(f"Metadata: {doc.metadata}")
    
    # Test specific queries
    test_query("What is the process for signed BAAs?")
    test_query("What should I do with unsigned BAAs?")
    test_query("What are the initial steps when receiving a BAA?")
    
    # Test vector store operations
    print("\nTesting vector store operations:")
    
    # Test adding new documents
    new_text = """
    BAA Amendment Process:
    1. Review amendment request
    2. Update existing BAA
    3. Get signatures
    4. Update records
    """
    
    print("\nAdding new document about BAA amendments...")
    texts = text_splitter.split_text(new_text)
    vector_store.add_texts(texts)
    
    # Test the new content
    test_query("What is the process for BAA amendments?")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_chatbot()
