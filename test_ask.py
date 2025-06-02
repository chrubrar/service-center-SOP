from chatbot import SOPChatbot
import json

def test_ask():
    try:
        # Initialize the chatbot
        print("Initializing chatbot...")
        chatbot = SOPChatbot()
        
        # Ask a question
        question = "What is the process for signed BAAs?"
        print(f"\nAsking question: {question}")
        
        response = chatbot.ask(question)
        print(f"\nResponse: {json.dumps(response, indent=2)}")
        
        return True
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_ask()
    print(f"\nTest {'succeeded' if success else 'failed'}")
