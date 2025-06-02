from chatbot import SOPChatbot
import json

def test_direct_bass():
    try:
        # Initialize the chatbot
        print("Initializing chatbot...")
        chatbot = SOPChatbot()
        
        # Ask the specific question about BASS
        question = "what is BASS"
        print(f"\nAsking question: {question}")
        
        # Use the askLLM method
        response = chatbot.askLLM(question)
        
        # Print the response
        print(f"\nResponse: {json.dumps(response, indent=2)}")
        
        return True
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_direct_bass()
    print(f"\nTest {'succeeded' if success else 'failed'}")
