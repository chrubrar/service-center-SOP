from chatbot import SOPChatbot
import json

def test_bass_question():
    try:
        # Initialize the chatbot with force_rebuild=True to ensure it picks up our new content
        print("Initializing chatbot with force_rebuild=True...")
        chatbot = SOPChatbot(force_rebuild_vectorstore=True)
        
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
    success = test_bass_question()
    print(f"\nTest {'succeeded' if success else 'failed'}")
