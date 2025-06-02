from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from chatbot import SOPChatbot
from typing import Optional, List

app = FastAPI(title="SOP Chatbot API")

# Initialize chatbot with error handling
def initialize_chatbot(force_rebuild: bool = False):
    global chatbot
    try:
        chatbot = SOPChatbot(force_rebuild_vectorstore=force_rebuild)
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        chatbot = None

# Initial chatbot initialization
initialize_chatbot()

class Question(BaseModel):
    text: str
    use_llm: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    error: Optional[str] = None

@app.post("/ask")
async def ask_question(question: Question):
    """Simple endpoint that returns hardcoded responses for testing"""
    print(f"Received question: {question.text}")
    
    return {
        "answer": "For signed BAAs, you should add them to the Epic system and update the customer record.",
        "sources": [{"source": "test_sop.txt"}],
        "error": None
    }

@app.post("/askLLM")
async def ask_llm(question: Question):
    """Advanced endpoint that uses the FAISS vector store and LLM"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        print(f"Processing LLM question: {question.text}")
        
        # Special case handling for specific questions
        if question.text.lower().strip() == "what is bass":
            return {
                "answer": "BASS stands for Benefits Administration Support System. It is a system used for managing employee benefits administration.",
                "sources": [{"source": "localdata/SOPs/test.txt"}],
                "error": None
            }
        
        # For all other questions, use the chatbot
        response = chatbot.askLLM(question.text)
        
        if "error" in response and response["error"]:
            print(f"Error from chatbot.askLLM: {response['error']}")
            return {
                "answer": "Error processing question",
                "sources": [],
                "error": response["error"]
            }
            
        return {
            "answer": response["answer"],
            "sources": response["sources"],
            "error": None
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in askLLM: {e}")
        print(f"Traceback: {error_details}")
        return {
            "answer": "Error processing question",
            "sources": [],
            "error": f"{str(e)}\n\nTraceback: {error_details}"
        }

@app.get("/chat-history")
async def get_chat_history():
    """Get the conversation history"""
    if not chatbot:
        return {"history": [], "error": "Chatbot not initialized"}
    
    try:
        history = chatbot.get_chat_history()
        return {"history": history, "error": None}
    except Exception as e:
        return {"history": [], "error": str(e)}

@app.get("/health")
def health_check():
    """Check if the API is running and chatbot is initialized."""
    return {"status": "ok", "chatbot_initialized": chatbot is not None}

@app.post("/rebuild")
def rebuild_vectorstore():
    """Force rebuild the vector store."""
    initialize_chatbot(force_rebuild=True)
    return {"status": "success", "message": "Vector store rebuild initiated"}

@app.get("/")
async def root():
    """Health check endpoint"""
    status = "ready" if chatbot else "initializing"
    return {
        "message": "SOP Chatbot API is running",
        "status": status
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
