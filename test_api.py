from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Test API")

class Question(BaseModel):
    text: str

class ChatResponse(BaseModel):
    answer: str
    sources: list

@app.post("/ask", response_model=ChatResponse)
async def ask_question(question: Question):
    try:
        print(f"Received question: {question.text}")
        
        # Create a simple test file first to make sure we can write to the filesystem
        with open("test_file.txt", "w") as f:
            f.write("Test file created\n")
        
        # Return a hardcoded response for testing
        response = {
            "answer": "For signed BAAs, you should add them to the Epic system and update the customer record.",
            "sources": [{"source": "test_sop.txt"}]
        }
        
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Exception in ask_question: {e}")
        print(f"Traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Test API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
