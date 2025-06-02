import requests
import json

def test_api_bass():
    try:
        # Start the API server in a separate terminal before running this script
        
        # Define the API endpoint
        url = "http://localhost:8000/askLLM"
        
        # Define the question
        data = {
            "text": "what is BASS",
            "use_llm": True
        }
        
        # Make the request
        print(f"Sending request to {url} with data: {json.dumps(data)}")
        response = requests.post(url, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_api_bass()
    print(f"\nTest {'succeeded' if success else 'failed'}")
