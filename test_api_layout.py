#!/usr/bin/env python3
"""
Test script for the SOP Chatbot API with LayoutLMv3/LayoutXLM integration.
This script tests both the /ask and /askLLM endpoints.
"""

import requests
import json
import sys
import argparse

def test_api(question, use_llm=True, host="localhost", port=8000):
    """Test the SOP Chatbot API with a question."""
    
    # Determine which endpoint to use
    endpoint = "/askLLM" if use_llm else "/ask"
    
    # Prepare the request
    url = f"http://{host}:{port}{endpoint}"
    headers = {"Content-Type": "application/json"}
    data = {"text": question, "use_llm": use_llm}
    
    print(f"\nSending request to {url}")
    print(f"Question: {question}")
    print(f"Using LLM: {use_llm}")
    
    # Send the request
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            print("\nResponse:")
            print(f"Answer: {result.get('answer', 'No answer provided')}")
            
            # Print sources if available
            sources = result.get('sources', [])
            if sources:
                print("\nSources:")
                for source in sources:
                    print(f"- {source.get('source', 'Unknown')}")
            else:
                print("\nNo sources provided.")
            
            # Print error if available
            error = result.get('error')
            if error:
                print(f"\nError details:")
                print("-" * 50)
                print(error)
                print("-" * 50)
            
            return result
        else:
            print(f"\nError: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"\nError: {e}")
        return None

def test_health(host="localhost", port=8000):
    """Test the health endpoint of the API."""
    url = f"http://{host}:{port}/health"
    
    print(f"\nChecking API health at {url}")
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Chatbot initialized: {result.get('chatbot_initialized', False)}")
            return result
        else:
            print(f"\nError: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"\nError: {e}")
        return None

def test_rebuild(host="localhost", port=8000):
    """Test the rebuild endpoint of the API."""
    url = f"http://{host}:{port}/rebuild"
    
    print(f"\nTriggering vector store rebuild at {url}")
    
    try:
        response = requests.post(url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Message: {result.get('message', 'No message provided')}")
            return result
        else:
            print(f"\nError: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"\nError: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test the SOP Chatbot API")
    parser.add_argument("--question", "-q", default="How do I process an unsigned BAA?",
                        help="Question to ask the chatbot")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use the simple /ask endpoint instead of /askLLM")
    parser.add_argument("--host", default="localhost",
                        help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000,
                        help="API port (default: 8000)")
    parser.add_argument("--health", action="store_true",
                        help="Check API health")
    parser.add_argument("--rebuild", action="store_true",
                        help="Trigger vector store rebuild")
    
    args = parser.parse_args()
    
    # Check health if requested
    if args.health:
        test_health(host=args.host, port=args.port)
        return 0
    
    # Trigger rebuild if requested
    if args.rebuild:
        test_rebuild(host=args.host, port=args.port)
        return 0
    
    # Test the API with the provided question
    test_api(args.question, not args.no_llm, host=args.host, port=args.port)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
