#!/usr/bin/env python3
"""
Test script to demonstrate the updated chatbot with LayoutLMv3/LayoutXLM integration.
This script will initialize the chatbot and ask a sample question.
"""

import os
import sys
from chatbot import SOPChatbot
import json

def main():
    # Check if we should force rebuild the vector store
    force_rebuild = "--rebuild" in sys.argv
    
    print("Initializing SOPChatbot with LayoutLMv3/LayoutXLM integration...")
    chatbot = SOPChatbot(force_rebuild_vectorstore=force_rebuild)
    
    # Sample questions to test
    test_questions = [
        "What is the process for handling signed BAAs?",
        "How do I process an unsigned BAA?",
        "What is BASS?",
        "What are the steps in the BOR model process?",
        "How do I handle a cancellation in EB service?"
    ]
    
    print("\nTesting chatbot with sample questions:")
    for i, question in enumerate(test_questions):
        print(f"\n{'-' * 80}")
        print(f"Question {i+1}: {question}")
        print(f"{'-' * 80}")
        
        # Get response using the askLLM method
        response = chatbot.askLLM(question)
        
        # Check if there was an error
        if "error" in response and response["error"]:
            print(f"Error: {response['error']}")
            continue
        
        # Print the answer
        print("\nAnswer:")
        print(response["answer"])
        
        # Print the sources
        if "sources" in response and response["sources"]:
            print("\nSources:")
            for source in response["sources"]:
                source_str = source["source"]
                if "page" in source and source["page"]:
                    source_str += f", page {source['page']}"
                print(f"- {source_str}")
    
    print(f"\n{'-' * 80}")
    print("Chat history:")
    chat_history = chatbot.get_chat_history()
    for msg in chat_history:
        role = "User" if msg["type"] == "human" else "Assistant"
        print(f"\n{role}: {msg['content'][:100]}..." if len(msg['content']) > 100 else f"\n{role}: {msg['content']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
