#!/usr/bin/env python3
"""
Script to rebuild the vector store using LayoutLMv3/LayoutXLM for better OCR and document understanding.
This will force rebuild the vector store with the improved document processing capabilities.
"""

import os
import sys
import shutil
from chatbot import SOPChatbot

def main():
    print("Starting vector store rebuild with LayoutLMv3/LayoutXLM...")
    
    # Check if vector store cache exists and backup if it does
    cache_dir = "vector_store_cache"
    if os.path.exists(cache_dir):
        backup_dir = f"{cache_dir}_backup"
        print(f"Backing up existing vector store to {backup_dir}")
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(cache_dir, backup_dir)
    
    try:
        # Initialize the chatbot with force_rebuild=True
        print("Initializing SOPChatbot with force_rebuild=True")
        chatbot = SOPChatbot(force_rebuild_vectorstore=True)
        print("Vector store rebuild completed successfully!")
        return 0
    except Exception as e:
        print(f"Error rebuilding vector store: {e}")
        # Restore backup if rebuild failed
        if os.path.exists(f"{cache_dir}_backup"):
            print("Restoring vector store from backup...")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            shutil.copytree(f"{cache_dir}_backup", cache_dir)
            print("Backup restored.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
