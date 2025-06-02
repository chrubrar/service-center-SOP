#!/bin/bash
# Script to test the simple /ask endpoint of the API

echo "Testing the /ask endpoint (hardcoded responses)..."
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I process a signed BAA?"}'

echo -e "\n\nTesting the API health..."
curl http://localhost:8000/health
