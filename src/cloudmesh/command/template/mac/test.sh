#!/bin/bash

# Configuration
KEY_FILE="../server_master_key.txt"
URL="http://localhost:8001/v1/chat/completions"
MODEL="google/gemma-4-31b-it"

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: $KEY_FILE not found."
    exit 1
fi

# Read key and trim whitespace
API_KEY=$(cat "$KEY_FILE" | xargs)

# Execute request
curl -X POST "$URL" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $API_KEY" \
     -d "{
       \"model\": \"$MODEL\",
       \"messages\": [{\"role\": \"user\", \"content\": \"Hello vLLM, are you there?\"}],
       \"temperature\": 0.7
     }"curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-31b-it",
    "messages": [{"role": "user", "content": "Hello vLLM!"}]
  }' | jq -c
