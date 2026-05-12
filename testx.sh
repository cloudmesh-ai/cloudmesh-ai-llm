#!/bin/bash

# Default port for the AI Commander server
PORT=18124
# API Key for authentication
API_KEY=$(cat ~/.config/cloudmesh/llm/server_master_key.txt 2>/dev/null)
BASE_URL="http://localhost:$PORT"

echo "Testing AI Commander server at $BASE_URL..."
echo "--------------------------------------------------"

# a) Health Check
echo "Testing Health Check..."
curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health"
if [ $? -eq 0 ]; then
    echo " [OK] Health endpoint responded"
else
    echo " [FAIL] Health endpoint failed"
fi
echo ""

# b) Models Check
echo "Testing Models Endpoint..."
curl -s -H "Authorization: Bearer $API_KEY" "$BASE_URL/v1/models" | jq .
if [ $? -eq 0 ]; then
    echo " [OK] Models endpoint responded"
else
    echo " [FAIL] Models endpoint failed"
fi
echo ""

# c) Test Query
echo "Testing Chat Completion Query..."
curl -s -X POST "$BASE_URL/v1/chat/completions" \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "google/gemma-4-31B-it",
       "messages": [{"role": "user", "content": "Hello, are you running?"}]
     }' | jq .
if [ $? -eq 0 ]; then
    echo " [OK] Query endpoint responded"
else
    echo " [FAIL] Query endpoint failed"
fi

echo "--------------------------------------------------"
echo "Test complete."