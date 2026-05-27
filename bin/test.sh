#!/opt/homebrew/bin/bash

# Ensure the script exits on failure
set -e

clear

echo "================================================================="
echo "        CLUSTER + LITELLM MULTI-MODEL DIAGNOSTIC v2.1            "
echo "================================================================="

# 1. Endpoint definitions: NAME|HOST|PORT|TYPE
# Ensure these match your cluster topology
ENDPOINTS=(
  "white|white|18000|vllm"
  "spark|spark|18001|vllm"
  "litellm|localhost|4000|litellm"
)

# Load Master Key from file
KEY_FILE="$HOME/gemma/server_master_key.txt"
MASTER_KEY="sk-1234"
if [ -f "$KEY_FILE" ]; then
    MASTER_KEY=$(cat "$KEY_FILE" | tr -d '[:space:]')
fi

# 2. Inference Test Function
test_model() {
    local TARGET=$1
    local PORT=$2
    local MODEL_ID=$3
    
    local AUTH="Authorization: Bearer $MASTER_KEY"

    echo -n "      Testing ${MODEL_ID}: "
    
    # Send test request with Auth header
    CHAT_RESPONSE=$(curl -s \
      -X POST "http://${TARGET}:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -H "$AUTH" \
      -d '{
        "model": "'"${MODEL_ID}"'",
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 32,
        "temperature": 0
      }')

    # Check for success (case-insensitive "ok" anywhere in content)
    LOWER_RESPONSE=$(echo "$CHAT_RESPONSE" | tr '[:upper:]' '[:lower:]')
    if [[ "$LOWER_RESPONSE" == *'"content":'*'ok'* ]]; then
        echo -e "\033[0;32mSUCCESS\033[0m"
    else
        echo -e "\033[0;31mFAILED\033[0m"
        echo "      Raw: ${CHAT_RESPONSE:0:100}..."
    fi
}

# 3. Main Loop
for ENTRY in "${ENDPOINTS[@]}"; do
    IFS="|" read -r NAME HOST PORT TYPE <<< "$ENTRY"

    # Resolve hostname
    TARGET=$( [ "$HOST" = "localhost" ] && echo "localhost" || ssh -G "${HOST}" 2>/dev/null | awk '/^hostname / {print $2}' )

    echo -e "\n------------------------------------------------------------------"
    echo "Testing target: ${NAME} (${TARGET}) on Port ${PORT}"
    echo "------------------------------------------------------------------"

    # [1/4] Network check
    echo -n "[1/4] Verifying network path... "
    if nc -z -w3 "${TARGET}" "${PORT}" 2>/dev/null; then
        echo "SUCCESS"
    else
        echo "FAILED" && continue
    fi

    # [2/4] Query model registry
    echo -n "[2/4] Querying model registry... "
    MODELS_RESPONSE=$(curl -s -X GET "http://${TARGET}:${PORT}/v1/models" -H "Authorization: Bearer $MASTER_KEY" --max-time 10)
    
    if [[ "$MODELS_RESPONSE" == *"object"* ]]; then
        echo "SUCCESS"
    else
        echo "FAILED" && continue
    fi

    # [3/4] Discover models (Using Bash 5 mapfile)
    echo "[3/4] Discovering models..."
    # Filter out modelperm- and other metadata IDs
    mapfile -t MODEL_IDS < <(echo "$MODELS_RESPONSE" | grep -o '"id":"[^"]*' | cut -d'"' -f4 | grep -v "modelperm-")
    
    if [ ${#MODEL_IDS[@]} -eq 0 ]; then echo "      No models found."; continue; fi
    echo "      Found ${#MODEL_IDS[@]} models."

    # [4/4] Run inference
    echo "[4/4] Executing live tests:"
    for MODEL in "${MODEL_IDS[@]}"; do
        test_model "${TARGET}" "${PORT}" "${MODEL}"
    done
done

echo -e "\n================================================================="
echo "                    DIAGNOSTIC COMPLETE                         "
echo "================================================================="