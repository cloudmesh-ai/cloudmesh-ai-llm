#!/bin/bash

printf '\e]11;#F0F8FF\a'

# Define constants
CONTAINER_NAME="open-webui"
LOCAL_TUNNEL_PORT=8001  # This is the port where your SSH tunnel 'lands' on your Mac/PC
WEBUI_PORT=3000
HF_TOKEN=$(cat ../server_master_key.txt) 
VLLM_API_KEY=$(cat ../server_master_key.txt)

# --- Color Codes ---
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check Docker status
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}ERROR: Docker is not running or could not be found.${NC}"
        echo "Please start the Docker Desktop application and try again."
        exit 1
    fi
}

# Run the check
check_docker

echo "Stopping existing container if it exists..."
docker stop $CONTAINER_NAME 2>/dev/null && docker rm $CONTAINER_NAME 2>/dev/null

echo "Launching Open WebUI..."
docker run -d \
  -p ${WEBUI_PORT}:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:${LOCAL_TUNNEL_PORT}/v1" \
  -e OPENAI_API_KEY="${VLLM_API_KEY}" \
  -e WEBUI_NAME="Gregors Gemma 4 Portal" \
  --name $CONTAINER_NAME \
  --restart always \
  ghcr.io/open-webui/open-webui:main

echo "------------------------------------------------"
echo "Setup Complete!"
echo "1. Ensure 'ssh gemma-server' is running in a terminal."
echo "2. Access the UI at: http://localhost:3000"
echo "------------------------------------------------"
