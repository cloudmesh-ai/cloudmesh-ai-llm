#!/usr/bin/env bash
# start_llm.sh - Unified startup for Cloudmesh AI LLM Environment

set -e

# Load Master Key for diagnostics
KEY_FILE="$HOME/gemma/server_master_key.txt"
MASTER_KEY="sk-1234"
if [ -f "$KEY_FILE" ]; then
    MASTER_KEY=$(cat "$KEY_FILE" | tr -d '[:space:]')
fi

echo "================================================================="
echo "        CLOUDMESH AI LLM - UNIFIED STARTUP                       "
echo "================================================================="

# 1. Start LiteLLM Proxy
echo -e "\n[1/4] Starting LiteLLM Proxy..."
cd litellm
python3 start.py
cd ..

# 2. Establish SSH Tunnels
echo -e "\n[2/4] Setting up SSH Tunnels..."
# White (Port 18000)
if ! nc -z localhost 18000 2>/dev/null; then
    echo "  -> Opening tunnel to white (18000)..."
    ssh -f -N -L 18000:localhost:18000 white
else
    echo "  -> Tunnel to white (18000) already active."
fi

# Spark (Port 18001)
if ! nc -z localhost 18001 2>/dev/null; then
    echo "  -> Opening tunnel to spark (18001)..."
    ssh -f -N -L 18001:localhost:18001 spark
else
    echo "  -> Tunnel to spark (18001) already active."
fi

# 3. Launch Backend Model
echo -e "\n[3/4] Launching Backend Model..."
echo "Running llm.sh selector..."
./llm.sh

# 4. Final Diagnostic
echo -e "\n[4/4] Environment Ready!"
echo "-----------------------------------------------------------------"
echo "LiteLLM Proxy: http://localhost:4000"
echo "Master Key:    $MASTER_KEY"
echo "-----------------------------------------------------------------"
read -p "Run final diagnostic check? [y/N]: " RUN_DIAG
if [[ "$RUN_DIAG" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    ./test.sh
fi

echo -e "\nStartup Sequence Complete."
