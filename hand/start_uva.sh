#!/bin/bash
# UVA HPC Start Script

# Load Apptainer module
module load apptainer

# Load tokens from config files
export HF_TOKEN=$(cat $HOME/.config/cloudmesh/llm/HF_token.txt)
export VLLM_API_KEY=$(cat $HOME/.config/cloudmesh/llm/server_master_key.txt)

# Run vLLM using Apptainer in the background
apptainer run --nv \
  -B /scratch/${USER}/hf_cache:/root/.cache/huggingface \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env VLLM_API_KEY="${VLLM_API_KEY}" \
  ${VLLM_IMAGE:-/scratch/${USER}/vllm_gemma4.sif} \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --load-format safetensors \
  --tool-call-parser gemma4 \
  --host 0.0.0.0 \
  --port ${PORT:-18123} &

# Network test: Wait for vLLM to start (listening on 0.0.0.0, testing via localhost)
echo "Waiting for vLLM to start on localhost:${PORT:-18123}..."
until nc -z 127.0.0.1 ${PORT:-18123}; do
  echo -n "."
  sleep 5
done
echo "vLLM is up and running on 0.0.0.0:${PORT:-18123} (verified via localhost)!"

# Bring the server process to the foreground
wait