#!/bin/bash
# UVA HPC Start Script

# Load Apptainer module
module load apptainer

# Run vLLM using Apptainer
apptainer run --nv \
  -B /scratch/${USER}/hf_cache:/root/.cache/huggingface \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env VLLM_API_KEY="${VLLM_API_KEY}" \
  vllm_gemma4.sif \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --load-format safetensors \
  --tool-call-parser gemma4 \
  --host 0.0.0.0 \
  --port 18123