# Credentials

# Sets the background to a light red/pinkish hue
printf '\e]11;#FFE4E1\a'

HF_TOKEN=$(cat ../server_master_key.txt) 
VLLM_API_KEY=$(cat ../server_master_key.txt)

# Optimized Docker Run (Updated to 0.85 for stability)
docker run --gpus '"device=0,1"' \
  --shm-size 16gb \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e VLLM_API_KEY="${VLLM_API_KEY}" \
  vllm/vllm-openai:gemma4-cu130 \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --load-format safetensors
