# Technical Manual: Deploying Gemma-4-31B on UVA Rivanna

This manual outlines the procedure for deploying a **vLLM** inference server using **Apptainer** on high-performance computing (HPC) resources. This setup utilizes **Local Port Forwarding** to bridge the private compute node API to your local workstation.

## 1. System Architecture & Requirements

Gemma-4-31B (FP16) requires approximately **62GB of VRAM** for model weights alone. To accommodate this on NVIDIA A100 (40GB) nodes, **Tensor Parallelism (TP)** must be used to shard the model across 4 GPUs.

- **Image:** `vllm_gemma4.sif` (Apptainer/Singularity)

- **Hardware:** 4x NVIDIA A100 GPUs (via `ijob` or `sbatch`)

- **Network:** SSH Tunneling via Local Port Forwarding

## 2. Directory Structure

Maintain the following structure in your `/scratch` space for persistence and high-speed I/O:

Plaintext

```         
/scratch/thf2bn/
├── vllm_gemma4.sif           # Apptainer Image
├── gemma.sh                  # Execution script (see Section 4)
├── hf_cache/                 # Persistent model weights storage
└── gemma/
    ├── HF_token.txt          # Hugging Face API Token (Gated access)
    └── server_master_key.txt  # vLLM API Bearer Token
```

## 3. Initial Environment Setup

Run these commands once to prepare the host filesystem:

Bash

```         
# Create the cache directory to prevent re-downloading 60GB on every job
mkdir -p /scratch/thf2bn/hf_cache

# Initialize the keys directory
mkdir -p $HOME/gemma
# Ensure HF_token.txt and server_master_key.txt are populated in $HOME/gemma/
```

## 4. Execution Script (`gemma.sh`)

Create this script in your scratch directory. It automates the environment variable export and the container launch.

Bash

```         
#!/bin/bash

# 1. Load Keys from the secure directory
export HF_TOKEN=$(cat $HOME/gemma/HF_token.txt)
export VLLM_API_KEY=$(cat $HOME/gemma/server_master_key.txt)

# 2. Run Container with 4-way Tensor Parallelism
module load apptainer

apptainer run --nv \
  -B /scratch/thf2bn/hf_cache:/root/.cache/huggingface \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env VLLM_API_KEY="${VLLM_API_KEY}" \
  vllm_gemma4.sif \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --load-format safetensors \
  --tool-call-parser gemma4
```

## 5. Deployment Workflow

### Step A: Request Allocation

Request a 4-GPU interactive session. Replace `udc-an26-7` with your assigned node ID in the following steps.

Bash

```         
ssh -tt uva "/opt/rci/bin/ijob -A bii_dsc_community -p bii-gpu --reservation=bi_fox_dgx --gres=gpu:a100:4 -c 4 --mem=64G"
```

### Step B: Launch Server

Inside the compute node:

Bash

```         
cd /scratch/thf2bn
./gemma.sh
```

Wait for the log: `INFO: Ready for local connections`.

### Step C: Establish the "Local Port Forwarding" Tunnel

The compute nodes are on a private network. To access the API, you must create an SSH tunnel from your **local Mac terminal**.

Bash

```         
# Run this on your local Mac
ssh -L 8000:udc-an26-7:8000 thf2bn@rivanna.itc.virginia.edu
```

## 6. Verification

Test the connection from your **local Mac** using `curl` and `jq` for formatting:

Bash

```         
# Check if the model is loaded
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" | jq

# Send a test inference request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" \
  -d '{
    "model": "google/gemma-4-31B-it",
    "messages": [{"role": "user", "content": "Hello Gemma!"}],
    "temperature": 0.7
  }' | jq
```

## 7. Troubleshooting

- **CUDA Out of Memory:** Verify `nvidia-smi` shows 4 GPUs. Ensure `--tensor-parallel-size` is exactly `4`.

- **Mount source doesn't exist:** Run `mkdir -p /scratch/thf2bn/hf_cache` on the host before launching.

- **Connection Refused:** Verify the tunnel is pointing to the correct `udc-anXX-X` node where the job is active.