# Usage Guide: Model Deployment & Proxy

This guide outlines the management of your local LLM infrastructure, supporting direct model execution via `llm2.py` and unified API access via the LiteLLM proxy (`start2.py`).

---

## Mode 1: Direct Model Launch (`llm2.py`)
Use this mode for development, testing, or when you need specific vLLM configurations.

### Launching a Model
1. Execute the launcher:
   python llm2.py
2. Select a model: The system reads configurations directly from `models.yaml` and displays an interactive table. Enter the numeric ID of your desired model or 'q' to quit.
3. Automated SSH Workflow: The script automatically performs the following actions on the target host:
   - Terminates existing vLLM containers and clears the target port (18000 or 18001).
   - Injects your HuggingFace token from ~/gemma/HF_token.txt.
   - Pulls and runs the appropriate Docker image (Uses vllm/vllm-openai:latest for host white; uses Nvidia's nvcr.io/nvidia/vllm:26.03-py3 for host spark).

### Direct API Access
Once launched, you can query the vLLM backend endpoints directly:

# Verify models available on white

```bash
curl http://white:18000/v1/models
```

# Submit a chat completion request to spark

```bash
curl http://spark:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "casperhansen/deepseek-r1-distill-llama-70b-awq", "messages": [{"role": "user", "content": "Hello"}]}'
```
---

## Mode 2: LiteLLM Proxy (Unified API)
Use this mode for production applications requiring access to multiple models through a single, authenticated endpoint.

### Option A: Python Launcher (Recommended)
The start2.py script manages your proxy life cycle.

# Start the LiteLLM proxy container
```bash
python start2.py
```

# Health Check: Probe all endpoints configured in your config.yaml
python start2.py --probe

### Option B: Alternative Launchers
- Make:
    ```bash
    make start   # Launch proxy
    make down    # Stop proxy
    ```
- Docker Compose:
  ```bash
  export LITELLM_MASTER_KEY=$(cat ~/gemma/server_master_key.txt)
  docker compose up -d
  ```

### Verifying Proxy Operation
Interact with your unified local gateway using your configured master key:

# List all proxy-managed models
```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)"
```
```bash
# Request a chat completion through the proxy routing layer
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" \
  -d '{
    "model": "gemma-4-it-white",
    "messages": [{"role": "user", "content": "Explain quantum computing"}]
  }'
```
---

## Configuration Reference

### Model Registry (models.yaml)
The active runtime configurations mapped in the infrastructure include:


Based on the configuration file provided, here are the updated tables for both hosts, including the requested "Tested" status column.

### Host: white (Port 18000) — Optimized for 24-48GB VRAM

| Model | LiteLLM ID | HuggingFace ID | Context | |
| --- | --- | --- | --- | --- |
| **Gemma 4 Instruct** | `gemma-4-it-white` | `google/gemma-4-e4b-it` | 16,384 | ✅  |
| **Qwen 2.5 Coder 32B** | `qwen-2.5-coder-32b-white` | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | 12,288 | ✅  |
| **Qwen 2.5 32B** | `qwen-2.5-32b-white` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 2,048 | ❌  |
| **Gemma 2 27B** | `gemma-2-27b-white` | `google/gemma-2-27b-it` | 8,192 | ❌  |

---

### Host: spark (Port 18001) — Multi-GPU Datacenter Tier

| Model | LiteLLM ID | HuggingFace ID | Context |  |
| --- | --- | --- | --- | --- |
| **DeepSeek R1 70B** | `deepseek-r1-70b-spark` | `casperhansen/deepseek-r1-distill-llama-70b-awq` | 32,768 | ❌  |
| **Qwen 3.6 MoE A3B Instruct** | `qwen-3.6-moe-a3b-instruct-spark` | `Qwen/Qwen3.6-35B-A3B-Instruct` | 32,768 | ❌  |
| **Qwen 3.6 MoE A3B** | `qwen-3.6-moe-a3b-spark` | `Qwen/Qwen3.6-35B-A3B` | 32,768 | ❌  |
| **Gemma 4 Instruct** | `gemma-4-it-spark` | `google/gemma-4-it` | 16,384 | ❌  |
| **Qwen3 Coder 32B Instruct** | `qwen3-coder-32b-spark` | `Qwen/Qwen3-Coder-32B-Instruct` | 16,384 | ❌  |
| **Llama 3 70B FP4 Instruct** | `llama-3-70b-fp4-spark` | `unsloth/Llama-3-70B-Instruct-quantized-FP4` | 8,192 | ❌  |
| **Llama 3 70B FP8** | `llama-3-70b-fp8-spark` | `neuralmagic/Meta-Llama-3-70B-FP8` | 8,192 | ❌  |
| **Gemma 2 27B** | `gemma-2-27b-spark` | `google/gemma-2-27b-it` | 8,192 | ❌  |
| **Llama 3 8B Instruct** | `llama-3-8b-spark` | `meta-llama/Meta-Llama-3-8B-Instruct` | 4,096 | ❌  |

### LiteLLM Model Naming
When querying via the proxy endpoint, use the structured identifier format: {model-name}-{host} (e.g., gemma-4-it-white, deepseek-r1-70b-spark).

---

## Troubleshooting

### Connection Issues
- Symptom: llm2.py hangs on SSH connectivity or returns connection refused.
- Fixes:
  - Verify key authentication manually: ssh white echo success should return instantly.
  - Confirm Docker daemon status on the target machine: ssh white systemctl status docker.
  - Ensure string resolutions for local hostnames are accurate in /etc/hosts.

### GPU Out of Memory
- Symptom: Container crashes immediately or prints CUDA OOM runtime errors.
- Fixes:
  - Lower the memory parameter inside models.yaml (e.g., scale from 0.95 down to 0.85).
  - Target AWQ, FP4, or FP8 quantized variations instead of standard unquantized precisions.
  - Constrain your KV cache footprint by shortening the maximum context length (max_len).

### Port Conflicts
- Symptom: Process throws an "Address already in use" exception.
- Fixes: llm2.py attempts an automatic fuser -k {port}/tcp kill block. If it fails, clean up the resources on the machine manually:
  ```bash
  ssh white 'docker rm -f vllm-server && fuser -k 18000/tcp'
  ssh spark 'docker rm -f vllm-server && fuser -k 18001/tcp'
  ```
### LiteLLM Routing Failures
- Symptom: API routes return 404 errors or "model not found" from the central proxy.
- Fixes:
  - Call python start2.py --probe to check underlying connectivity across the grid.
  - Cross-examine exact case-sensitive model names against string entries inside config.yaml.

---

## Security Requirements
- Credential Storage: Store HuggingFace tokens and API master files inside the restricted ~/gemma/ path with strict Unix 600 owner read-only permissions.
- Token Handshakes: Secrets are injected directly via environmental scopes within isolated container setups; they are never leaked out to system logs.
- Network Isolation: Core direct ports (18000, 18001) map to public host interfaces. Restrict outside traffic through rigid network firewall profiles and strictly enforce key-based SSH parameters across your cluster.

---

## Quick Reference Card

# 1. Spin up a direct model selection menu
```bash
python llm2.py
```

# 2. Fire up the central proxy router 
```bash
python start2.py
```
# 3. Audit running status across the environment
```bash
python start2.py --probe
```
# 4. Stop environment services safely
```bash
make down
ssh white 'docker stop vllm-server'
ssh spark 'docker stop vllm-server'
```
