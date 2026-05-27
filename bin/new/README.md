# Cloudmesh AI LLM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

This repository manages local LLM infrastructure, supporting direct model execution via `llm2.py`, unified API access via the LiteLLM proxy (`start2.py`), and UVA GENAI Bridge access for remote models.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Installation & Setup](#installation--setup)
4. [Mode 1: Direct Model Launch (`llm2.py`)](#mode-1-direct-model-launch-llm2py)
5. [Mode 2: LiteLLM Proxy (Unified API)](#mode-2-litellm-proxy-unified-api)
6. [Mode 3: UVA GENAI Bridge (Remote Access)](#mode-3-uva-genai-bridge-remote-access)
7. [UVA Kimi via LiteLLM](#uva-kimi-via-litellm)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)
10. [Security Requirements](#security-requirements)
11. [Utility Scripts](#utility-scripts)
12. [Makefile Reference](#makefile-reference)
13. [Architecture Overview](#architecture-overview)
14. [Multi-Model Workflow Example](#multi-model-workflow-example)
15. [Quick Reference Card](#quick-reference-card)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify your setup
make check

# 3. Start the LiteLLM proxy
make start

# 4. Launch a model directly
python llm2.py
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `llm2.py` | Interactive vLLM launcher for remote GPU hosts |
| `start2.py` | LiteLLM proxy manager with health checking |
| `proxy_shim.py` | Request transformer for UVA Kimi compatibility |
| `kimitest.py` | Standalone UVA Kimi API streaming test |
| `check_setup.py` | Environment validation script |
| `models.yaml` | Model registry with deployment configurations |
| `config.yaml` | LiteLLM proxy routing configuration |
| `Makefile` | Common operation shortcuts |
| `requirements.txt` | Python dependencies |

---

## Installation & Setup

### Prerequisites

- **Python 3.8+** with pip
- **Docker** (local and on remote hosts `white` and `spark`)
- **SSH key-based authentication** to hosts `white` and `spark`
- **HuggingFace account** with access token

### 1. Clone and Install

```bash
git clone <repository-url>
cd cloudmesh-ai-llm
pip install -r requirements.txt
```

### 2. Configure SSH Access

Ensure passwordless SSH works to your hosts:

```bash
ssh white echo "Connected to white"
ssh spark echo "Connected to spark"
```

Add to `~/.ssh/config` for easier access:

```ssh-config
Host white
    HostName <white-ip-or-hostname>
    User <username>
    IdentityFile ~/.ssh/id_rsa

Host spark
    HostName <spark-ip-or-hostname>
    User <username>
    IdentityFile ~/.ssh/id_rsa

Host uva
    HostName open-webui.rc.virginia.edu
    User <uva-username>
    LocalForward 8080 open-webui.rc.virginia.edu:443
```

### 3. Setup Secure Credential Storage

Create the secure credentials directory:

```bash
mkdir -p ~/gemma
chmod 700 ~/gemma
```

Create required token files:

```bash
# HuggingFace token (for downloading models)
echo "your_huggingface_token" > ~/gemma/HF_token.txt

# LiteLLM master key (for proxy authentication)
echo "your_secure_random_key" > ~/gemma/server_master_key.txt

# UVA Kimi key (for remote access)
echo "your_uva_kimi_key" > ~/gemma/uva-kimmi-key.txt

# OpenAI API key (optional, for OpenAI model access)
echo "your_openai_key" > ~/gemma/openai_api_key.txt

# Set secure permissions
chmod 600 ~/gemma/*.txt
```

### 4. Verify Setup

Run the validation script:

```bash
make check
```

---

## Mode 1: Direct Model Launch (`llm2.py`)

Use this mode for development, testing, or when you need specific vLLM configurations.

### Launching a Model

1. Execute the launcher:
   ```bash
   python llm2.py
   ```

2. **Select a model:** The system reads configurations from `models.yaml` and displays an interactive table. Enter the numeric ID of your desired model or 'q' to quit.

3. **Automated SSH Workflow:** The script automatically:
   - Terminates existing vLLM containers and clears the target port
   - Injects your HuggingFace token from `~/gemma/HF_token.txt`
   - Pulls and runs the appropriate Docker image:
     - Host `white`: `vllm/vllm-openai:latest`
     - Host `spark`: `nvcr.io/nvidia/vllm:26.03-py3`

### Direct API Access

Once launched, query the vLLM backend directly:

```bash
# Verify models available on white
curl http://white:18000/v1/models

# Submit a chat completion request to spark
curl http://spark:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "casperhansen/deepseek-r1-distill-llama-70b-awq", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Mode 2: LiteLLM Proxy (Unified API)

Use this mode for production applications requiring access to multiple models through a single, authenticated endpoint.

### Option A: Python Launcher (Recommended)

The `start2.py` script manages your proxy lifecycle.

```bash
# Start the LiteLLM proxy container
python start2.py

# Health Check: Probe all endpoints configured in your config.yaml
python start2.py --probe

# Verbose mode for debugging
python start2.py -v
```

### Option B: Make Commands

- **Makefile shortcuts:**
  ```bash
  make start   # Launch proxy
  make stop    # Stop proxy
  make probe   # Check endpoint health
  ```

- **Docker Compose:**
  ```bash
  export LITELLM_MASTER_KEY=$(cat ~/gemma/server_master_key.txt)
  docker compose up -d
  ```

### Verifying Proxy Operation

Interact with your unified local gateway:

```bash
# List all proxy-managed models
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)"

# Request a chat completion through the proxy
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" \
  -d '{
    "model": "gemma-4-it-white",
    "messages": [{"role": "user", "content": "Explain quantum computing"}]
  }'
```

---

## Mode 3: UVA GENAI Bridge (Remote Access)

Access UVA's remote Kimi K2.5 models through an SSH tunnel.

### 1. Establish the Tunnel (Must be active for requests to work)

Run this in a dedicated terminal:
```bash
ssh -L 8080:open-webui.rc.virginia.edu:443 uva
```

> **Tip:** Add to `~/.ssh/config` to simplify:
> ```
> Host uva-genai
>   HostName open-webui.rc.virginia.edu
>   User [your-uva-user]
>   LocalForward 8080 open-webui.rc.virginia.edu:443
> ```
> Then use: `ssh -N uva-genai`

### 2. VS Code / Cline Configuration

| Setting    | Value                                         |
|------------|-----------------------------------------------|
| Provider   | OpenAI Compatible                             |
| Base URL   | `http://localhost:8080/api`                   |
| API Key    | [from `~/gemma/uva-kimmi-key.txt`]            |
| Model ID   | `Kimi K2.5`                                   |

### 3. CLI Automation

Add this to your `.bashrc` or `.zshrc`:
```bash
ask_kimi() {
    curl -ks -X POST "https://localhost:8080/api/chat/completions" \
         -H "Authorization: Bearer $(tr -d '[:space:]' < ~/gemma/uva-kimmi-key.txt)" \
         -H "Content-Type: application/json" \
         -H "Host: open-webui.rc.virginia.edu" \
         -d "{\"model\": \"Kimi K2.5\", \"messages\": [{\"role\": \"user\", \"content\": \"$1\"}]}"
}
```

Usage:
```bash
ask_kimi "What is quantum computing?"
```

### Troubleshooting Quick-Check

| Error                | Solution                                                    |
|----------------------|-------------------------------------------------------------|
| 401 UNAUTHORIZED     | Check key formatting (`tr -d '[:space:]'`)                  |
| CONNECTION REFUSED   | The SSH tunnel terminal is likely closed                    |
| SSL ERROR            | Ensure `-k` flag is used with curl                          |
| 404 NOT FOUND        | Ensure Base URL is `http://localhost:8080/api`              |

---

## UVA Kimi via LiteLLM

Two integration options for accessing UVA Kimi through the LiteLLM proxy.

### Prerequisites for Both Options

1. Set the UVA Kimi key:
   ```bash
   export UVA_KIMI_KEY=$(cat ~/gemma/uva-kimmi-key.txt)
   ```

2. Start the SSH tunnel:
   ```bash
   ssh -L 8080:open-webui.rc.virginia.edu:443 uva
   ```

### Option 1: Direct Integration (simpler)

```bash
# Start LiteLLM proxy
python start2.py

# Test via proxy
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5-uva-direct",
    "messages": [{"role": "user", "content": "Hello from direct integration"}]
  }'
```

### Option 2: Via proxy_shim.py (more robust)

```bash
# In terminal 1: Start the shim proxy
python proxy_shim.py

# In terminal 2: Start LiteLLM
python start2.py

# Test via shim
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5-uva-shim",
    "messages": [{"role": "user", "content": "Hello from shim proxy"}]
  }'
```

### Comparing Options

| Feature | Direct (`kimi-k2.5-uva-direct`) | Via Shim (`kimi-k2.5-uva-shim`) |
|---------|--------------------------------|--------------------------------|
| Setup | Simple - just LiteLLM | Requires running `proxy_shim.py` |
| Header injection | LiteLLM handles it | Shim handles it |
| Payload cleaning | Minimal | More aggressive |
| SSL handling | `ssl_verify: false` | Shim manages validation |
| Error messages | Direct from UVA | Shim can add context |
| Streaming | Supported | Supported |

**Recommendation:** Try Option 1 first. If you encounter 422 or other errors, switch to Option 2.

---

## Configuration Reference

### Model Registry (`models.yaml`)

The active runtime configurations mapped in the infrastructure:

#### Host: white (Port 18000) — Optimized for 24-48GB VRAM

| Model | LiteLLM ID | HuggingFace ID | Context | Tested |
|-------|------------|----------------|---------|--------|
| **Gemma 4 Instruct** | `gemma-4-it-white` | `google/gemma-4-e4b-it` | 16,384 | ✅ |
| **Qwen 2.5 Coder 32B** | `qwen-2.5-coder-32b-white` | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | 12,288 | ✅ |
| **Qwen 2.5 32B** | `qwen-2.5-32b-white` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 2,048 | ❌ |
| **Gemma 2 27B** | `gemma-2-27b-white` | `google/gemma-2-27b-it` | 8,192 | ❌ |

#### Host: spark (Port 18001) — Multi-GPU Datacenter Tier

| Model | LiteLLM ID | HuggingFace ID | Context | Tested |
|-------|------------|----------------|---------|--------|
| **DeepSeek R1 70B** | `deepseek-r1-70b-spark` | `casperhansen/deepseek-r1-distill-llama-70b-awq` | 32,768 | ❌ |
| **Qwen 3.6 MoE A3B Instruct** | `qwen-3.6-moe-a3b-instruct-spark` | `Qwen/Qwen3.6-35B-A3B-Instruct` | 32,768 | ❌ |
| **Qwen 3.6 MoE A3B** | `qwen-3.6-moe-a3b-spark` | `Qwen/Qwen3.6-35B-A3B` | 32,768 | ❌ |
| **Gemma 4 Instruct** | `gemma-4-it-spark` | `google/gemma-4-it` | 16,384 | ❌ |
| **Qwen3 Coder 32B Instruct** | `qwen3-coder-32b-spark` | `Qwen/Qwen3-Coder-32B-Instruct` | 16,384 | ❌ |
| **Llama 3 70B FP4 Instruct** | `llama-3-70b-fp4-spark` | `unsloth/Llama-3-70B-Instruct-quantized-FP4` | 8,192 | ❌ |
| **Llama 3 70B FP8** | `llama-3-70b-fp8-spark` | `neuralmagic/Meta-Llama-3-70B-FP8` | 8,192 | ❌ |
| **Gemma 2 27B** | `gemma-2-27b-spark` | `google/gemma-2-27b-it` | 8,192 | ❌ |
| **Llama 3 8B Instruct** | `llama-3-8b-spark` | `meta-llama/Meta-Llama-3-8B-Instruct` | 4,096 | ❌ |

### UVA Kimi Models (via LiteLLM)

| Model | LiteLLM ID | Description |
|-------|------------|-------------|
| **Kimi K2.5 (Direct)** | `kimi-k2.5-uva-direct` | Direct via SSH tunnel |
| **Kimi K2.5 (Shim)** | `kimi-k2.5-uva-shim` | Via `proxy_shim.py` |

### LiteLLM Model Naming

When querying via the proxy endpoint, use the structured identifier format: `{model-name}-{host}` (e.g., `gemma-4-it-white`, `deepseek-r1-70b-spark`).

---

## Troubleshooting

### Connection Issues

**Symptom:** `llm2.py` hangs on SSH connectivity or returns connection refused.

**Fixes:**
- Verify key authentication manually: `ssh white echo success` should return instantly.
- Confirm Docker daemon status on the target machine: `ssh white systemctl status docker`
- Ensure hostname resolutions for local hostnames are accurate in `/etc/hosts`

### GPU Out of Memory

**Symptom:** Container crashes immediately or prints CUDA OOM runtime errors.

**Fixes:**
- Lower the memory parameter inside `models.yaml` (e.g., scale from 0.95 down to 0.85)
- Target AWQ, FP4, or FP8 quantized variations instead of standard unquantized precisions
- Constrain your KV cache footprint by shortening the maximum context length (`max_len`)

### Port Conflicts

**Symptom:** Process throws an "Address already in use" exception.

**Fixes:** `llm2.py` attempts an automatic `fuser -k {port}/tcp` kill block. If it fails, clean up manually:

```bash
ssh white 'docker rm -f vllm-server && fuser -k 18000/tcp'
ssh spark 'docker rm -f vllm-server && fuser -k 18001/tcp'
```

### LiteLLM Routing Failures

**Symptom:** API routes return 404 errors or "model not found" from the central proxy.

**Fixes:**
- Call `python start2.py --probe` to check underlying connectivity
- Cross-examine exact case-sensitive model names against entries inside `config.yaml`

---

## Security Requirements

- **Credential Storage:** Store HuggingFace tokens and API master files inside the restricted `~/gemma/` path with strict Unix 600 owner read-only permissions
- **Token Handshakes:** Secrets are injected directly via environmental scopes within isolated container setups; they are never leaked out to system logs
- **Network Isolation:** Core direct ports (18000, 18001) map to public host interfaces. Restrict outside traffic through network firewall profiles and strictly enforce key-based SSH parameters
- **Secret File Permissions:**
  ```bash
  chmod 600 ~/gemma/*.txt
  ```

---

## Utility Scripts

### `llm2.py` - Direct Model Launcher

Interactive model launcher that deploys vLLM containers on remote GPU hosts.

**Features:**
- Interactive table-based model selection from `models.yaml`
- Automated SSH workflow: kills existing containers, clears ports, pulls images, starts vLLM
- Host-specific Docker image selection
- Automatic HuggingFace token injection from `~/gemma/HF_token.txt`

**Usage:**
```bash
python llm2.py
```

### `start2.py` - LiteLLM Proxy Manager

Manages the LiteLLM proxy container that provides unified API access to all models.

**Features:**
- Loads environment variables from `~/gemma/` credential files automatically
- Starts LiteLLM with proper configuration from `config.yaml`
- Health check (`--probe`) validates all configured endpoints
- Verbose mode (`-v`) for debugging

**Usage:**
```bash
# Start the LiteLLM proxy
python start2.py

# Probe all configured model endpoints
python start2.py --probe

# Verbose mode
python start2.py -v
```

### `proxy_shim.py` - UVA Kimi Proxy Shim

Local proxy server that forwards requests to UVA GENAI with proper header and payload transformation.

**Why use it:**
- Handles required `Host` header injection
- Removes problematic fields from request payloads (prevents 422 errors)
- Manages SSL certificate validation issues
- Provides better error context

**Usage:**
```bash
# Start the shim (runs on localhost:8081)
python proxy_shim.py
```

### `kimitest.py` - UVA Kimi Streaming Test

Standalone script for testing streaming responses from UVA Kimi K2.5 API.

**Features:**
- Reads API key from `~/gemma/uva-kimmi-key.txt`
- Demonstrates proper SSE (Server-Sent Events) streaming parsing
- Shows incremental chunk handling

**Usage:**
```bash
# Ensure SSH tunnel is active
ssh -L 8080:open-webui.rc.virginia.edu:443 uva

# Run the test
python kimitest.py
```

### `check_setup.py` - Environment Validator

Validates your environment configuration before running models.

**Checks performed:**
- SSH connectivity to `white` and `spark` hosts
- Docker daemon status on remote hosts
- Required credential files in `~/gemma/`
- Proper file permissions (600) on credential files
- Hostname resolution for `white` and `spark`

**Usage:**
```bash
make check
# or
python check_setup.py
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make help` | Display available commands |
| `make install` | Install Python dependencies from `requirements.txt` |
| `make start` | Start the LiteLLM proxy container |
| `make stop` | Stop the LiteLLM proxy container |
| `make probe` | Health check all model endpoints |
| `make check` | Run setup validation |
| `make tunnel` | Start SSH tunnel for UVA Kimi (runs in foreground) |
| `make clean` | Stop and remove all containers (LiteLLM + vLLM) |
| `make stop-white` | Stop vLLM container on `white` host only |
| `make stop-spark` | Stop vLLM container on `spark` host only |

**Example workflow:**
```bash
# Full environment shutdown
make clean

# Or stop individual components
make stop          # Stop LiteLLM proxy
make stop-white    # Stop model on white
make stop-spark    # Stop model on spark
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOUR LOCAL MACHINE                                │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   llm2.py   │    │  start2.py  │    │proxy_shim.py│    │  kimitest.py│  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └─────────────┘  │
│         │                  │                  │                             │
│         │ SSH              │ Docker           │ HTTP                        │
│         │                  │                  │                             │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐                     │
│  │  white:22   │    │  LiteLLM    │◄───┤localhost:8081│                    │
│  │  spark:22   │    │  Proxy:4000 │    │  (shim)      │                    │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘                     │
│         │                  │                            ┌─────────────────┐ │
│         │                  └────────────────────────────►│  SSH Tunnel     │ │
│         │                                               │  localhost:8080 │ │
└─────────┼───────────────────────────────────────────────┴─────────────────┘ │
          │                                                                   │
          │                              ┌──────────────────────────────────┘
          │                              │
┌─────────▼──────────┐      ┌────────────▼─────────────┐
│  ┌──────────────┐  │      │  ┌─────────────────────┐ │
│  │white:18000   │  │      │  │open-webui.rc.virginia│ │
│  │vLLM Container│  │      │  │.edu:443              │ │
│  └──────────────┘  │      │  │UVA GENAI Kimi K2.5   │ │
│                    │      │  └─────────────────────┘ │
│  ┌──────────────┐  │      └──────────────────────────┘
│  │spark:18001   │  │
│  │vLLM Container│  │
│  └──────────────┘  │
└────────────────────┘

DATA FLOW:
1. llm2.py ──SSH──► white/spark hosts ──► vLLM containers (direct model access)
2. start2.py ──Docker──► LiteLLM Proxy (localhost:4000) ──► routes to white/spark
3. proxy_shim.py ──HTTP──► SSH Tunnel ──► UVA GENAI (Kimi K2.5)
```

### Component Interaction

| Component | Purpose | Protocol | Port |
|-----------|---------|----------|------|
| `llm2.py` | Deploy models to GPU hosts | SSH | 22 |
| `start2.py` | Manage LiteLLM proxy | Docker API | - |
| LiteLLM Proxy | Unified API gateway | HTTP | 4000 |
| vLLM (white) | Model inference API | HTTP | 18000 |
| vLLM (spark) | Model inference API | HTTP | 18001 |
| `proxy_shim.py` | UVA request transformer | HTTP | 8081 |
| SSH Tunnel | Secure UVA connection | SSH | 8080 |

---

## Multi-Model Workflow Example

This example demonstrates starting models on both `white` and `spark`, then accessing them through the LiteLLM proxy.

### Step 1: Start Model on White Host

```bash
# Terminal 1: Launch a model on white
python llm2.py

# Select: Gemma 4 Instruct (ID: 0)
# This deploys to white:18000
```

### Step 2: Start Model on Spark Host

```bash
# Terminal 2: Launch a model on spark
python llm2.py

# Select: DeepSeek R1 70B (ID: 4)
# This deploys to spark:18001
```

### Step 3: Start LiteLLM Proxy

```bash
# Terminal 3: Start the proxy
make start

# Output: LiteLLM proxy running on http://localhost:4000
```

### Step 4: Verify Both Endpoints

```bash
# List all available models
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)"

# Should show: gemma-4-it-white, deepseek-r1-70b-spark, etc.
```

### Step 5: Test Each Model

```bash
# Test model on white (Gemma 4)
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-it-white",
    "messages": [{"role": "user", "content": "Explain Docker containers"}]
  }'

# Test model on spark (DeepSeek)
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-70b-spark",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

### Step 6: Shutdown

```bash
# Stop LiteLLM proxy
make stop

# Stop models on remote hosts
make stop-white
make stop-spark

# Or clean everything at once
make clean
```

---

## Quick Reference Card

```bash
# ═════════════════════════════════════════════════════════════════
# SETUP & VALIDATION
# ═════════════════════════════════════════════════════════════════

# Verify your environment is ready
make check

# Install dependencies
make install

# ═════════════════════════════════════════════════════════════════
# SINGLE MODEL (Direct Mode)
# ═════════════════════════════════════════════════════════════════

# Launch and select model interactively
python llm2.py

# ═════════════════════════════════════════════════════════════════
# MULTI-MODEL (Proxy Mode)
# ═════════════════════════════════════════════════════════════════

# Terminal 1: Start model on white
python llm2.py  # Select model for white

# Terminal 2: Start model on spark  
python llm2.py  # Select model for spark

# Terminal 3: Start proxy
make start

# Verify all endpoints
make probe

# ═════════════════════════════════════════════════════════════════
# UVA KIMI ACCESS
# ═════════════════════════════════════════════════════════════════

# Start SSH tunnel (keep running)
make tunnel
# or: ssh -L 8080:open-webui.rc.virginia.edu:443 uva

# Test direct access
python kimitest.py

# Test via LiteLLM (with proxy running)
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/gemma/server_master_key.txt)" \
  -d '{"model": "kimi-k2.5-uva-direct", "messages": [{"role": "user", "content": "Hello"}]}'

# ═════════════════════════════════════════════════════════════════
# SHUTDOWN
# ═════════════════════════════════════════════════════════════════

# Stop proxy only
make stop

# Stop proxy + all models
make clean

# Stop individual model hosts
make stop-white
make stop-spark
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.