# cloudmesh-ai-llm

`cloudmesh-ai-llm` is a management extension for deploying and interacting with LLM servers, optimized for DGX/A100 systems and HPC environments like UVA Rivanna. It provides a unified interface for server lifecycle management, remote orchestration, and API interaction.

## Quick Start: The "One-Command" Experience

The fastest way to get an LLM backend running and connected to a user interface is using the `launch` orchestrator.

```bash
# Start server, establish tunnel, verify health, and launch WebUI
cmc launch llm gemma-4-31b --ui

# Or launch with Claude Code
cmc launch llm gemma-4-31b --claude
```

**What happens under the hood?**
1. **Health Check**: Checks if the server is already running and tunneled.
2. **Tunneling**: Establishes an SSH tunnel to the remote host.
3. **Server Start**: If not already running, triggers the remote LLM start process.
4. **Verification**: Polls the API until the model is fully loaded and responsive.
5. **Frontend**: Launches the requested UI (WebUI or Claude).

## Prerequisites

Before using the `llm` commands, ensure you have the following:

1.  **Container Runtime**: Docker (with NVIDIA Container Toolkit) for DGX, or Apptainer for UVA.
2.  **Credentials**: The launcher expects two files in your config directory for authentication:
    - `~/.config/cloudmesh/ai/keys/hugging.txt`: Your Hugging Face read token.
    - `~/.config/cloudmesh/ai/keys/server_master_key.txt`: The API key to be used for the LLM server (`VLLM_API_KEY`).

## UVA Rivanna Specific Setup

Deploying on UVA Rivanna requires additional steps due to HPC constraints:

1.  **Working Directory**: All LLM operations, including image storage and model caching, must be performed in your scratch space: `/scratch/$USER`.
2.  **SIF Image**: You must build or obtain the Apptainer `.sif` image (e.g., `vllm_gemma4.sif`) and place it in your scratch directory.
3.  **Model Cache**: Ensure your Hugging Face cache is directed to `/scratch/$USER/hf_cache` to avoid filling up your home quota.

## Installation

Install the package via pip:

```bash
pip install .
```

## Command Reference

### 1. Server Management (`cmc llm`)

Manage LLM servers on remote hosts using named configurations stored in `~/.config/cloudmesh/llm.yaml`.

#### Configuration & Setup
```bash
# Interactively configure default host and settings
cmc llm configure

# Set the default server
cmc llm default server gemma-4-31b

# Set the default client
cmc llm default client default-client

# Reset server configurations to defaults
cmc llm reset
```

#### Lifecycle Control
```bash
# Start a named server (using default host)
cmc llm start gemma-4-31b

# Start a server by interactively selecting from a table
cmc llm start --ui

# Stop a server
cmc llm stop gemma-4-31b

# Forcefully kill a server
cmc llm kill gemma-4-31b

# Check status (performs robust API health check)
cmc llm status gemma-4-31b

# Retrieve remote logs
cmc llm logs gemma-4-31b
```

#### API Interaction
```bash
# Send a quick text prompt
cmc llm prompt "Explain the difference between a transformer and a recurrent neural network."

# Prompt from a file
cmc llm prompt --file my_complex_prompt.txt
```

### 2. Orchestration (`cmc launch`)

The `launch` group provides high-level workflows to get AI tools running.

```bash
# Full pipeline: Start -> Tunnel -> Health Check -> WebUI
cmc launch llm gemma-4-31b --ui

# Full pipeline: Start -> Tunnel -> Health Check -> Claude
cmc launch llm gemma-4-31b --claude

# Launch only the WebUI (assumes backend is already ready)
cmc launch webui

# Launch only Claude (assumes backend is already ready)
cmc launch claude

# Stop a launched tool
cmc launch stop webui
```

## Configuration

### Unified Configuration (`llm.yaml`)
Located at `~/.config/cloudmesh/llm.yaml`. This file contains both the server definitions and the user's default settings.

Example Configuration:
```yaml
config:
  default_host: dgx-node-1
  default_service: gemma-4-31b

cloudmesh:
  ai:
    default:
      server: gemma-4-31b
      client: default-client
    servers:
      gemma-4-31b:
        host: dgx-node-1
        account: "bii_dsc_community"
        partition: "bii-gpu"
        gres: "gpu:a100:4"
        cpus: 4
        mem: "64G"
        image: "/scratch/$USER/vllm_gemma4.sif"
        model: "google/gemma-4-31B-it"
        tensor_parallel_size: 4
        gpu_memory_utilization: 0.90
        max_model_len: 32768
    client: {}
```

## Technical Details

### API Endpoint
By default, the server exposes an OpenAI-compatible API at: `http://127.0.0.1:8000/v1` (via SSH tunnel).

### Remote Execution
The extension uses a unified `RemoteExecutor` to handle SSH and SFTP operations, ensuring consistent behavior across DGX and UVA environments.