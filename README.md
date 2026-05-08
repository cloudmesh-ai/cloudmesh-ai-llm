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

Manage LLM servers on remote hosts using named configurations stored in `~/.config/cloudmesh/ai/vllm_servers.yaml`.

#### Configuration & Setup
```bash
# Set the default server (used as fallback if no host is specified)
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

# Start a server with specific GPU devices
cmc llm start gemma-4-31b --device 0,1

# Preview the start command without executing
cmc llm start gemma-4-31b --dryrun

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

Full pipeline: Start -> Tunnel -> Health Check -> WebUI
```bash
cmc launch llm gemma-4-31b --ui
```

Full pipeline: Start -> Tunnel -> Health Check -> Claude
```
cmc launch llm gemma-4-31b --claude
```

Launch only the WebUI (assumes backend is already ready)
```
cmc launch webui
```

Launch only Claude (assumes backend is already ready)
```
cmc launch claude
```


Install AI tools (e.g., aider)
(this is no longer needed as we switchedt to docker, so aider is installed in a container)
```
cmc launch install aider
```

Stop a launched tool
```
cmc launch stop webui
```

## AI Clients

The `launch` command supports several AI clients that can connect to your vLLM backend. These tools allow you to interact with your models via a web browser, a terminal, or an agentic coding environment.

### 1. Open WebUI
Open WebUI is a self-hosted, extensible web interface that provides a ChatGPT-like experience for your local or remote LLMs.

- **Setup**: No local installation required. The launcher automatically manages a Docker container.
- **Usage & Cost**: Because you are hosting the UI and the backend (vLLM) yourself, **there are no per-token charges**. 
- **Account Creation**: When you first launch the UI and navigate to the URL, you will be prompted to create an account. **The first account created becomes the Administrator** of the local instance.
- **Configuration**: Defined under `cloudmesh.ai.client.openwebui` in `llm.yaml`.

**Example YAML:**
```yaml
cloudmesh:
  ai:
    client:
      openwebui:
        OPENAI_API_KEY: "{SERVER_MASTER_KEY}"
        OPENAI_API_BASE: "http://localhost:8001/v1"
        port: 3000
```
- **Run**: `cmc launch webui`

### 2. Claude Code
Claude Code is an agentic CLI tool from Anthropic that can read your files, run commands, and write code directly in your repository.

- **Setup**: Requires local installation via npm:
  ```bash
  npm install -g @anthropic-ai/claude-code
  ```
- **Configuration**: Defined under `cloudmesh.ai.client.claude` in `llm.yaml`. It uses specific environment variables to redirect Claude's requests to your vLLM server.

**Example YAML:**
```yaml
cloudmesh:
  ai:
    client:
      claude:
        OPENAI_API_KEY: "{SERVER_MASTER_KEY}"
        OPENAI_API_BASE: "http://localhost:8001/v1"
        ANTHROPIC_MODEL: "google/gemma-4-31B-it"
        ANTHROPIC_DEFAULT_HAIKU_MODEL: "google/gemma-4-31B-it"
        ANTHROPIC_DEFAULT_SONNET_MODEL: "google/gemma-4-31B-it"
        CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS: 1
        CLAUDE_CODE_ATTRIBUTION_HEADER: 0
```
- **Run**: `cmc launch claude`

### 3. Aider
Aider is an AI pair programming tool that allows you to edit code in your local git repository using an LLM. It is highly efficient at making targeted changes across multiple files.

- **Setup**: Handled automatically by the launcher via Docker to ensure a consistent environment.
- **Configuration**: Defined under `cloudmesh.ai.client.aider` in `llm.yaml`.

**Example YAML:**
```yaml
cloudmesh:
  ai:
    client:
      aider:
        OPENAI_API_KEY: "{SERVER_MASTER_KEY}"
        OPENAI_API_BASE: "http://localhost:8001/v1"
        model: "google/gemma-4-31B-it"
```
- **Run**: `cmc launch aider --docker`


## Key Management & Security

To keep your configuration flexible and secure, `cloudmesh-ai-llm` supports separating your architectural configuration from your sensitive secrets.

### Why Separate Keys?
It is a best practice to keep API keys, passwords, and tokens in a separate file (e.g., `keys.yaml`) rather than directly in `llm.yaml`. This allows you to:
- **Share Configurations**: You can share your `llm.yaml` with colleagues to help them set up their environment without sharing your private keys.
- **Version Control**: You can safely commit `llm.yaml` to a git repository while keeping `keys.yaml` in your `.gitignore`.
- **Centralized Secrets**: Manage all your AI keys in one place for multiple different tools.

### The `#load` Directive
At the top of your `llm.yaml` file, you can use the `#load` directive to import external files:

```yaml
#load: /Users/grey/.config/cloudmesh/keys.yaml
cloudmesh:
  ai:
    ...
```

The launcher reads this directive and merges the contents of the specified file into the in-memory configuration.

### Variable Substitution
Once a file is loaded, you can reference its values using the `{VARIABLE_NAME}` syntax. For example, if your `keys.yaml` contains:
```yaml
SERVER_MASTER_KEY: "sk-1234567890abcdef"
```
You can use it in `llm.yaml` like this:
```yaml
OPENAI_API_KEY: "{SERVER_MASTER_KEY}"
```
The system automatically resolves these placeholders at runtime, ensuring your secrets remain encrypted or isolated in their own secure files.

## Configuration

### Unified Configuration (`vllm_servers.yaml`)

Located at `~/.config/cloudmesh/ai/vllm_servers.yaml`. This file contains both the server definitions and the user's default settings.

Example Configuration:

```yaml
cloudmesh:
  ai:
    default:
      server: gemma-4-31b
      client: default-client
    server:
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