# Remote .env File Configuration Guide

This guide explains how to use `.env` files locally and deploy them to remote servers with cloudmesh-ai-llm.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Remote Deployment](#remote-deployment)
- [Docker Integration](#docker-integration)
- [Environment Variable Reference](#environment-variable-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The cloudmesh-ai-llm system supports environment variable configuration through `.env` files. This allows you to:

1. **Keep secrets out of version control** - Store API keys, passwords, and other sensitive data in `.env` files
2. **Easily switch environments** - Use `.env.local`, `.env.production`, etc.
3. **Deploy to remote servers** - Automatically sync `.env` files to remote hosts via SSH
4. **Override YAML configuration** - Environment variables take precedence over YAML config values

---

## Quick Start

### 1. Create your .env file

```bash
cp .env.template .env
# Edit .env with your actual values
```

### 2. Load environment variables in your code

```python
from cloudmesh.ai.vllm.config import VLLMConfig

# Create config and merge .env variables
config = VLLMConfig()
config.merge_env_vars()  # Loads from .env file
```

### 3. Deploy to remote server

```python
from cloudmesh.ai.vllm.server_uva import UVAServer

server = UVAServer(host="uva")
server.deploy_with_env("gemma", local_env_path=".env")
```

---

## Local Development

### Loading .env Files

```python
from cloudmesh.ai.vllm.config import VLLMConfig

# Method 1: Load from default .env file
config = VLLMConfig()
config.merge_env_vars()

# Method 2: Load from specific file
config.merge_env_vars(env_file=".env.local")

# Method 3: Load as dictionary without modifying current config
env_vars = VLLMConfig.load_env_file(".env.production")
print(env_vars)
```

### Available Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VLLM_API_KEY` | API key for vLLM authentication | `sk-abc123` |
| `VLLM_MODEL` | Model name to use | `gemma` or `meta-llama/Meta-Llama-3-8B-Instruct` |
| `VLLM_GPU_MEMORY_UTILIZATION` | GPU memory fraction (0.0-1.0) | `0.9` |
| `VLLM_MAX_MODEL_LEN` | Maximum sequence length | `4096` |
| `VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism | `1` |
| `VLLM_DTYPE` | Data type for model weights | `auto`, `float16`, `bfloat16` |
| `CLOUDMESH_AI_USER` | Username for SSH connections | `your_username` |
| `CLOUDMESH_AI_HOST` | Host for server connection | `localhost` or `remote.server.com` |
| `CLOUDMESH_AI_PORT` | Port for server connection | `8000` |

### Nested Configuration with Double Underscores

Environment variables can override nested YAML configuration using double underscores (`__`):

```bash
# YAML path: cloudmesh.ai.server.uva.gemma.remote_port
CLOUDMESH_AI_SERVER__UVA__GEMMA__REMOTE_PORT=8001

# YAML path: cloudmesh.ai.server.uva.llama3.model
CLOUDMESH_AI_SERVER__UVA__LLAMA3__MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

---

## Remote Deployment

### Upload .env to Remote Server

```python
from cloudmesh.ai.vllm.server_uva import UVAServer

server = UVAServer(host="uva", launch_mode="remote")

# Upload .env file to remote server
success = server.upload_env_file(
    local_env_path=".env",
    remote_env_path="~/.cloudmesh/.env"
)
```

### Run Commands with Environment Variables

```python
# Execute command with sourced env file
result = server.run_with_env(
    command="python -c 'import os; print(os.getenv(\"VLLM_API_KEY\"))'",
    env_path="~/.cloudmesh/.env"
)
print(result.stdout)
```

### Full Deployment with .env

```python
# Deploy server with environment configuration
server.deploy_with_env(
    name="gemma",
    local_env_path=".env",
    sbatch=False
)
```

This will:
1. Upload `.env` to `~/.cloudmesh/.env_gemma` on the remote server
2. Merge environment variables into the configuration
3. Start the vLLM server

---

## Docker Integration

### Using .env with Docker Run

```python
from cloudmesh.ai.vllm.docker_manager import DockerManager

docker = DockerManager()

# Run container with env file
docker.run_with_env_file(
    image="vllm/vllm-openai:latest",
    container_name="vllm-server",
    env_file=".env",
    ports={8000: 8000},
    volumes={"/models": "/models"},
    VLLM_MODEL="gemma"
)
```

### Using .env with Docker Compose

```python
from cloudmesh.ai.vllm.docker_manager import DockerManager

docker = DockerManager()

# Start compose with env file
docker.run_compose_with_env(
    compose_file="docker-compose.yml",
    env_file=".env",
    service="litellm",
    detach=True
)
```

### Generating Docker .env Files

```python
from cloudmesh.ai.vllm.docker_manager import DockerManager
from cloudmesh.ai.vllm.config import VLLMConfig

config = VLLMConfig()
docker = DockerManager()

# Generate .env from current config
docker.generate_docker_env_file(
    config=config.to_dict(),
    output_path="docker/.env"
)
```

---

## Environment Variable Reference

### VLLM Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VLLM_API_KEY` | string | - | Authentication key |
| `VLLM_MODEL` | string | - | Model identifier |
| `VLLM_GPU_MEMORY_UTILIZATION` | float | 0.9 | GPU memory fraction |
| `VLLM_MAX_MODEL_LEN` | int | 4096 | Max sequence length |
| `VLLM_TENSOR_PARALLEL_SIZE` | int | 1 | GPU count for parallelism |
| `VLLM_DTYPE` | string | auto | Model dtype |
| `VLLM_QUANTIZATION` | string | - | Quantization method |
| `VLLM_MAX_NUM_SEQS` | int | 256 | Max concurrent sequences |
| `VLLM_ENFORCE_EAGER` | bool | false | Disable CUDA graph |

### Cloudmesh Configuration

| Variable | Type | Description |
|----------|------|-------------|
| `CLOUDMESH_AI_API_KEY` | string | API key (alternative to VLLM_API_KEY) |
| `CLOUDMESH_AI_HOST` | string | Server hostname/IP |
| `CLOUDMESH_AI_PORT` | int | Server port |
| `CLOUDMESH_AI_USER` | string | SSH username |
| `CLOUDMESH_USER_CONFIG_PATH` | string | Path to YAML config file |

### SSH Configuration

| Variable | Type | Description |
|----------|------|-------------|
| `SSH_HOST` | string | SSH host alias |
| `SSH_USER` | string | SSH username |
| `SSH_KEY_PATH` | string | Path to SSH private key |
| `SSH_PORT` | int | SSH port (default: 22) |

### Docker Configuration

| Variable | Type | Description |
|----------|------|-------------|
| `DOCKER_IMAGE` | string | Docker image name |
| `DOCKER_CONTAINER_NAME` | string | Container name |
| `DOCKER_NETWORK` | string | Network mode |

---

## Best Practices

### 1. Never Commit .env Files

Your `.gitignore` is already configured to exclude `.env` files:

```gitignore
.env
.env.local
.env.development
.env.test
.env.production
.env.*.local
```

### 2. Use .env.template

Always commit `.env.template` with example values (no real secrets):

```bash
cp .env.template .env
# Edit .env with real values
```

### 3. Separate Environments

```bash
.env              # Default (local development)
.env.local        # Local overrides (not committed)
.env.production   # Production settings (not committed)
.env.template     # Template committed to version control
```

### 4. Secure Remote .env Files

After deploying to remote servers:

```bash
ssh user@remote "chmod 600 ~/.cloudmesh/.env"
```

### 5. Python-dotenv

The `python-dotenv` package is included as a required dependency in `pyproject.toml`. It is used automatically when loading environment files.

### 6. Validate Environment Variables

```python
from cloudmesh.ai.vllm.config import VLLMConfig

config = VLLMConfig()
config.merge_env_vars()

# Check required variables
required = ['VLLM_API_KEY', 'VLLM_MODEL']
missing = [var for var in required if not os.getenv(var)]
if missing:
    raise ValueError(f"Missing required env vars: {missing}")
```

---

## Troubleshooting

### Issue: python-dotenv import error

```
ImportError: python-dotenv is required for .env file support.
```

**Solution:**
This should not occur as `python-dotenv` is a required dependency. If you see this error, reinstall the package:

```bash
pip install --force-reinstall cloudmesh-ai-llm
```

### Issue: .env file not found

```
FileNotFoundError: [Errno 2] No such file or directory: '.env'
```

**Solution:**
```bash
# Create from template
cp .env.template .env
# Or specify full path
config.merge_env_vars(env_file="/full/path/to/.env")
```

### Issue: Environment variables not overriding config

**Check:**
1. Variable names match exactly (case-sensitive)
2. Using correct format for nested config (double underscores)
3. `merge_env_vars()` was called after VLLMConfig initialization

```python
config = VLLMConfig()
config.merge_env_vars()  # Must call this!
print(config.get('cloudmesh.ai.api_key'))  # Should show env value
```

### Issue: Remote upload fails

**Check SSH connectivity:**
```bash
ssh user@remote "echo 'SSH works'"
```

**Check file permissions:**
```bash
ls -la .env
# Should be readable by current user
```

**Use absolute paths:**
```python
server.upload_env_file(
    local_env_path=os.path.abspath(".env"),
    remote_env_path="/home/user/.cloudmesh/.env"
)
```

### Issue: Docker container can't read .env

**Ensure .env is in build context:**
```dockerfile
# Dockerfile
COPY .env /app/.env
```

**Or mount at runtime:**
```bash
docker run --env-file .env myimage
```

---

## Example Workflows

### Workflow 1: Local Development with .env

```bash
# 1. Set up environment
cp .env.template .env
# Edit .env with your API keys and settings

# 2. Run locally
python -c "
from cloudmesh.ai.vllm.config import VLLMConfig
config = VLLMConfig()
config.merge_env_vars()
print(f'Model: {config.get(\"cloudmesh.ai.model\")}')
"
```

### Workflow 2: Deploy to Multiple Environments

```python
# deploy.py
import sys
from cloudmesh.ai.vllm.server_uva import UVAServer

environment = sys.argv[1]  # 'staging' or 'production'

server = UVAServer(host=f"uva-{environment}")
server.deploy_with_env(
    name="gemma",
    local_env_path=f".env.{environment}"
)
```

```bash
# Deploy to staging
python deploy.py staging

# Deploy to production
python deploy.py production
```

### Workflow 3: CI/CD with GitHub Secrets

```yaml
# .github/workflows/deploy.yml
name: Deploy

on: [push]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Create .env file
        run: |
          echo "VLLM_API_KEY=${{ secrets.VLLM_API_KEY }}" >> .env
          echo "VLLM_MODEL=${{ vars.VLLM_MODEL }}" >> .env
      
      - name: Deploy to remote
        run: python deploy.py production
```

---

## Summary

The `.env` file support in cloudmesh-ai-llm provides:

1. ✅ **Local config management** - Easy YAML overrides with env vars
2. ✅ **Remote deployment** - Automatic .env file sync via SSH/SCP
3. ✅ **Docker integration** - Native `--env-file` support
4. ✅ **Security** - Keep secrets out of version control
5. ✅ **Flexibility** - Environment-specific configurations

For questions or issues, refer to the main project documentation or open an issue.