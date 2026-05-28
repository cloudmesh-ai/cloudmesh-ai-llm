# Environment Configuration and Management

This manual provides a comprehensive guide to managing environment variables using `.env` files for `cloudmesh-ai-llm`. It is designed to serve as both a **User Manual** for daily operations and a **Developer Guide** for technical integration.

---

## User Guide

### Quick Start

To get started with environment variables, follow these three simple steps:

#### 1. Setup your `.env` file
Create a local `.env` file from the provided template:
```bash
cp .env.template .env
# Edit .env with your actual API keys and settings
```

**Recommended `.env` Template:**
```bash
# VLLM / API Configuration
VLLM_API_KEY=your_api_key_here
VLLM_MODEL=gemma
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_MODEL_LEN=4096
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_DTYPE=auto

# Cloudmesh Configuration
CLOUDMESH_AI_USER=your_username
CLOUDMESH_AI_HOST=localhost
CLOUDMESH_AI_PORT=8000
CLOUDMESH_AI_API_KEY=your_api_key_here

# SSH Configuration (for remote deployments)
SSH_HOST=your_remote_host
SSH_USER=your_username
SSH_KEY_PATH=~/.ssh/id_rsa

# Docker Configuration
DOCKER_IMAGE=vllm/vllm-openai:latest
DOCKER_CONTAINER_NAME=vllm-server
DOCKER_NETWORK=host

# Development
DEBUG=false
LOG_LEVEL=INFO
```

#### 2. Load variables in Python
```python
from cloudmesh.ai.vllm.config import VLLMConfig

config = VLLMConfig()
config.merge_env_vars()  # Automatically loads from .env
```

#### 3. Deploy to a remote server
```python
from cloudmesh.ai.vllm.server_uva import UVAServer

server = UVAServer(host="uva")
server.deploy_with_env("gemma", local_env_path=".env")
```

### `cmc env` Command Reference

The `cmc env` suite manages `.env` files across local and remote environments.

| Command | Description | Common Flags | Status |
| :--- | :--- | :--- | :---: |
| `probe [HOST]` | Compare local `.env` with remote | `--format table\|json\|yaml` | ✅ |
| `sync HOST` | Intelligent merge of local into remote | `--strategy local_wins\|remote_wins\|ask` `--dry-run` | ✅ |
| `cp HOST` | Destructive copy local $\rightarrow$ remote | `--force` `--backup` `--no-backup` | ✅ |
| `secure [HOST]` | Set file permissions to `600` | *(None)* | ✅ |
| `validate [HOST]` | Security and permission check | `--fix` | ✅ |
| `edit [HOST]` | Interactive edit (downloads $\rightarrow$ edits $\rightarrow$ uploads) | `-l [LOCAL_FILE]` | ✅ |
| `cat [HOST]` | Display contents with security warning | `-l [LOCAL_FILE]` | ✅ |
| `diff HOST` | Show detailed differences | `--show` (reveals secrets) | ✅ |

### Detailed Usage Examples

#### Comparison and Syncing
```bash
# Compare local .env with UVA server
cmc env probe uva

# Preview a sync without applying changes
cmc env sync uva --dry-run

# Sync with interactive conflict resolution
cmc env sync uva --strategy ask
```

#### Security and Maintenance
```bash
# Secure local .env file (chmod 600)
cmc env secure

# Validate remote file and automatically fix permission issues
cmc env validate uva --fix
```

#### Content Management
```bash
# Edit remote file using local editor
cmc env edit uva

# Display remote .env content on uva
cmc env cat uva
```

---

## Configuration Reference

### Environment Variable Table

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **VLLM Configuration** | | | |
| `VLLM_API_KEY` | string | - | Authentication key for vLLM server |
| `VLLM_MODEL` | string | - | Model identifier (e.g., `gemma`) |
| `VLLM_GPU_MEMORY_UTILIZATION` | float | `0.9` | GPU memory fraction (0.0-1.0) |
| `VLLM_MAX_MODEL_LEN` | int | `4096` | Maximum sequence length |
| `VLLM_TENSOR_PARALLEL_SIZE` | int | `1` | Number of GPUs for parallelism |
| `VLLM_DTYPE` | string | `auto` | Weights data type (`float16`, `bfloat16`) |
| **Cloudmesh Configuration** | | | |
| `CLOUDMESH_AI_USER` | string | - | SSH username |
| `CLOUDMESH_AI_HOST` | string | - | Server hostname/IP |
| `CLOUDMESH_AI_PORT` | int | `8000` | Server port |
| `CLOUDMESH_AI_API_KEY` | string | - | Alternative to `VLLM_API_KEY` |
| **SSH Configuration** | | | |
| `SSH_HOST` | string | - | SSH host alias from `~/.ssh/config` |
| `SSH_USER` | string | - | SSH username |
| `SSH_KEY_PATH` | string | - | Path to private key |
| **Docker Configuration** | | | |
| `DOCKER_IMAGE` | string | - | Docker image name |
| `DOCKER_CONTAINER_NAME` | string | - | Container name |
| `DOCKER_NETWORK` | string | - | Network mode |

### Nested Configuration (Double Underscores)

Environment variables can override deeply nested YAML configuration paths by replacing dots with double underscores (`__`).

**Example Mapping:**
*   **YAML Path:** `cloudmesh.ai.server.uva.gemma.remote_port`
*   **Env Var:** `CLOUDMESH_AI_SERVER__UVA__GEMMA__REMOTE_PORT=8001`

---

## Technical & Developer Guide

### Internal Architecture

The core logic resides in the `EnvManager` class (`src/cloudmesh/ai/vllm/env_manager.py`).

**Key Operational Workflows:**
*   **Intelligent Sync**: Implements a 3-way merge logic to prevent accidental overwrites of remote-only variables.
*   **Security Layer**: Every remote write operation automatically triggers a `chmod 600` to protect secrets.
*   **Secret Masking**: The `cat` and `diff` commands pass content through a masking filter that replaces sensitive patterns (API keys, tokens) with `***`.

### Infrastructure Integration

#### SSH Operations
Remote management is handled via SCP and SSH. The system checks for file existence and current permissions before attempting modifications to ensure minimal disruption.

#### Docker Integration
The `DockerManager` leverages the `--env-file` flag of the Docker CLI.
*   **Runtime**: `docker run --env-file .env ...`
*   **Compose**: `docker-compose --env-file .env up`
*   **Generation**: The system can export the current `VLLMConfig` directly into a Docker-compatible `.env` file.

---

## Best Practices & Security

> [!IMPORTANT]
> **Never commit `.env` files to version control.** Your `.gitignore` is pre-configured to exclude all `.env*` files.

*   **Use Templates**: Always provide a `.env.template` with dummy values for other developers.
*   **Permission Strictness**: Always ensure remote `.env` files are set to `600`. Use `cmc env secure [HOST]` to enforce this.
*   **Environment Separation**: Maintain separate files for different stages:
    *   `.env.local` $\rightarrow$ Local development
    *   `.env.staging` $\rightarrow$ Staging server
    *   `.env.production` $\rightarrow$ Production server

---

## Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| `ImportError: python-dotenv` | Missing dependency | `pip install --force-reinstall cloudmesh-ai-llm` |
| `FileNotFoundError: .env` | Missing file | `cp .env.template .env` |
| Config not overriding YAML | Case mismatch or missing call | Ensure `config.merge_env_vars()` is called after initialization. |
| Remote upload fails | SSH Connectivity | Verify access: `ssh user@remote "echo 'SSH works'"` |

---

## Example Workflows

### Workflow 1: Local Setup
```bash
cp .env.template .env
# Edit .env with your keys
python -c "from cloudmesh.ai.vllm.config import VLLMConfig; c=VLLMConfig(); c.merge_env_vars(); print(c.get('cloudmesh.ai.model'))"
```

### Workflow 2: Multi-Environment Deployment
```python
# deploy.py
import sys
from cloudmesh.ai.vllm.server_uva import UVAServer

env = sys.argv[1] # 'staging' or 'production'
server = UVAServer(host=f"uva-{env}")
server.deploy_with_env("gemma", local_env_path=f".env.{env}")
```

### Workflow 3: CI/CD Integration
In GitHub Actions, use secrets to generate the file on the fly:
```yaml
- name: Create .env file
  run: |
    echo "VLLM_API_KEY=${{ secrets.VLLM_API_KEY }}" >> .env
    echo "VLLM_MODEL=${{ vars.VLLM_MODEL }}" >> .env
```

---

## Summary

The environment system provides a secure, flexible bridge between local development and remote deployment:
1. ✅ **Local**: Easy YAML overrides.
2. ✅ **Remote**: Automatic SSH/SCP sync.
3. ✅ **Docker**: Native environment file support.
4. ✅ **Security**: Secret masking and forced permissions.