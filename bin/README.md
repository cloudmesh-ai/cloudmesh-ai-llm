# Cloudmesh AI LLM Environment

A unified interface for managing and testing local and remote Large Language Models (LLMs) using vLLM and LiteLLM.

## Project Structure

- `llm.sh`: Interactive selector for launching vLLM engines on remote hosts (`white`, `spark`).
- `start_llm.sh`: **Unified startup script** that manages LiteLLM, SSH tunnels, and backend models.
- `test.sh` / `test.py`: Diagnostic tools to verify connectivity and inference across the cluster.
- `litellm/`: Configuration and docker setup for the LiteLLM Proxy.
  - `config.yaml`: Model routing and master key configuration.
  - `start.py`: Docker management for the LiteLLM container.

## Quick Start

To launch the entire environment (Proxy + Tunnels + Backend):

```bash
./start_llm.sh
```

Follow the prompts to select a backend model and run diagnostics.

## Remote Architecture Map

| Host  | Resource          | VRAM | Primary Model                     | Context | Port  |
|-------|-------------------|------|-----------------------------------|---------|-------|
| white | RTX 3090 (x86)    | 24GB | Qwen 2.5 Coder 32B (AWQ)          | **12k** | 18000 |
| spark | Blackwell (ARM)   | 120G+| DeepSeek R1 Distill 70B (AWQ)      | **32k** | 18001 |

## VS Code Integration (Cline / Continue)

Point your VS Code extension to the LiteLLM Proxy for a unified experience:

- **API Provider:** `OpenAI Compatible`
- **Base URL:** `http://localhost:4000/v1`
- **API Key:** (Contents of `~/gemma/server_master_key.txt`)
- **Active Model IDs:** 
  - `qwen-coder-32b` (Recommended for coding)
  - `deepseek-r1-70b` (Recommended for reasoning)

## Troubleshooting

### SSH Tunnels
LiteLLM requires local SSH tunnels to reach remote backends. The `start_llm.sh` script handles this automatically, but you can manualy open them if needed:
- White: `ssh -L 18000:localhost:18000 white`
- Spark: `ssh -L 18001:localhost:18001 spark`

### Diagnostics
If a model isn't responding, run the diagnostic script:
```bash
./test.sh
```
It will verify the network path, registry availability, and run a live inference test on each endpoint.
