# How to Start Your AI Models

## Quick Start: Launch a Model

If you are a UVA user and want to get a model running immediately, use the following command:

```bash
# Start the standard Gemma model
cmc llm start uva.gemma
```

---

## Before You Start

To ensure a smooth launch, please verify the following:

*   **VPN Connection**: If you are accessing remote GPU clusters (like UVA), ensure you are connected to the institutional VPN.
*   **SSH Access**: You should have SSH access to the target GPU nodes. We recommend using an SSH key for password-less login.
*   **Configuration**: Most users start with the provided defaults. If you need to customize your server, your settings are stored in `~/.config/cloudmesh/llm.yaml`.

## Getting Started

This guide will help you launch Large Language Models (LLMs) on your GPU hardware, from initial setup to a fully running service.

Cloudmesh AI uses the `cmc llm start` command to handle the "heavy lifting." Instead of you having to manually manage remote servers and complex networking, the tool automatically:

1.  **Connects**: It sets up a secure "tunnel" so your laptop can talk to the remote GPU as if it were running locally.
2.  **Deploys**: It uploads the necessary start scripts to the GPU node.
3.  **Manages**: It handles the remote process lifecycles (starting and stopping).
4.  **Verifies**: It polls the model's health to tell you exactly when it's ready for use.

### Which model should I use: Gemma or Gemma 2?

We provide two main versions of the Gemma model on the UVA infrastructure:

*   **`uva.gemma`**: The standard version. It is typically faster and requires fewer resources, making it ideal for simpler tasks or quick testing.
*   **`uva.gemma2`**: The next-generation version. It offers significantly better reasoning capabilities, higher accuracy, and improved performance on complex tasks. **Choose this if you need the highest quality responses.**

To launch your chosen model, run:

```bash
# For the standard version:
cmc llm start uva.gemma

# For the high-performance version:
cmc llm start uva.gemma2
```

The `start` command will:

1. Resolve the server identity from your configuration.
2. Establish an SSH tunnel to the remote GPU node.
3. Launch the vLLM server using the pre-defined template.
4. Poll the health endpoint until the model is ready.

## Hardware-Specific Deployment

Depending on your available hardware, you may need specific configurations to optimize performance.

### RTX 3090 (Consumer GPUs)
For deployments on single RTX 3090 cards, focus on memory utilization and quantization to avoid Out-Of-Memory (OOM) errors.

- **Memory Optimization**: Set `gpu_memory_utilization` to `0.9` or lower in your `llm.yaml`.
- **Quantization**: Use AWQ or GPTQ quantized models to fit larger models into 24GB VRAM.
- **Launch Command**:
  ```bash
  cmc llm start my-rtx3090-server
  ```

### NVIDIA Spark (Cluster/HPC)
When deploying on NVIDIA Spark clusters or high-performance computing environments:

- **Resource Allocation**: Ensure your `allocation.yaml` correctly specifies the number of GPUs.
- **Launch Mode**: Use `sbatch` or `ijob` as configured in the server identity to handle job scheduling.
- **Launch Command**:
  ```bash
  cmc llm start spark-cluster-node
  ```

## Validation & Testing

Once the service is reported as "Ready", you should verify the API is functioning correctly.

### Quick Testing with Curl

Use these commands from your local machine (where the tunnel is active).

**1. Verify Models Endpoint**
Check if the vLLM server is listing the loaded model:
```bash
curl http://localhost:8000/v1/models
```

**2. Send a Completion Request**
Test a simple prompt to ensure the model generates text:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-2b-it",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

### Using Built-in Tests

The `ai-llm` package includes a test suite that can verify the connectivity and health of your backends.

```bash
# Run connectivity tests for the current configuration
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m pytest tests/test_vllm.py
```

---

## Next Steps: Monitoring Your Service

Once your model is running and validated, you should monitor its resource usage and API health to ensure stability. 

For detailed instructions on using Grafana dashboards, checking service status, and streaming logs, please refer to the [Monitoring Guide](monitoring.md).
