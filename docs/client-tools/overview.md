# Client Tooling & Integration

Cloudmesh AI is backend-agnostic and supports any client tool that uses the OpenAI API standard. This allows you to connect coding assistants and UIs to your private GPU clusters.

## Client Ecosystem

Depending on your workflow, you can choose from:

- **Coding Assistants**:
  - **Cline**: A powerful AI agent for VS Code that can read/write files and execute commands.
  - **Aider**: A command-line tool for pair programming with an LLM.
  - **Continue**: An open-source autopilot for VS Code and JetBrains.
- **Web Interfaces**:
  - **Open WebUI**: A feature-rich, self-hosted web interface for interacting with LLMs.
- **CLI Tools**:
  - **Claude Code**: High-performance CLI for agentic coding. _(Note: This requires a paid subscription to enable switching between Plan and Act modes; therefore, full verification is pending. This integration is provided for future upgrades)._

## Cline Setup

Here is how to set up Cline with Cloudmesh AI.

### Command Line Setup

While Cline is a VS Code extension, you can manage your model configurations via the `cmc llm` command line tools to ensure the backend is ready.

1. **Start your desired backend**:
   ```bash
   cmc llm start uva.gemma
   ```
2. **Verify the endpoint**:
   Ensure the tunnel is active and the API is responding at `http://localhost:8000/v1`.

### Model Selection

You have two primary ways to connect Cline to your models:

#### 1. Direct Connection (Single Backend)

Use this when you are focused on one specific GPU node.

*   **API Provider**: `OpenAI Compatible`
*   **Base URL**: `http://localhost:8000/v1`
*   **API Key**: (Can be any string if the backend doesn't require one, or your specific key)
*   **Model ID**: The exact model name (e.g., `google/gemma-2b-it`)

#### 2. Proxy Service Connection (Multi-Backend)

Use this to access multiple models (including `uva kimmi`) without changing settings.

*   **API Provider**: `OpenAI Compatible`
*   **Base URL**: `http://proxy.cloudmesh.ai/v1`
*   **API Key**: Your proxy service API key.
*   **Model ID**: The model name as routed by the proxy (e.g., `gemma`, `kimmi`).

### Configuration Tips for Seamless Integration

To get the best performance from Cline with local vLLM instances:

*   **Context Window**: Match the context window in Cline to the one specified in your vLLM template (e.g., 8192 or 32768).
*   **Stop Sequences**: Ensure stop sequences are configured to prevent the model from hallucinating both sides of a conversation.
*   **System Prompt**: Use a clear system prompt that defines the model's role as a software engineer to improve code quality.
