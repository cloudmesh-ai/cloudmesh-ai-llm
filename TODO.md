# vLLM AI LLM TODO List

## 🔴 High Priority

- [ ] **Configuration Validation**: Implement schema validation for `llm.yaml` using a library like `pydantic` or `jsonschema` to catch configuration errors early.
- [x] **Error Handling**: Replace `try...except Exception: pass` blocks with specific exception handling and meaningful logging to avoid silent failures.
- [x] **Consistent Config Access**: Ensure all components use `VLLMConfig` for path resolution instead of hardcoding `~/.config/cloudmesh/llm.yaml`.

## 🟡 Medium Priority

- [x] **Graceful Process Termination**: Update `_kill_port_process` to use `SIGTERM` before `SIGKILL`.
- [x] **Logging Verbosity**: Add a global `--debug` or `--verbose` flag to the `llm` command group to expose raw SSH/Curl commands.
- [x] **Resource Leak Prevention**: Ensure all `subprocess.Popen` instances (especially log tailing) are tracked and terminated in all error paths.
  - [x] Implement context managers or `try...finally` blocks around `subprocess.Popen` to guarantee `.terminate()` or `.kill()` is called.
  - [x] Implement a process tracking registry to identify and kill orphan processes
  - [x] Ensure `stdout` and `stderr` pipes are explicitly closed to prevent file descriptor leaks.
- [x] **Unified Host Resolution**: Centralize logic for resolving `host`, `user`, and `remote_port` to avoid redundancy across `Client`, `Orchestrator`, and `Config`.

## 🔵 Low Priority / Enhancements

- [x] **Configuration Templates**: Provide a set of standard templates for common models (e.g., Gemma, Llama) to simplify initial setup.
  - **Storage**: Store templates as static YAML files within the package (e.g., `src/cloudmesh/ai/vllm/templates/`) to ensure they are distributed with the software.
  - **CLI**: Implement a `cmc llm template` command to list available templates and apply them to the user's `llm.yaml`.
- [x] **Advanced Log Filtering**: Add the ability to filter logs by keyword or level (e.g., `cmc llm logs <name> --grep "error"`).
- [x] **Unified CLI Experience**: Review `VLLMOrchestrator` and `vllm_command` to ensure a consistent interface for all management tasks.
  - [x] Refactor `vllm.py` to use `VLLMOrchestrator` and `VLLMConfig` instead of manual YAML loading and host resolution.
  - [x] Consolidate `llm launch` and `llm start` for client launching to avoid user confusion.
  - [x] Update `llm prompt` to use active server port and model from configuration instead of hardcoded values.
  - [x] Unify `llm status` and `llm info` into a consistent status reporting interface.
- [x] **Unit Testing (Phase 1)**: Increase test coverage for `VLLMConfig.expand_external_references` and `VLLMClient.get_status`.
- [x] **Unit Testing (Phase 2)**:
  - [x] **Process Management**: Test `ProcessRegistry` registration and `terminate_all` effectiveness.
  - [x] **Tunnel Management**: Verify `TunnelManager` state persistence and PID tracking.
  - [x] **Config Resolution**: Test `resolve_server_identity` fallback paths and `merge_env_vars` mapping.
  - [x] **Orchestration Logic**: Mock-based tests for `VLLMOrchestrator.prepare_backend` pipeline.
  - [x] **Client Operations**: Test `get_logs` and `stream_logs` error handling.
  - [x] **Launcher Config**: Verify config resolution in `WebUILauncher`, `AiderLauncher`, and `ClaudeLauncher`.

## Documentation Overhaul (Proposed MkDocs Outline)

Transform the documentation into a comprehensive User Manual focusing on practical deployment and usage.

### 1. User Manual: Starting AI Services

*Goal: Guide users from zero to a running model on specific hardware.*

- **Getting Started with vLLM Services**
  - Overview of the `cmc llm start` command.
  - Step-by-step guide to starting `uva.gemma` and `uva.gemma2`.
- **Hardware-Specific Deployment**
  - **RTX 3090**: Optimization tips and specific start commands for consumer GPUs.
  - **NVIDIA Spark**: Configuration and launch process for Spark-based clusters.
- **Validation & Testing**
  - **Quick Testing with Curl**:
    - Sample `curl` command to verify the `/v1/models` endpoint.
    - Sample `curl` command to send a completion request to the active model.
  - **Using Built-in Tests**:
    - How to run the `ai-llm` test suite to verify backend connectivity and health.

### 2. The Proxy Service

*Goal: Explain how to centralize and manage multiple AI backends.*

- **Introduction to the Proxy Service**
  - What is the proxy service and why use it?
  - Architecture: Client -\> Proxy -\> vLLM Backend.
- **Using Backends via Proxy**
  - Configuring the proxy to route requests to different models.
  - Step-by-step: Using the proxy to access `uva.gemma`.
- **Accessing Special Models**
  - Detailed guide on using `uva kimmi` via the proxy service.

### 3. Client Tooling & Integration

*Goal: Show how to connect external AI tools to the Cloudmesh AI infrastructure.*

- **Client Ecosystem Overview**
  - List of supported tools (Cline, Aider, Open WebUI, etc.).
- **Deep Dive: Cline Setup**
  - **Command Line Setup**: How to configure Cline via the CLI.
  - **Model Selection**: Selecting a specific local model vs. using the proxy service.
  - **Configuration**: Setting the base URL and API keys for seamless integration.

### 4. API Sample Programs

*Goal: Provide copy-pasteable code for developers.*

- **Python Client Examples**
  - Simple chat completion using the `openai` Python library.
  - Streaming responses example.
- **Advanced Integration**
  - Handling timeouts and retries when connecting to remote vLLM instances.