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

## 🚀 Future Roadmap (Proposed Improvements)

### User Experience (UX) Polishing

- [ ] **Interactive Configuration**: Implement a guided interactive wizard for initial setup of `llm.yaml`.
  - The `cmc llm configure` command is basic; a wizard would help users set up their first server without needing to look at the YAML schema.
- [ ] **Wait-for-Ready Indicators**: Add visual spinners/progress bars while polling for model loading.
  - Make the wait feel shorter and more informative during the model loading phase.

### Robustness & Reliability

- [ ] **Automatic Tunnel Recovery**: Background "keep-alive" check to detect and restart dropped tunnels automatically.
  - Implement a check that detects if a tunnel has dropped and restarts it without requiring the user to run `cmc llm start` again.
- [ ] **Graceful Shutdown**: Explicitly close local SSH tunnels and clean up remote temporary files during `cmc llm stop`.
  - Ensure `cmc llm stop` kills the remote process, closes the local SSH tunnel, and cleans up temporary files on the remote node.
- [ ] **Comprehensive Integration Tests**: Expand `tests/` to include end-to-end tests for `monitor stack` (Docker container and API verification).
  - Verify that Docker containers are actually created and the Grafana/Prometheus APIs are reachable.

### Advanced Feature Expansion

- [ ] **Multi-Model Orchestration**: Support for managing multiple concurrent servers with quick context switching.
  - Allow the tool to manage multiple running servers simultaneously and provide a way to "switch" the default server context quickly.
- [ ] **Batch Job Integration**: Implement `cmc llm batch` for non-interactive inference using `batch_job.py`.
  - Fully integrate `batch_job.py` into the CLI for non-interactive inference tasks.
- [ ] **Custom Dashboard Templates**: Allow users to specify custom Grafana JSON files for monitoring.
  - Enable users to specify their own Grafana JSON files instead of relying solely on the built-in `vllm.json`.

### Documentation "Final Mile"

- [x] **Visual Architecture**: Add Mermaid.js diagrams to `docs/ENVIRONMENT.md` to visualize the tunnel and observability flow.
  - Create visual diagrams to explain the flow: `Remote GPU Node` $\rightarrow$ `SSH Tunnel` $\rightarrow$ `Localhost` $\rightarrow$ `Prometheus` $\rightarrow$ `Grafana`.
- [ ] **Expanded API Examples**: Add diverse Python/Node.js implementation examples in `docs/api-samples/`.
  - Show how to use the orchestrator's backends in real-world applications with multiple language examples.

##