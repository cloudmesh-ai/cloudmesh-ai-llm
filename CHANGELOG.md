# Changelog

All notable changes to `cloudmesh-ai-llm` will be documented in this file.

## [7.0.4.dev1] - 2026-04-26

### Added
- **Tool Installation**: Added `cmc launch install aider` to automate the installation of Aider with built-in Python version validation (3.10-3.12).
- **WebUI Health Polling**: Implemented active polling of the WebUI port before opening the browser to prevent "Internal Server Error" on startup.
- **Robust Error Handling**: Implemented a custom exception hierarchy for vLLM operations to provide clearer failure diagnostics.
- **Advanced Tunnel Management**: Added PID tracking and automatic cleanup for SSH tunnels to prevent orphaned processes.
- **Deep Health Checks**: Enhanced server status monitoring with a multi-stage state machine: `Offline` $\rightarrow$ `Starting` $\rightarrow$ `Ready`.
- **Dynamic Defaulting**: Implemented a first-in-list fallback mechanism for server selection when no specific host is provided.
- **Expanded Test Suite**: Added comprehensive integration tests for the improved launch and tunnel lifecycles.

### Changed
- **Configuration Schema**:
    - Removed `default_host` and `default_service` from `VLLMConfig` in favor of dynamic defaulting.
- **macOS Compatibility**: Updated browser launch mechanism to use `os.system("open ...")` for better reliability on macOS.
    - Changed YAML configuration keys from plural `servers` to singular `server` for consistency.
- **Project Renaming**: Renamed project and command group from `vllm` to `llm` for broader applicability.
- **Command Interface**:
    - Removed `cmc llm set-host`.
    - Updated `cmc llm default` to support specific types: `cmc llm default server [NAME]` and `cmc llm default client [NAME]`.
- **Documentation**: Updated `README.md` to reflect the new command set and configuration format.

### Security
- **History Purge**: Permanently removed sensitive deployment scripts and local configuration directories (`dgx/`, `donttouch/`, `start-claude-4.sh`) from the git history.

## [7.0.3.dev1] - 2026-04-26

### Changed
- **Project Renaming**: Renamed project and command group from `vllm` to `llm` for broader applicability.
- **Configuration Schema**: 
    - Migrated to a unified dictionary-based structure under `cloudmesh.ai.servers`.
    - Introduced `cloudmesh.ai.default` to manage default `server` and `client` settings.
- **Command Interface**:
    - Removed `cmc llm set-host`.
    - Updated `cmc llm default` to support specific types: `cmc llm default server [NAME]` and `cmc llm default client [NAME]`.
- **Documentation**: Updated `README.md` to reflect the new command set and configuration format.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.0.2.dev1] - 2026-04-19

### Added
- **vLLM Management Command**: Introduced a comprehensive set of commands (`launch`, `status`, `kill`, `prompt`) for managing vLLM servers.
- **DGX Optimized Deployment**: 
    - Specialized support for NVIDIA DGX systems using `NVIDIA_VISIBLE_DEVICES`.
    - Support for Tensor Parallelism (TP) across multiple GPUs.
    - Automated Docker-based deployment for isolated environments.
- **Secure Credential Integration**: Implemented secure shell substitution for HuggingFace tokens and API keys.
- **Open WebUI Integration**: Added optional one-click deployment of Open WebUI for a graphical interface to the vLLM server.
- **Professional CLI Output**:
    - Implemented `rich.syntax.Syntax` with `solarized-light` theme for high-quality bash highlighting in dry-runs.
    - Added automatic terminal background color setting (white) via `printf` for better readability of light-themed output.
- **Shell-Safe Export**: Added `--export` functionality that prefixes non-script lines with `# `, allowing the output to be saved directly as a valid shell script.
- **Test Suite**: Added unit and integration tests to verify the launch and management lifecycle.

### Changed
- Initial project structure established to provide a streamlined way to deploy and interact with vLLM on high-performance AI infrastructure.