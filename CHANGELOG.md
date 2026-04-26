# Changelog

All notable changes to `cloudmesh-ai-llm` will be documented in this file.

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