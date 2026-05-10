# Changelog

All notable changes to `cloudmesh-ai-llm` will be documented in this file.

## [Unreleased]
### Added
- **SQueue Refactoring**: Moved `SQueue` to its own module and added a `cancel` method for Slurm jobs.
- **Email Notifications**: Added support for `--mail-user` and `--mail-type=BEGIN` in UVA sbatch scripts.
- **Robust Job Retrieval**: Implemented pipe-separated format for `squeue` to avoid SSH quoting issues.

## [7.0.4.dev1] - 2026-04-26
### Added
- **Aider Installation**: Automated Aider installation with Python version validation.
- **WebUI Monitoring**: WebUI health polling and enhanced server status monitoring.
- **Tunnel Management**: Advanced tunnel management with PID tracking.
### Changed
- **Project Renaming**: Project renamed from `vllm` to `llm`.
- **Configuration Update**: Updated configuration schema and command interface for better consistency.

## [7.0.3.dev1] - 2026-04-26
### Changed
- **Configuration Migration**: Migrated to a unified dictionary-based configuration structure.
- **Command Update**: Updated `cmc llm default` command.

## [7.0.2.dev1] - 2026-04-19
### Added
- **vLLM Management**: Core vLLM management commands (`launch`, `status`, `kill`, `prompt`).
- **DGX Optimization**: Optimized deployment for NVIDIA DGX systems.
- **WebUI Integration**: Open WebUI integration and professional CLI output.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).