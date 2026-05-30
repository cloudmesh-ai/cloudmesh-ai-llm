# Cloudmesh AI LLM

Welcome to the Cloudmesh AI LLM documentation. This project provides an orchestration layer for deploying and managing Large Language Model (LLM) backends using vLLM across distributed GPU infrastructure.

## Overview

Cloudmesh AI simplifies the complexity of managing remote GPU nodes, SSH tunneling, and API compatibility. Whether you are deploying on a single RTX 3090 or a massive NVIDIA Spark cluster, our tools provide a consistent interface to start, monitor, and connect to your AI models.

## Quick Start Guide

If you are new to Cloudmesh AI, we recommend following these steps to get started:

1.  **[Installation](./installation.md)**: Set up the package and configure your SSH access.
2.  **[Starting AI Models](./user-manual/starting-services.md)**: Learn how to use `cmc llm start` to launch models like `uva.gemma` and `uva.gemma2`.
3.  **[The Proxy Service](./proxy-service/introduction.md)**: Discover how to use our centralized gateway to access multiple backends and special models like `uva kimmi`.
4.  **[Client Tooling](./client-tools/overview.md)**: Connect professional tools like **Cline**, **Aider**, and **Open WebUI** to your private infrastructure.
5.  **[API Samples](./api-samples/python-examples.md)**: Get started with Python code examples using the OpenAI-compatible API.

## Key Features

- **Automated Orchestration**: Handles SSH tunnels and remote process lifecycles automatically.
- **Hardware Optimized**: Templates and configurations for a wide range of NVIDIA GPUs.
- **API Compatible**: Fully compatible with the OpenAI API specification.
- **Tool Integration**: Seamlessly integrates with modern AI coding assistants.

## Project Links

- **GitHub**: [cloudmesh-ai/cloudmesh-ai-llm](https://github.com/cloudmesh-ai/cloudmesh-ai-llm)
- **Issues**: [Report a bug or request a feature](https://github.com/cloudmesh-ai/cloudmesh-ai-llm/issues)
- **Changelog**: [View recent updates](./CHANGELOG.md)