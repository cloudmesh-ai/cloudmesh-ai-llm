# Cloudmesh AI LLM - Makefile
# Common operations for managing LLM infrastructure

.PHONY: help install start stop probe clean tunnel check \
        stop-white stop-spark llmctl-launch llmctl-proxy

# Default target
help:
	@echo "Cloudmesh AI LLM - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install Python dependencies and llmctl"
	@echo "  make check        Run setup validation checks"
	@echo ""
	@echo "LiteLLM Proxy:"
	@echo "  make start        Start LiteLLM proxy container"
	@echo "  make stop         Stop LiteLLM proxy container"
	@echo "  make probe        Check health of all model endpoints"
	@echo "  make logs         Show LiteLLM proxy logs"
	@echo ""
	@echo "UVA Kimi Tunnel:"
	@echo "  make tunnel       Start SSH tunnel for UVA Kimi (foreground)"
	@echo "  make tunnel-bg    Start SSH tunnel in background"
	@echo "  make tunnel-stop  Stop SSH tunnel"
	@echo "  make tunnel-status Check tunnel status"
	@echo ""
	@echo "Model Management:"
	@echo "  make launch       Launch a model interactively (llmctl launch)"
	@echo "  make stop-white   Stop vLLM container on white host"
	@echo "  make stop-spark   Stop vLLM container on spark host"
	@echo "  make clean        Stop and remove all containers"
	@echo ""

# Install dependencies and llmctl
install:
	pip install -r requirements.txt
	pip install -e .

# Validation
check:
	llmctl check

# LiteLLM Proxy Management
start:
	llmctl proxy start

stop:
	llmctl proxy stop

probe:
	llmctl proxy probe

logs:
	llmctl proxy logs -f

# SSH Tunnel Management
tunnel:
	@echo "Starting SSH tunnel to UVA GENAI..."
	@echo "Press Ctrl+C to stop"
	llmctl tunnel start

tunnel-bg:
	llmctl tunnel start --background

tunnel-stop:
	llmctl tunnel stop

tunnel-status:
	llmctl tunnel status

# Model Launch
launch:
	llmctl launch

# Stop models on specific hosts
stop-white:
	-ssh white 'docker stop vllm-server && docker rm vllm-server' 2>/dev/null || true

stop-spark:
	-ssh spark 'docker stop vllm-server && docker rm vllm-server' 2>/dev/null || true

# Clean up all containers
clean:
	@echo "Stopping LiteLLM proxy..."
	-llmctl proxy stop 2>/dev/null || true
	@echo "Cleaning up vLLM containers on remote hosts..."
	-ssh white 'docker rm -f vllm-server' 2>/dev/null || true
	-ssh spark 'docker rm -f vllm-server' 2>/dev/null || true
	@echo "Cleanup complete"