# Makefile
.PHONY: start down

# Reads the key, strips whitespace, and exports it for the session
export LITELLM_MASTER_KEY := $(shell cat ~/gemma/server_master_key.txt | tr -d '[:space:]')

start:
	@echo "Starting LiteLLM with key: $${LITELLM_MASTER_KEY:0:4}..."
	docker compose up -d

down:
	docker compose down