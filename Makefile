# MLOps course Stage 0 (ClearML infrastructure). Python env is managed by uv.

SHELL := /bin/bash

SERVER_COMPOSE := infra/clearml-server/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help install server-up server-down server-logs credentials agent smoke dataset clean

help: ## show this help
	@echo "MLOps ClearML Stage 0"
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | sort | awk -F'\t' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## sync the uv environment (.venv)
	uv sync

server-up: ## start the ClearML server stack
	docker compose -f $(SERVER_COMPOSE) up -d
	@echo "web UI: http://localhost:8080 (first boot takes ~1-2 min)"

server-down: ## stop the ClearML server stack
	docker compose -f $(SERVER_COMPOSE) down

server-logs: ## tail server logs
	docker compose -f $(SERVER_COMPOSE) logs -f --tail=50

credentials: ## configure SDK: paste Web UI credentials via clearml-init -> ~/clearml.conf
	uv run clearml-init

agent: ## run the agent on the students queue (foreground)
	bash infra/agent/run-agent.sh

smoke: ## enqueue a smoke task to verify agent execution
	uv run python scripts/smoke_task.py

# --- Stage 1: dataset ---
dataset: ## Stage 1: build IMDB subset + register a versioned ClearML Dataset
	uv run python prepare_data.py
	uv run python create_dataset.py

clean: ## remove the uv environment
	rm -rf .venv
