.PHONY: run stop logs dev rebuild shell lint fmt test hooks check track clean help

# ============================================================
#  App Users
# ============================================================

run:  ## Start the app
	docker compose up -d --build

stop:  ## Stop the app
	docker compose down

logs:  ## Follow container logs
	docker compose logs -f

# ============================================================
#  Developers
# ============================================================

install:  ## Create venv and install all dependencies (including dev)
	uv sync

dev:  ## Start in development mode (bind mount)
	docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d --build

rebuild:  ## Rebuild without cleaning volumes
	docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d --build

shell:  ## Open a shell inside the running container
	docker exec -it mineral_classifier_container bash

# ============================================================
#  Code Quality (requires local uv)
# ============================================================

lint:  ## Run ruff linter
	uv run ruff check .

fmt:  ## Run ruff formatter
	uv run ruff format .

test:  ## Run tests
	uv run pytest

hooks:  ## Install pre-commit hooks
	uv run pre-commit install

check:  ## Run all pre-commit hooks on all files
	uv run pre-commit run --all-files

# ============================================================
#  DVC
# ============================================================

track:  ## Track results with DVC (inside container)
	docker exec -it mineral_classifier_container bash -c "cd /app && dvc add results/"
	@echo ""
	@echo "Now run locally:"
	@echo "  git add results.dvc .gitignore"
	@echo "  git commit -m Update results"

# ============================================================
#  Cleanup
# ============================================================

clean:  ## Remove containers, volumes, and build cache
	docker compose down -v --rmi local
	docker builder prune -f

# ============================================================
#  Help
# ============================================================

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Create venv and install all dependencies"
	@echo "  run       Start the app"
	@echo "  stop      Stop the app"
	@echo "  logs      Follow container logs"
	@echo "  dev       Start in development mode"
	@echo "  rebuild   Rebuild without cleaning volumes"
	@echo "  shell     Open shell in container"
	@echo "  lint      Run ruff linter"
	@echo "  fmt       Run ruff formatter"
	@echo "  test      Run tests"
	@echo "  hooks     Install pre-commit hooks"
	@echo "  check     Run all pre-commit hooks"
	@echo "  track     Track results with DVC"
	@echo "  clean     Remove containers and volumes"
	@echo "  help      Show this help message"
