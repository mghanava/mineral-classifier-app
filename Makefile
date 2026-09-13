.PHONY: run stop logs dev rebuild shell lint fmt test hooks check track reset abort clean help

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
	uv sync --extra dev

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
#  DVC / Pipeline
# ============================================================

reset:  ## Wipe generated results (keep DVC cache) to restart the pipeline fresh
	docker compose exec -T mineral_classifier sh -c 'find /app/results -mindepth 1 ! -name .gitkeep -delete' || \
	docker run --rm --entrypoint sh -v "$(CURDIR)/results:/results" my_mineral_classifier:latest -c 'find /results -mindepth 1 ! -name .gitkeep -delete'

abort:  ## Abort a running DVC pipeline run (keeps the dashboard up)
	docker cp scripts/abort_pipeline.py mineral_classifier_container:/tmp/abort_pipeline.py
	docker compose exec -T mineral_classifier python /tmp/abort_pipeline.py

track:  ## Track results with DVC (inside container)
	docker exec -it mineral_classifier_container bash -c "cd /app && dvc add results/"
	@echo ""
	@echo "Now run locally:"
	@echo "  git add results.dvc .gitignore"
	@echo "  git commit -m Update results"

# ============================================================
#  Cleanup
# ============================================================

clean:  ## Remove containers, volumes, artifacts, and build cache
	docker compose exec -T mineral_classifier sh -c 'find /app/results /app/.dvc/cache -mindepth 1 ! -name .gitkeep -delete' || \
	docker run --rm --entrypoint sh -v "$(CURDIR)/results:/results" -v "$(CURDIR)/.dvc/cache:/cache" my_mineral_classifier:latest -c 'find /results -mindepth 1 ! -name .gitkeep -delete; find /cache -mindepth 1 -delete'
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
	@echo "  reset     Wipe generated results, keep DVC cache"
	@echo "  abort     Stop a running DVC pipeline run"
	@echo "  clean     Remove containers, volumes, artifacts, and build cache"
	@echo "  help      Show this help message"
