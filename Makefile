.PHONY: help setup doctor config test lint fmt check index review serve eval clean clean-cache

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies and create runtime directories
	uv sync --extra analysis
	@mkdir -p .codesage/cache .codesage/repos reports
	@test -f .env || (cp .env.example .env && echo "Created .env -- add your API keys")

doctor:  ## Check which providers and models are reachable with your keys
	uv run codesage doctor

config:  ## Show effective settings
	uv run codesage config

test:  ## Run the test suite (offline, no API key needed)
	uv run pytest tests/ -q

lint:  ## Lint
	uv run ruff check src/ tests/ evals/

fmt:  ## Format
	uv run ruff format src/ tests/ evals/
	uv run ruff check --fix src/ tests/ evals/

check: lint test  ## Lint and test

index:  ## Deterministic stage only, no API key needed:  make index REPO=.
	uv run codesage index $(REPO)

review:  ## Review a repository:  make review REPO=https://github.com/owner/name
	uv run codesage review $(REPO)

serve:  ## Run the API and dashboard on http://localhost:8000
	uv run uvicorn codesage.api.app:app --reload --port 8000

eval:  ## Run the mutation benchmark and regenerate RESULTS.md
	uv run python -m evals.run_eval

clean:  ## Remove caches and build artefacts (keeps the LLM response cache)
	rm -rf .pytest_cache .ruff_cache dist build **/__pycache__
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

clean-cache:  ## Also drop the LLM response cache -- re-runs will cost quota again
	rm -rf .codesage/cache
