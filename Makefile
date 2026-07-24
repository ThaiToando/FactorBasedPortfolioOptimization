.PHONY: help install data backtest sim figures test test-all lint typecheck fingerprint all clean docker

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "%-12s %s\n", $$1, $$2}'

install:         ## Create the locked environment and git hooks
	uv sync --all-extras && uv run pre-commit install

data:            ## Download and cache all inputs (~2 min)
	uv run fbpo fetch-data --config configs/base.yaml

backtest:        ## Run the full backtest (~100 s)
	uv run fbpo backtest --config configs/base.yaml

sim:             ## Monte Carlo estimation-noise study (~3 min)
	uv run fbpo simulate --config configs/simulation.yaml

figures:         ## Regenerate all figures (~25 s)
	uv run fbpo figures

fingerprint:     ## Record the environment fingerprint
	uv run fbpo env-fingerprint

test:            ## Fast test suite
	uv run pytest -n auto -m "not slow"

test-all:        ## Including slow replication tests
	uv run pytest -n auto

lint:            ## Lint and format check
	uv run ruff check . && uv run ruff format --check .

typecheck:       ## Static type check
	uv run mypy src

all: data backtest sim figures  ## Full pipeline from scratch

docker:          ## Build the reproducible image
	docker build -t fbpo:dev .

clean:
	rm -rf data/processed reports/figures/*.png reports/results_*.parquet
