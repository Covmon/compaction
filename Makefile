RUN ?= v1
OUT ?= results/

.PHONY: install test lint mock eval report

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

# Offline plumbing test: scripted agent, 3 tasks x 3 strategies, 1 replicate.
# Results are NOT findings. Run name is fixed to "mock" so `make report RUN=mock` works.
mock:
	uv run compaction-eval --mock --tasks 3 --seeds 1 --out $(OUT) --run-name mock

# Real sweep against the Anthropic API (needs ANTHROPIC_API_KEY).
# Explicit flags match the cost estimate in README.md (10 tasks x 3 strategies x 2 replicates).
eval:
	uv run compaction-eval --strategies truncate,summary,indexed --tasks 10 --seeds 2 --out $(OUT) --run-name $(RUN)

report:
	uv run compaction-report --run $(OUT)runs/$(RUN)
