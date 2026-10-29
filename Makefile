.PHONY: help clean lint-fix setup test

UV_COMMON_ARGS = --no-active

help:
	echo "See README.md"

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d \( -name '__pycache__' -o -name '.ipynb_checkpoints' \) -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist

lint-fix:
	uv run $(UV_COMMON_ARGS) prek run --all-files

setup:
	uv sync $(UV_COMMON_ARGS) --all-extras
