# Setup development environment.
setup:
    uv venv
    uv sync --all-extras --dev

# Install dependencies only.
install:
    uv sync --all-extras --dev

# Clean.
clean:
    rm -rf .venv
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf dist
    rm -rf build
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +

# Testing.
test:
    uv run pytest tests/

# Linting.
lint:
    uv run ruff check .
    uv run ty check .

# Formatting. Note that the docformatter config is defined in pyproject.toml.
# The `|| true` is to ignore the non-zero exit code when no files are found to
# format.
format:
    uv run ruff format .
    uv run docformatter --in-place --recursive . || true 