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