# langgraph-events

Event-driven abstraction for LangGraph. State IS events.

## Commands

- **Tests:** `uv run pytest tests/`
- **Lint:** `uv run ruff check src/ tests/`
- **Format:** `uv run ruff format src/ tests/`
- **Type check:** `uv run mypy src/`

## Structure

- `src/langgraph_events/` — library source
- `tests/` — BDD-style with pytest-describe (`describe_`/`when_`/`it_`)
- `examples/` — usage examples

## Conventions

- Python 3.11+, line length 88
- Use `uv` to run all tooling (not bare `python` or `pytest`)
- Ruff for linting and formatting (config in pyproject.toml)
- mypy strict mode
- Tests: `describe_` groups by API surface, `when_` mirrors code branches, `it_` names the assertion. Test each behavior once at the API boundary where it's consumed. Shared event classes in `conftest.py`; scenario-specific events inline.
