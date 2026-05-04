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
- `scripts/release.py` — release automation (see below)

## Release

Use `uv run scripts/release.py {major|minor|patch|X.Y.Z}` — do not hand-edit version strings. The script bumps `pyproject.toml`/`README.md`/`docs/index.md`, stamps `[Unreleased]` in `CHANGELOG.md` with today's date, runs `uv lock`, commits as `release: vX.Y.Z`, and tags. Preflight requires a clean working tree on `main` or any `release/*` branch, plus a non-empty `[Unreleased]` section. Add `--dry-run` to preview. After it runs, the final message prints the exact `git push origin <branch> vX.Y.Z` command to run, which triggers the TestPyPI → PyPI publish workflow.

## Conventions

- Python 3.10+, line length 88
- Use `uv` to run all tooling (not bare `python` or `pytest`)
- Ruff for linting and formatting (config in pyproject.toml)
- mypy strict mode
- Tests: `describe_` groups by API surface, `when_` mirrors code branches, `it_` names the assertion. Test each behavior once at the API boundary where it's consumed. Shared event classes in `conftest.py`; scenario-specific events inline. Event classes used as handler type annotations must be defined at module level (not inside `describe_`/`when_` blocks) so Python can resolve forward references at runtime.

## TDD

Iron law: no production code without a failing test first.

1. **Red** — write one failing test for the next behavior.
2. **Verify red** — `uv run pytest tests/path::test` and confirm it fails for the *expected* reason (feature missing, not typo).
3. **Green** — minimal implementation to pass. No extras, no future-proofing.
4. **Verify green** — same test passes; full `uv run pytest tests/` stays green.
5. **Refactor** — clean up while green. No new behavior.

Never write implementation before its test. If you do, delete it and start the cycle properly.
