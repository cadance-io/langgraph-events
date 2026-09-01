# langgraph-events

Event-driven abstraction for LangGraph. State IS events.

## Writing style

CRITICAL: Always prioritise clarity of expression.

Write in ASD-STE100 Simplified Technical English. This applies to every output: chat replies,
commit messages, MR descriptions, issue text, plans, docstrings and code comments.

| Limit | Value |
|---|---|
| Ideas per sentence | 1 |
| Words per instruction / description | 20 / 25 |
| Nouns per cluster (articles, adjectives excluded) | 3 |
| Sentences per paragraph, one topic each | 6 |
| Modals allowed | must, do not, can, recommended |

- An instruction must use active voice. A description can stay passive. Do not invent an actor
  to remove a passive.
- Use simple tenses. Keep the subject, the verb and the article explicit.
- Give each word one meaning and one part of speech. Pick one verb per action, then reuse it.
- Keep a technical term when it is the right term. Define it on first use in each reply. Do not
  reuse a term coined earlier in the session as if it were common vocabulary.
- Do not use slang, metaphor or a contraction.
- A parenthesis or an em-dash carries only a reference or a short aside. A requirement must be
  its own sentence or table row.
- `/` means "or" outside a code span. Do not join clauses with a semicolon.
- Put the condition, and any warning, before the instruction. A warning must state the
  consequence and the avoidance.

Re-read any long-form output against this list before you send it. Long-form output means an MR
description, issue text, a plan or a session summary. Cut what only makes sense from inside the
conversation.

Reason: a long session drifts. The model conditions on its own earlier output, so dense phrasing
compounds until the reader cannot use it.

WARNING: Never trade clarity for brevity. A cut scope qualifier or safety condition makes the
text wrong. Cut words, keep every qualifier and condition.

## Commands

Run all tooling through `uv`. Do not call bare `python` or `pytest`.

- **Tests:** `uv run pytest tests/`
- **Lint:** `uv run ruff check src/ tests/`
- **Format:** `uv run ruff format src/ tests/`
- **Type check:** `uv run mypy src/`

## Structure

- `src/langgraph_events/` — library source
- `tests/` — BDD-style with pytest-describe (`describe_`/`when_`/`it_`)
- `examples/` — usage examples
- `scripts/release.py` — release automation

## Conventions

- Python 3.11+. Line length 88. mypy strict. Ruff lints and formats, configured in
  `pyproject.toml`.
- `describe_` groups by API surface. `when_` mirrors a code branch. `it_` names the assertion.
- Test each behaviour once, at the API boundary where it is consumed.
- Put a shared event class in `conftest.py`. Put a scenario-specific event inline.
- WARNING: Python resolves a forward reference at runtime. An event class used as a handler type
  annotation must be defined at module level, not inside a `describe_` or `when_` block.

## Release

Run `uv run scripts/release.py {major|minor|patch|X.Y.Z}`. Add `--dry-run` to preview.

WARNING: Do not hand-edit a version string. The script owns every one of them.

The script bumps `pyproject.toml`, `README.md` and `docs/index.md`, stamps `[Unreleased]` in
`CHANGELOG.md` with today's date, runs `uv lock`, commits as `release: vX.Y.Z`, then tags.
Preflight requires a clean working tree on `main` or any `release/*` branch, and a non-empty
`[Unreleased]` section. The final message prints the exact `git push origin <branch> vX.Y.Z`
command. That push triggers the TestPyPI to PyPI publish workflow.

## TDD

Iron law: no production code without a failing test first.

1. **Red.** Write one failing test for the next behaviour.
2. **Verify red.** Run `uv run pytest tests/path::test`. Confirm it fails for the expected
   reason. A missing feature is expected. A typo is not.
3. **Green.** Write the minimal implementation that passes. Add no extra. Do not future-proof.
4. **Verify green.** The same test passes. The full `uv run pytest tests/` stays green.
5. **Refactor.** Clean up while green. Add no new behaviour.

WARNING: Never write an implementation before its test. If you do, delete it and start again.
