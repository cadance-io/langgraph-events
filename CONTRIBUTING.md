# Contributing

## Development

```bash
uv sync --group dev
uv run pytest tests/
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Releasing

### One-time setup

Before the first release, the repo owner must:

1. **Register trusted publishers on PyPI** — go to [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/) and add a pending trusted publisher:
   - Project name: `langgraph-events`
   - Owner: `cadance-io`
   - Repository: `langgraph-events`
   - Workflow: `publish.yml`
   - Environment: `pypi`

2. **Create GitHub environment** — go to repo Settings → Environments and create `pypi`

### Cutting a release

1. Make sure `CHANGELOG.md` has entries under `[Unreleased]`.

2. Run the release script:

   ```bash
   # Preview changes
   uv run python scripts/release.py minor --dry-run

   # Cut the release (bumps version, stamps changelog, commits, tags)
   uv run python scripts/release.py minor
   ```

   The script accepts `major`, `minor`, `patch`, or an explicit version like `1.0.0`.

3. Push to trigger the publish workflow:

   ```bash
   git push origin main v0.3.0
   ```

   The workflow builds the package and publishes to PyPI.

### What the release script does

`scripts/release.py` automates:

- Bumps version in `pyproject.toml`, `README.md`, `docs/index.md`
- Stamps `[Unreleased]` in `CHANGELOG.md` with the new version and today's date
- Updates changelog footer comparison links
- Runs `uv lock` to sync `uv.lock`
- Commits with message `release: vX.Y.Z`
- Creates git tag `vX.Y.Z`

Preflight checks ensure you're on `main`, working tree is clean, `[Unreleased]` has content, and the tag doesn't already exist.

### Changelog

We use [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Add entries under `[Unreleased]` as you work — the release script moves them to the new version section automatically.
