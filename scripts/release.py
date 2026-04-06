#!/usr/bin/env python3
"""Automate the release process: bump version, update changelog, commit, tag."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/cadance-io/langgraph-events"

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
BUMP_TYPES = ("major", "minor", "patch")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False, cwd=ROOT, **kwargs
    )


def read_current_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "(.+?)"', text, re.MULTILINE)
    if not match:
        print("Error: could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def compute_new_version(current: str, arg: str) -> str:
    major, minor, patch = (int(x) for x in current.split("."))

    if arg in BUMP_TYPES:
        if arg == "major":
            major, minor, patch = major + 1, 0, 0
        elif arg == "minor":
            minor, patch = minor + 1, 0
        else:
            patch += 1
        return f"{major}.{minor}.{patch}"

    if not VERSION_RE.match(arg):
        print(f"Error: '{arg}' is not a valid bump type or version", file=sys.stderr)
        sys.exit(1)

    new_tuple = tuple(int(x) for x in arg.split("."))
    cur_tuple = (major, minor, patch)
    if new_tuple <= cur_tuple:
        print(
            f"Error: new version {arg} must be greater than current {current}",
            file=sys.stderr,
        )
        sys.exit(1)

    return arg


def preflight(new_version: str) -> None:
    errors: list[str] = []

    result = _run(["git", "status", "--porcelain"])
    if result.stdout.strip():
        errors.append("Working tree is not clean. Commit or stash changes first.")

    result = _run(["git", "branch", "--show-current"])
    branch = result.stdout.strip()
    if branch != "main":
        errors.append(f"Must be on main branch (currently on '{branch}').")

    changelog = (ROOT / "CHANGELOG.md").read_text()
    unreleased_match = re.search(
        r"## \[Unreleased\]\n(.*?)(?=\n## \[)", changelog, re.DOTALL
    )
    if not unreleased_match or not unreleased_match.group(1).strip():
        errors.append("[Unreleased] section in CHANGELOG.md is empty.")

    result = _run(["git", "tag", "--list", f"v{new_version}"])
    if result.stdout.strip():
        errors.append(f"Tag v{new_version} already exists.")

    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def bump_version_strings(current: str, new: str, *, dry_run: bool) -> None:
    replacements = [
        (ROOT / "pyproject.toml", f'version = "{current}"', f'version = "{new}"'),
        (ROOT / "README.md", f"Experimental (v{current})", f"Experimental (v{new})"),
        (
            ROOT / "docs" / "index.md",
            f"Experimental (v{current})",
            f"Experimental (v{new})",
        ),
    ]

    for path, old, new_str in replacements:
        text = path.read_text()
        if old not in text:
            print(f"Warning: '{old}' not found in {path.name}", file=sys.stderr)
            continue
        if dry_run:
            print(f"  {path.relative_to(ROOT)}: {old!r} → {new_str!r}", file=sys.stderr)
        else:
            path.write_text(text.replace(old, new_str, 1))


def update_changelog(current: str, new: str, *, dry_run: bool) -> None:
    path = ROOT / "CHANGELOG.md"
    lines = path.read_text().splitlines(keepends=True)
    today = date.today().isoformat()

    # Find [Unreleased] heading and next version heading
    unreleased_idx = None
    next_version_idx = None
    for i, line in enumerate(lines):
        if line.startswith("## [Unreleased]"):
            unreleased_idx = i
        elif (
            unreleased_idx is not None
            and line.startswith("## [")
            and i > unreleased_idx
        ):
            next_version_idx = i
            break

    if unreleased_idx is None or next_version_idx is None:
        print("Error: could not parse CHANGELOG.md structure", file=sys.stderr)
        sys.exit(1)

    # Extract unreleased content (between headings)
    unreleased_content = lines[unreleased_idx + 1 : next_version_idx]

    # Rebuild file
    result_lines = [
        *lines[:unreleased_idx],
        "## [Unreleased]\n",
        "\n",
        f"## [{new}] - {today}\n",
        *unreleased_content,
        *lines[next_version_idx:],
    ]

    # Update footer links
    updated = []
    link_inserted = False
    for line in result_lines:
        if line.startswith("[Unreleased]:"):
            updated.append(f"[Unreleased]: {REPO_URL}/compare/v{new}...HEAD\n")
            if not link_inserted:
                updated.append(f"[{new}]: {REPO_URL}/compare/v{current}...v{new}\n")
                link_inserted = True
        else:
            updated.append(line)

    new_text = "".join(updated)

    if dry_run:
        print(
            f"  CHANGELOG.md: stamp [Unreleased] as [{new}] - {today}", file=sys.stderr
        )
    else:
        path.write_text(new_text)


def sync_lockfile(*, dry_run: bool) -> None:
    if dry_run:
        print("  Would run: uv lock", file=sys.stderr)
    else:
        subprocess.run(["uv", "lock"], check=True, cwd=ROOT)


def commit_and_tag(new: str) -> None:
    _run(
        [
            "git",
            "add",
            "pyproject.toml",
            "README.md",
            "docs/index.md",
            "CHANGELOG.md",
            "uv.lock",
        ],
    )
    subprocess.run(
        ["git", "commit", "-m", f"release: v{new}"],
        check=True,
        cwd=ROOT,
    )
    _run(["git", "tag", f"v{new}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Release langgraph-events")
    parser.add_argument(
        "version",
        help="Bump type (major/minor/patch) or explicit version (X.Y.Z)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    args = parser.parse_args()

    current = read_current_version()
    new = compute_new_version(current, args.version)

    print(f"Releasing: v{current} → v{new}", file=sys.stderr)
    if args.dry_run:
        print("(dry run — no files will be modified)\n", file=sys.stderr)

    preflight(new)

    print("\nVersion bumps:", file=sys.stderr)
    bump_version_strings(current, new, dry_run=args.dry_run)

    print("\nChangelog:", file=sys.stderr)
    update_changelog(current, new, dry_run=args.dry_run)

    print("\nLockfile:", file=sys.stderr)
    sync_lockfile(dry_run=args.dry_run)

    if not args.dry_run:
        print("\nCommitting and tagging...", file=sys.stderr)
        commit_and_tag(new)
        print(
            f"\n✅ Release v{new} is ready. Run:\n\n"
            f"  git push origin main v{new}\n\n"
            f"This will trigger the publish workflow (TestPyPI → PyPI).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
