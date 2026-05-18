"""CI gate: ``python -m langgraph_events.serde.migrations``.

Resolves a user graph factory (``module:attr``), diffs its topology
against a committed baseline, and exits non-zero when the baseline
disagrees — so a forgotten ``@migrate_from`` / ``@backfill`` / ``Migration``
fails the build instead of surfacing as a production read error.

Usage::

    python -m langgraph_events.serde.migrations myapp.graph:build baseline.json
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from langgraph_events.serde.migrations.detect import detect_changes

_USAGE = (
    "usage: python -m langgraph_events.serde.migrations "
    "<module:factory> <baseline.json>"
)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print(_USAGE, file=sys.stderr)
        return 2
    factory_path, baseline = args
    module_name, _, attr = factory_path.partition(":")
    if not module_name or not attr:
        print(_USAGE, file=sys.stderr)
        return 2
    factory = getattr(importlib.import_module(module_name), attr)
    graph = factory() if callable(factory) else factory

    report = detect_changes(graph, Path(baseline))
    if report.has_changes():
        print(
            "Graph topology diverges from the baseline — migrations may be missing:",
            file=sys.stderr,
        )
        for label, items in (
            ("added", report.added),
            ("removed", report.removed),
            ("confident renames", report.confident_renames),
            ("ambiguous", report.ambiguous),
            ("unmatched removed", report.unmatched_removed),
        ):
            if items:
                print(f"  {label}: {items}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
