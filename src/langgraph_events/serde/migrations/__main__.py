"""The migration CLI: ``python -m langgraph_events.serde.migrations``.

``<target>`` is the ``module:attr`` of a ``MigrationTarget``. Exit 0 is a
pass or a done write, 1 is a finding or a store that is not clean, 2 is a
usage or target error, 3 is a ``rewrite`` under the wrong release.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from langgraph_events.serde.migrations._baseline import (
    Baseline,
    MigrationsUnrecorded,
)
from langgraph_events.serde.migrations._gate import (
    PROGRAM,
    TARGET_PLACEHOLDER,
    Invocation,
    PreviousRequiredError,
    check,
    hold_migration,
    update_baseline,
)
from langgraph_events.serde.migrations._live_set import WrongRelease, rewrite_live_set
from langgraph_events.serde.migrations._target import MigrationTarget, TargetError

_SUBCOMMANDS = ("check", "baseline", "fingerprint", "rewrite")
_WRONG_RELEASE = 3


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2 and args[0] not in _SUBCOMMANDS and not args[0].startswith("-"):
        print(
            f"The form '<module:factory> <baseline>' was removed. Declare a "
            f"MigrationTarget(graph=..., baseline=Path({args[1]!r})) and run:\n"
            f"{PROGRAM} check {TARGET_PLACEHOLDER}",
            file=sys.stderr,
        )
        return 2
    parser = _parser()
    try:
        options = parser.parse_args(args)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 2
    try:
        return int(options.run(options))
    except (TargetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM)
    commands = parser.add_subparsers(required=True, metavar="command")

    check_ = commands.add_parser("check", help="run the migration gate")
    check_.add_argument("target")
    check_.add_argument("--previous", type=Path)
    check_.set_defaults(run=_check)

    baseline = commands.add_parser("baseline", help="write the baseline")
    actions = baseline.add_subparsers(required=True, metavar="action")
    update = actions.add_parser("update", help="write the baseline the code needs")
    update.add_argument("target")
    update.add_argument("--previous", type=Path)
    update.set_defaults(run=_update)
    hold = actions.add_parser("hold", help="keep a shipped migration one release")
    hold.add_argument("target")
    hold.add_argument("label")
    hold.add_argument("--reason", required=True)
    hold.add_argument("--previous", type=Path, required=True)
    hold.set_defaults(run=_hold)

    fingerprint = commands.add_parser(
        "fingerprint", help="print the fingerprint of a baseline file"
    )
    fingerprint.add_argument("baseline", type=Path)
    fingerprint.set_defaults(run=_fingerprint)

    rewrite = commands.add_parser(
        "rewrite", help="rewrite the live set of the store until it reads clean"
    )
    rewrite.add_argument("target")
    rewrite.add_argument("--expect", required=True)
    rewrite.add_argument("--apply", action="store_true")
    rewrite.add_argument("--passes", type=int, default=3)
    rewrite.set_defaults(run=_rewrite)
    return parser


def _check(options: argparse.Namespace) -> int:
    report = check(_load(options.target), previous=options.previous)
    previous = str(options.previous) if options.previous is not None else None
    print(report.render(Invocation(options.target, previous)))
    return 0 if report.passed else 1


def _update(options: argparse.Namespace) -> int:
    try:
        update_baseline(_load(options.target), previous=options.previous)
    except PreviousRequiredError as refusal:
        print(f"error: {refusal.render(Invocation(options.target))}", file=sys.stderr)
        return 2
    return 0


def _hold(options: argparse.Namespace) -> int:
    hold_migration(
        _load(options.target),
        options.label,
        reason=options.reason,
        previous=options.previous,
    )
    return 0


def _fingerprint(options: argparse.Namespace) -> int:
    path: Path = options.baseline
    if not path.exists():
        raise ValueError(f"No baseline at {path}.")
    fingerprint = Baseline.read(path).fingerprint
    if isinstance(fingerprint, MigrationsUnrecorded):
        raise ValueError(
            f"The baseline at {path} is version {fingerprint.version} and "
            f"records no migrations. Run `baseline update` to write version 4."
        )
    print(fingerprint)
    return 0


def _rewrite(options: argparse.Namespace) -> int:
    outcome = rewrite_live_set(
        _load(options.target),
        expect=options.expect,
        apply=options.apply,
        passes=options.passes,
    )
    invocation = Invocation(options.target)
    if isinstance(outcome, WrongRelease):
        print(outcome.render(invocation), file=sys.stderr)
        return _WRONG_RELEASE
    print(outcome.render(invocation, options.expect))
    return 0 if outcome.clean else 1


def _load(spec: str) -> MigrationTarget:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise TargetError(f"The target {spec!r} is not 'module:attr'.")
    try:
        target = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as error:
        raise TargetError(f"The target {spec!r} does not import: {error}") from error
    if not isinstance(target, MigrationTarget):
        raise TargetError(
            f"The target {spec!r} is a {type(target).__name__}, not a MigrationTarget."
        )
    return target


if __name__ == "__main__":
    raise SystemExit(main())
