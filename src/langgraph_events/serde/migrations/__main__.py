"""The migration CLI: ``python -m langgraph_events.serde.migrations``.

``<target>`` is the ``module:attr`` of a ``MigrationTarget``. Exit 0 is a
pass or a done write, 1 is a finding or a store worse than ``rewrite
--accept``, 2 is a usage, target or store error, a ``restore`` with no
backup included, 3 is a ``rewrite`` under the wrong release. A report goes
to stdout, and an error to stderr.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from langgraph_events.serde.migrations._backup import restore_live_set
from langgraph_events.serde.migrations._baseline import (
    Baseline,
    MigrationsUnrecorded,
)
from langgraph_events.serde.migrations._core import PROGRAM, TARGET_PLACEHOLDER
from langgraph_events.serde.migrations._gate import (
    Invocation,
    PreviousRequiredError,
    check,
    hold_migration,
    update_baseline,
)
from langgraph_events.serde.migrations._live_set import (
    LiveSetState,
    WrongRelease,
    rewrite_live_set,
)
from langgraph_events.serde.migrations._target import MigrationTarget, TargetError

_SUBCOMMANDS = ("check", "baseline", "fingerprint", "rewrite", "restore")
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
    except TargetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        # The target, its store or an input raised. Without this handler,
        # Python exits 1, and exit 1 tells a deploy step to run --apply.
        message = " ".join(str(error).split())
        print(f"error: {type(error).__name__}: {message}", file=sys.stderr)
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
    rewrite.add_argument("target", help="the module:attr of the MigrationTarget")
    rewrite.add_argument(
        "--expect",
        required=True,
        metavar="FP",
        help="the fingerprint of the baseline of the release that must run",
    )
    rewrite.add_argument(
        "--apply", action="store_true", help="write the rewrite, else only plan"
    )
    rewrite.add_argument(
        "--passes", type=int, default=3, help="the most apply passes (default: 3)"
    )
    rewrite.add_argument(
        "--backup",
        metavar="TAG",
        help="back up the live set of a PostgresSaver store before the first plan",
    )
    rewrite.add_argument(
        "--accept",
        choices=[str(state) for state in LiveSetState],
        default=str(LiveSetState.CLEAN),
        help="the worst state of the last plan that still exits 0 (default: clean)",
    )
    rewrite.set_defaults(run=_rewrite)

    restore = commands.add_parser(
        "restore", help="put back the backup TAG of the live set"
    )
    restore.add_argument("target", help="the module:attr of the MigrationTarget")
    restore.add_argument(
        "tag", metavar="TAG", help="the tag that rewrite --backup was given"
    )
    restore.set_defaults(run=_restore)
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
        log=sys.stdout,
        backup_tag=options.backup,
    )
    invocation = Invocation(options.target)
    if isinstance(outcome, WrongRelease):
        print(outcome.render(invocation, apply=options.apply), file=sys.stderr)
        return _WRONG_RELEASE
    accept = LiveSetState(options.accept)
    print(outcome.render(invocation, options.expect, accept))
    return 0 if outcome.state <= accept else 1


def _restore(options: argparse.Namespace) -> int:
    counts = restore_live_set(_load(options.target), options.tag)
    print(f"lge restore: put back {counts} from the backup {options.tag}.")
    return 0


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
