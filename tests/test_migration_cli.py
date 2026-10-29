"""Tests for the migration CLI: ``python -m langgraph_events.serde.migrations``."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from _live_set_fixtures import _copy_to_namespace, _releases
from _migration_fixtures import (
    _CLI,
    _TRIMMED,
    Shop,
    Trimmed,
    _baselined,
    _deleted_transform,
    _read,
    _target,
)

from langgraph_events.serde.migrations import MigrationTarget, ThreadKey
from langgraph_events.serde.migrations.__main__ import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


CLI_TARGET: MigrationTarget | None = None


def describe_gate_commands():
    def _install(monkeypatch: pytest.MonkeyPatch, target: MigrationTarget) -> str:
        monkeypatch.setattr(sys.modules[__name__], "CLI_TARGET", target)
        return f"{__name__}:CLI_TARGET"

    def describe_check():
        def when_the_gate_passes():
            def it_exits_zero(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                spec = _install(monkeypatch, _baselined(_target(tmp_path, Shop)))

                assert main(["check", spec]) == 0
                assert capsys.readouterr().out.startswith("lge migration gate: passed")

        def when_the_gate_finds_a_problem():
            def it_exits_one_and_prints_the_callers_command(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                spec = _install(monkeypatch, _target(tmp_path, Shop))

                assert main(["check", spec]) == 1
                assert f"Run: {_CLI} baseline update {spec}" in capsys.readouterr().out

        def when_the_target_does_not_import():
            def it_exits_two(capsys: pytest.CaptureFixture[str]):
                assert main(["check", "no_such_module:target"]) == 2
                assert "does not import" in capsys.readouterr().err

        def when_the_attribute_is_not_a_target():
            def it_exits_two(capsys: pytest.CaptureFixture[str]):
                assert main(["check", "_migration_fixtures:_target"]) == 2
                assert "not a MigrationTarget" in capsys.readouterr().err

    def describe_baseline_update():
        def when_the_target_imports():
            def it_writes_the_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
                target = _target(tmp_path, Shop)
                spec = _install(monkeypatch, target)

                assert main(["baseline", "update", spec]) == 0
                assert _read(target.baseline)["version"] == 4

        def when_a_shipped_migration_left_the_code():
            def it_exits_two_and_prints_the_callers_command(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                target, _ = _deleted_transform(tmp_path)
                spec = _install(monkeypatch, target)

                assert main(["baseline", "update", spec]) == 2
                assert (
                    f"Run: {_CLI} baseline update {spec} --previous <baseline file "
                    f"of the previous release>" in capsys.readouterr().err
                )

        def when_the_previous_file_is_missing():
            def it_exits_two(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                spec = _install(monkeypatch, _target(tmp_path, Shop))

                code = main(
                    ["baseline", "update", spec, "--previous", str(tmp_path / "x.json")]
                )

                assert code == 2
                assert "No previous baseline at" in capsys.readouterr().err

    def describe_baseline_hold():
        def when_every_argument_is_given():
            def it_writes_the_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
                target = _baselined(_target(tmp_path, Trimmed))
                previous = tmp_path / "prev.json"
                previous.write_text(target.baseline.read_text())
                label = _TRIMMED
                spec = _install(monkeypatch, target)

                code = main(
                    [
                        "baseline",
                        "hold",
                        spec,
                        label,
                        "--reason",
                        "subgraph",
                        "--previous",
                        str(previous),
                    ]
                )

                assert code == 0
                assert (
                    _read(target.baseline)["migrations"][0]["hold"]["reason"]
                    == "subgraph"
                )

        def when_the_reason_is_missing():
            def it_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
                spec = _install(monkeypatch, _target(tmp_path, Trimmed))

                assert main(["baseline", "hold", spec, "rename m:X"]) == 2

    def describe_fingerprint():
        def when_the_file_records_migrations():
            def it_prints_the_fingerprint_of_the_file(
                tmp_path: Path, capsys: pytest.CaptureFixture[str]
            ):
                baseline = tmp_path / "baseline.json"
                baseline.write_text(
                    json.dumps(
                        {
                            "version": 4,
                            "events": [],
                            "migrations": [{"label": "rename m:Old"}],
                        }
                    )
                )

                assert main(["fingerprint", str(baseline)]) == 0
                assert capsys.readouterr().out == "lge-m1-dbc7afada42b30d1\n"

        def when_the_file_predates_v4():
            def it_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
                baseline = tmp_path / "baseline.json"
                baseline.write_text(json.dumps({"version": 3, "events": []}))

                assert main(["fingerprint", str(baseline)]) == 2
                assert "records no migrations" in capsys.readouterr().err

    def when_the_removed_positional_form_is_used():
        def it_exits_two_and_prints_the_new_command(
            capsys: pytest.CaptureFixture[str],
        ):
            assert main(["myapp.graph:build", "baseline.json"]) == 2
            assert (
                f"{_CLI} check <module:attr of the MigrationTarget>"
                in capsys.readouterr().err
            )

    def when_invoked_as_a_module():
        def it_advertises_the_working_entrypoint():
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-m", "langgraph_events.serde.migrations"],
                capture_output=True,
                check=False,
            )

            assert result.returncode == 2
            assert (
                b"usage: python -m langgraph_events.serde.migrations" in result.stderr
            )


def describe_rewrite_command():
    def _install(monkeypatch: pytest.MonkeyPatch, target: MigrationTarget) -> str:
        monkeypatch.setattr(sys.modules[__name__], "CLI_TARGET", target)
        return f"{__name__}:CLI_TARGET"

    def describe_rewrite():
        def when_the_apply_reads_clean():
            def it_exits_zero_and_logs_each_pass(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                releases = _releases(tmp_path)
                releases.seed("t1")
                spec = _install(monkeypatch, releases.current)

                code = main(["rewrite", spec, "--expect", releases.expect, "--apply"])

                out, err = capsys.readouterr()
                assert code == 0
                assert out == (
                    "lge rewrite: clean (last plan: plan: 1 thread, 0 rewrite, "
                    "1 unchanged, 0 refused)\n"
                )
                assert err == (
                    "pass 1: applied: 1 thread, 1 rewrite, 0 unchanged, 0 refused\n"
                )

        def when_a_plan_finds_a_thread_to_rewrite():
            def it_exits_one_and_names_the_apply_command(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                releases = _releases(tmp_path)
                releases.seed("t1")
                spec = _install(monkeypatch, releases.current)

                code = main(["rewrite", spec, "--expect", releases.expect])

                assert code == 1
                assert capsys.readouterr().out == (
                    "lge rewrite: not clean (last plan: plan: 1 thread, 1 rewrite, "
                    "0 unchanged, 0 refused)\n"
                    "  t1: needs a rewrite\n"
                    f"  Run: {_CLI} rewrite {spec} --apply --expect {releases.expect}\n"
                )

        def when_the_last_plan_refuses_a_thread():
            def it_prints_the_reason_and_the_apply_command(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                releases = _releases(tmp_path)
                target = MigrationTarget(
                    graph=releases.current.graph,
                    baseline=releases.current.baseline,
                    store=releases.current.store,
                    threads=lambda saver: [ThreadKey("ghost", "")],
                )
                spec = _install(monkeypatch, target)

                code = main(["rewrite", spec, "--expect", releases.expect, "--apply"])

                out = capsys.readouterr().out
                assert code == 1
                assert "  ghost: refused, no checkpoint for this thread id\n" in out
                assert out.endswith(
                    f"  Run: {_CLI} rewrite {spec} --apply --expect {releases.expect}\n"
                )

        def when_a_subgraph_checkpoint_holds_an_old_shape():
            def it_exits_one_and_names_the_hold_command(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                releases = _releases(tmp_path)
                releases.seed("t1")
                _copy_to_namespace(releases.saver, "t1", "child:1")
                spec = _install(monkeypatch, releases.current)
                (label,) = [
                    key.label for key in releases.current.snapshot().migrations.keys
                ]

                code = main(["rewrite", spec, "--expect", releases.expect, "--apply"])

                assert code == 1
                assert capsys.readouterr().out == (
                    "lge rewrite: not clean (last plan: plan: 1 thread, 0 rewrite, "
                    "1 unchanged, 0 refused)\n"
                    f"  t1 in subgraph namespace 'child:1': rewrite, read by: {label}\n"
                    "  The rewrite never writes a subgraph checkpoint. Keep the "
                    f"migration one more release: {_CLI} baseline hold {spec} "
                    f"'{label}' --reason <why> "
                    "--previous <baseline file of the previous release>\n"
                )

        def when_the_running_release_is_not_the_expected_one():
            def it_exits_three_and_names_both_fingerprints(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                releases = _releases(tmp_path)
                spec = _install(monkeypatch, releases.previous)
                running = releases.previous.snapshot().migrations.fingerprint

                code = main(["rewrite", spec, "--expect", releases.expect, "--apply"])

                assert code == 3
                assert capsys.readouterr().err == (
                    f"lge rewrite: wrong release. The running code holds the "
                    f"migrations {running}, not {releases.expect}.\n"
                    "  running migrations: none\n"
                    f"  Run the command in the release whose baseline gives "
                    f"{releases.expect}: {_CLI} rewrite {spec} --expect "
                    f"{releases.expect}\n"
                )
