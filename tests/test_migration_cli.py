"""Tests for the migration CLI: ``python -m langgraph_events.serde.migrations``."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from typing import TYPE_CHECKING

from _dict_saver import DictSaver
from _live_set_fixtures import (
    _cfg,
    _copy_to_namespace,
    _Counted,
    _graph,
    _releases,
    _unsettling,
)
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

from langgraph_events.serde.migrations import (
    MigrationTarget,
    ThreadKey,
    list_thread_keys,
)
from langgraph_events.serde.migrations.__main__ import main

if TYPE_CHECKING:
    import contextlib
    from pathlib import Path

    import pytest
    from langgraph.checkpoint.memory import MemorySaver

    from langgraph_events import EventGraph


CLI_TARGET: MigrationTarget | None = None


def _install(monkeypatch: pytest.MonkeyPatch, target: MigrationTarget) -> str:
    monkeypatch.setattr(sys.modules[__name__], "CLI_TARGET", target)
    return f"{__name__}:CLI_TARGET"


def describe_gate_commands():
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
            def it_prints_the_reason_and_no_apply_command(
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
                assert out.endswith(
                    "  ghost: refused, no checkpoint for this thread id\n"
                )

        def when_the_last_plan_refuses_a_subgraph_checkpoint():
            def it_prints_the_reason_and_no_apply_command(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                releases = _releases(tmp_path)
                releases.seed("t1")
                _copy_to_namespace(releases.saver, "t1", "child:1")
                del releases.saver.storage["t1"][""]
                target = MigrationTarget(
                    graph=_graph,
                    baseline=tmp_path / "b.json",
                    store=releases.current.store,
                )
                spec = _install(monkeypatch, target)
                expect = target.snapshot().migrations.fingerprint

                code = main(["rewrite", spec, "--expect", expect, "--apply"])

                out = capsys.readouterr().out.splitlines()
                assert code == 1
                assert out[1].startswith(
                    "  t1 in subgraph namespace 'child:1': refused, history names "
                )
                assert out[1].endswith(
                    "which the serde cannot revive; add a tombstone first"
                )
                assert len(out) == 2

        def when_the_store_is_a_sqlite_saver():
            def it_rewrites_each_namespace_and_exits_zero(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                from langgraph.checkpoint.sqlite import SqliteSaver

                path = str(tmp_path / "store.db")
                with SqliteSaver.from_conn_string(path) as saver:
                    releases = _releases(tmp_path, saver)
                    releases.seed("t1", "t2")
                    _copy_to_namespace(saver, "t1", "child:1")
                target = dataclasses.replace(
                    releases.current, store=lambda: SqliteSaver.from_conn_string(path)
                )
                spec = _install(monkeypatch, target)

                code = main(["rewrite", spec, "--expect", releases.expect, "--apply"])

                out, err = capsys.readouterr()
                assert code == 0
                assert err == (
                    "pass 1: applied: 3 threads, 3 rewrite, 0 unchanged, 0 refused\n"
                )
                assert out.startswith("lge rewrite: clean")

        def when_the_store_is_a_custom_saver():
            def when_the_target_lists_its_threads():
                def with_list_thread_keys():
                    def it_rewrites_each_namespace_and_exits_zero(
                        tmp_path: Path,
                        monkeypatch: pytest.MonkeyPatch,
                        capsys: pytest.CaptureFixture[str],
                    ):
                        releases = _releases(tmp_path, DictSaver())
                        releases.seed("t1")
                        _copy_to_namespace(releases.saver, "t1", "child:1")
                        target = dataclasses.replace(
                            releases.current, threads=list_thread_keys
                        )
                        spec = _install(monkeypatch, target)

                        code = main(
                            ["rewrite", spec, "--expect", releases.expect, "--apply"]
                        )

                        assert code == 0
                        assert capsys.readouterr().err == (
                            "pass 1: applied: 2 threads, 2 rewrite, 0 unchanged, "
                            "0 refused\n"
                        )

            def when_the_saver_does_not_overwrite_a_checkpoint():
                def it_exits_one_and_names_the_saver_behaviour(
                    tmp_path: Path,
                    monkeypatch: pytest.MonkeyPatch,
                    capsys: pytest.CaptureFixture[str],
                ):
                    releases = _releases(tmp_path, DictSaver(put_overwrites=False))
                    releases.seed("t1")
                    tup = releases.saver.get_tuple(_cfg("t1"))
                    assert tup is not None
                    checkpoint_id = tup.config["configurable"]["checkpoint_id"]
                    target = dataclasses.replace(
                        releases.current, threads=list_thread_keys
                    )
                    spec = _install(monkeypatch, target)

                    code = main(
                        ["rewrite", spec, "--expect", releases.expect, "--apply"]
                    )

                    out = capsys.readouterr().out.splitlines()
                    assert code == 1
                    assert out[1].startswith(
                        f"  t1: needs a rewrite, the last pass refused it: the "
                        f"saver did not overwrite checkpoint {checkpoint_id}; "
                    )

        def when_a_class_stage_transform_does_not_settle():
            def it_exits_one_and_names_the_migration(
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str],
            ):
                target = _unsettling(tmp_path)
                expect = target.snapshot().migrations.fingerprint
                spec = _install(monkeypatch, target)

                code = main(["rewrite", spec, "--expect", expect, "--apply"])

                out = capsys.readouterr().out.splitlines()
                assert code == 1
                assert out[1] == (
                    f"  t1: refused, a migration of {_Counted.__module__}:"
                    f"{_Counted.__qualname__} does not settle: a read of the "
                    f"rewritten value in channel 'events' changes it again. "
                    f"Make each class-stage transform return its own output "
                    f"unchanged, then rerun"
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
                    f"{releases.expect}: {_CLI} rewrite {spec} --apply --expect "
                    f"{releases.expect}\n"
                )


def describe_target_errors():
    def when_the_graph_factory_raises():
        def it_exits_two_and_names_the_error(
            tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch,
            capsys: pytest.CaptureFixture[str],
        ):
            def build(saver: object) -> EventGraph:
                raise KeyError("DATABASE_URL")

            spec = _install(
                monkeypatch, MigrationTarget(graph=build, baseline=tmp_path / "b.json")
            )

            assert main(["check", spec]) == 2
            assert capsys.readouterr().err == "error: KeyError: 'DATABASE_URL'\n"

    def when_the_graph_factory_takes_no_checkpointer():
        def it_exits_two_and_names_the_factory_signature(
            tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch,
            capsys: pytest.CaptureFixture[str],
        ):
            def build() -> EventGraph:
                raise AssertionError("the factory ran")

            target = MigrationTarget(graph=build, baseline=tmp_path / "b.json")  # type: ignore[arg-type]
            spec = _install(monkeypatch, target)

            assert main(["check", spec]) == 2
            assert capsys.readouterr().err == (
                "error: The graph factory of the target cannot take a "
                "checkpointer: too many positional arguments. Write it as "
                "def build(checkpointer) -> EventGraph, and build the graph on "
                "that checkpointer.\n"
            )

    def when_the_store_opens_an_async_saver():
        def it_exits_two_and_names_the_sync_saver(
            tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch,
            capsys: pytest.CaptureFixture[str],
        ):
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            releases = _releases(tmp_path)
            target = dataclasses.replace(
                releases.current,
                store=lambda: AsyncSqliteSaver.from_conn_string(str(tmp_path / "s.db")),
            )
            spec = _install(monkeypatch, target)

            code = main(["rewrite", spec, "--expect", releases.expect, "--apply"])

            assert code == 2
            assert capsys.readouterr().err == (
                "error: The store= function of the target returned an async "
                "context manager. The rewrite command needs a sync checkpointer. "
                "Return the sync saver of the same database, for example "
                "SqliteSaver.from_conn_string(path) for an AsyncSqliteSaver, or "
                "PostgresSaver.from_conn_string(url) for an AsyncPostgresSaver.\n"
            )

    def when_the_store_does_not_open():
        def it_exits_two_and_prints_the_error_on_one_line(
            tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch,
            capsys: pytest.CaptureFixture[str],
        ):
            releases = _releases(tmp_path)

            def no_store() -> contextlib.AbstractContextManager[MemorySaver]:
                raise ConnectionError("connection refused\n\tIs the server running?")

            target = MigrationTarget(
                graph=releases.current.graph,
                baseline=releases.current.baseline,
                store=no_store,
            )
            spec = _install(monkeypatch, target)

            code = main(["rewrite", spec, "--expect", releases.expect, "--apply"])

            assert code == 2
            assert capsys.readouterr().err == (
                "error: ConnectionError: connection refused Is the server running?\n"
            )
