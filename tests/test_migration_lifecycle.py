"""The lifecycle of a read-time migration, end to end, on a Postgres store.

Each release is its own source tree with the same module names, and each
command runs in its own process, as in a deploy. All releases share one
Postgres database.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_TARGET = """\
import os
from pathlib import Path

from langgraph.checkpoint.postgres import PostgresSaver

from app.graph import build_graph
from langgraph_events.serde.migrations import MigrationTarget

target = MigrationTarget(
    graph=build_graph,
    baseline=Path(__file__).parent / "lge-baseline.json",
    store=lambda: PostgresSaver.from_conn_string(os.environ["APP_DATABASE_URL"]),
)
"""

_RUN = """\
import sys

from app.graph import Requested
from app.target import target

command, thread_id = sys.argv[1:]
config = {"configurable": {"thread_id": thread_id}}
with target.store() as saver:
    graph = target.graph(saver)
    if command == "seed":
        graph.invoke(Requested(item="book"), config=config)
    else:
        print(graph.get_state(config).events[-1])
"""

_GRAPH_HEAD = """\
from langgraph_events import EventGraph, IntegrationEvent, on
from langgraph_events.serde import NamespaceAwareSerde, transform_fields


class Requested(IntegrationEvent):
    item: str = ""
"""

_GRAPH_TAIL = """
@on(Requested)
def order(event: Requested) -> Ordered:
    return Ordered(item=event.item)


def build_graph(saver):
    saver.serde = NamespaceAwareSerde(events=(Requested, Ordered))
    return EventGraph([order], checkpointer=saver)
"""

_BEFORE = """
class Ordered(IntegrationEvent):
    item: str = ""
    legacy_total: int = 7
"""

_MIGRATED = """
def drop_legacy_total(fields):
    fields.pop("legacy_total", None)
    return fields


@transform_fields(drop_legacy_total)
class Ordered(IntegrationEvent):
    item: str = ""
"""

_AFTER = """
class Ordered(IntegrationEvent):
    item: str = ""
"""

_CLI = (sys.executable, "-m", "langgraph_events.serde.migrations")
_SURVIVOR = "transform app.graph:Ordered via app.graph:drop_legacy_total"


class _Release:
    """One release: a source tree under *root* with its own baseline file."""

    def __init__(self, root: Path, events: str, url: str, previous: "_Release | None"):
        self.root = root
        self.url = url
        app = root / "app"
        app.mkdir(parents=True)
        (app / "__init__.py").write_text("")
        (app / "graph.py").write_text(_GRAPH_HEAD + events + _GRAPH_TAIL)
        (app / "target.py").write_text(_TARGET)
        (app / "run.py").write_text(_RUN)
        if previous is not None:
            shutil.copy(previous.baseline, self.baseline)

    @property
    def baseline(self) -> Path:
        return self.root / "app" / "lge-baseline.json"

    def run(self, *argv: str) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "PYTHONPATH": str(self.root), "APP_DATABASE_URL": self.url}
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv, cwd=self.root, env=env, capture_output=True, text=True, check=False
        )

    def lge(self, *argv: str) -> subprocess.CompletedProcess[str]:
        return self.run(*_CLI, *argv)

    def app(self, command: str, thread_id: str) -> subprocess.CompletedProcess[str]:
        return self.run(sys.executable, "-m", "app.run", command, thread_id)

    def fingerprint(self) -> str:
        result = self.lge("fingerprint", str(self.baseline))
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()


def _fields(baseline: Path, qualname: str) -> list[str]:
    (entry,) = [
        e
        for e in json.loads(baseline.read_text())["events"]
        if e["qualname"] == qualname
    ]
    return entry["fields"]


def describe_the_lifecycle_of_a_read_time_migration():
    def when_release_n_ships_a_transform_and_release_n_plus_1_deletes_it():
        def it_runs_every_documented_step_against_one_postgres_store(
            tmp_path: Path, postgres_url: str
        ):
            target = "app.target:target"
            older = _Release(tmp_path / "n-1", _BEFORE, postgres_url, None)
            assert older.lge("baseline", "update", target).returncode == 0
            assert older.app("seed", "t1").returncode == 0

            n = _Release(tmp_path / "n", _MIGRATED, postgres_url, older)
            assert n.lge("baseline", "update", target).returncode == 0
            ci_n = n.lge("check", target, "--previous", str(older.baseline))
            assert ci_n.returncode == 0, ci_n.stdout

            pre_n = older.lge("rewrite", target, "--expect", older.fingerprint())
            assert pre_n.returncode == 0, "N-1 reads its own store clean"

            fp_n = n.fingerprint()
            wrong = older.lge("rewrite", target, "--apply", "--expect", fp_n)
            assert wrong.returncode == 3, "N-1 cannot run the step of N"

            n_plus_1 = _Release(tmp_path / "n+1", _AFTER, postgres_url, n)
            unread = n_plus_1.app("read", "t1")
            assert unread.returncode != 0, "N+1 cannot read the old shape yet"
            assert "legacy_total" in unread.stderr

            plan = n.lge("rewrite", target, "--expect", fp_n)
            assert plan.returncode == 1, plan.stdout
            assert "  t1: needs a rewrite\n" in plan.stdout
            post_n = n.lge("rewrite", target, "--apply", "--expect", fp_n)
            assert post_n.returncode == 0, post_n.stdout

            kept = _Release(tmp_path / "n+1-kept", _MIGRATED, postgres_url, n)
            survived = kept.lge("check", target, "--previous", str(n.baseline))
            assert survived.returncode == 1
            assert f"  survived-release  {_SURVIVOR}\n" in survived.stdout

            assert _fields(n_plus_1.baseline, "Ordered") == ["item", "legacy_total"]
            update = n_plus_1.lge(
                "baseline", "update", target, "--previous", str(n.baseline)
            )
            assert update.returncode == 0, update.stderr
            assert _fields(n_plus_1.baseline, "Ordered") == ["item"]
            ci = n_plus_1.lge("check", target, "--previous", str(n.baseline))
            assert ci.returncode == 0, ci.stdout
            test_gate = n_plus_1.lge("check", target)
            assert test_gate.returncode == 0, test_gate.stdout

            pre = n.lge("rewrite", target, "--expect", fp_n)
            assert pre.returncode == 0, "N reads the store clean before N+1 runs"

            read = n_plus_1.app("read", "t1")
            assert read.returncode == 0, read.stderr
            assert "item='book'" in read.stdout
            post = n_plus_1.lge(
                "rewrite", target, "--apply", "--expect", n_plus_1.fingerprint()
            )
            assert post.returncode == 0, post.stdout
