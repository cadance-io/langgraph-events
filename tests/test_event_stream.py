"""``EventStream``: a graph over a durable log, appended per superstep."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import _store_scenarios as sc
import pytest
from conftest import Order
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_events import (
    EventGraph,
    EventLog,
    IntegrationEvent,
    Interrupted,
    InterruptWithoutCheckpointerError,
    on,
)
from langgraph_events.store import (
    EventCodec,
    EventStream,
    JsonlEventStore,
    MemoryEventStore,
    Record,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

TESTS = Path(__file__).parent


class FailsOnceMemoryEventStore(MemoryEventStore):
    """A store whose first append fails before any record becomes durable."""

    def __init__(self) -> None:
        super().__init__()
        self._fails = True

    def append(self, records: Sequence[Record]) -> None:
        if self._fails:
            self._fails = False
            raise OSError("append failed")
        super().append(records)


class Request(IntegrationEvent):
    what: str = ""


class Granted(IntegrationEvent):
    request: Request | None = None


@on(Request)
def grant(event: Request) -> Granted:
    return Granted(request=event)


class Recall(IntegrationEvent):
    pass


class Recalled(IntegrationEvent):
    first: Request | None = None


@on(Recall)
def recall(event: Recall, log: EventLog) -> Recalled:
    return Recalled(first=log.filter(Request)[0])


class Ask(IntegrationEvent):
    pass


class AskApproval(Interrupted):
    pass


@on(Ask)
def ask(event: Ask) -> AskApproval:
    return AskApproval()


class Boom(IntegrationEvent):
    pass


@on(Order.Shipped)
def shipped(event: Order.Shipped) -> None:
    return None


@on(sc.Tock)
def explode(event: sc.Tock) -> Boom:
    raise RuntimeError("handler failed")


def _committed(n: int) -> MemoryEventStore:
    store = MemoryEventStore()
    stream = EventStream(
        EventGraph([sc.guard], reducers=[sc.ticks()]), store, EventCodec()
    )
    for _ in range(n):
        stream.invoke(sc.Tick())
    return store


def _reopen(store: MemoryEventStore) -> EventStream:
    return EventStream(
        EventGraph([sc.guard], reducers=[sc.ticks()]), store, EventCodec()
    )


def describe_EventStream():
    def when_the_graph_has_a_checkpointer():
        def it_refuses_the_graph():
            graph = EventGraph([grant], checkpointer=InMemorySaver())
            with pytest.raises(ValueError, match="without checkpointer="):
                EventStream(graph, MemoryEventStore(), EventCodec())

    def describe_invoke():
        def when_an_append_fails_once():
            def when_the_same_seed_retries():
                def with_a_valid_link():
                    def it_reopens():
                        store = FailsOnceMemoryEventStore()
                        stream = EventStream(EventGraph([grant]), store, EventCodec())
                        request = Request(what="retry")
                        with pytest.raises(OSError, match="append failed"):
                            stream.invoke(request)
                        stream.invoke(request)
                        reopened = EventStream(EventGraph([grant]), store, EventCodec())
                        assert reopened.log[1].request is reopened.log[0]

        def when_the_turn_completes():
            def it_returns_only_the_events_of_this_turn():
                stream = EventStream(
                    EventGraph([grant]), MemoryEventStore(), EventCodec()
                )
                stream.invoke(Request(what="a"))
                run = stream.invoke(Request(what="b"))
                assert [type(e).__name__ for e in run] == ["Request", "Granted"]
                assert len(stream.log) == 4

            def it_injects_the_stored_log_then_the_turn_into_a_handler():
                seen: list[list[str]] = []

                @on(sc.Tock)
                def look(event: sc.Tock, log: EventLog) -> None:
                    seen.append([f"{type(e).__name__}{i}" for i, e in enumerate(log)])

                store = MemoryEventStore()
                graph = EventGraph([sc.guard, look], reducers=[sc.ticks()])
                EventStream(graph, store, EventCodec()).invoke(sc.Tick())
                EventStream(graph, store, EventCodec()).invoke(sc.Tick())
                assert seen == [
                    ["Tick0", "Tock1"],
                    ["Tick0", "Tock1", "Tick2", "Tock3"],
                ]

            def it_takes_a_list_of_seeds():
                stream = EventStream(
                    EventGraph([grant]), MemoryEventStore(), EventCodec()
                )
                run = stream.invoke([Request(what="a"), Request(what="b")])
                assert [type(e).__name__ for e in run] == [
                    "Request",
                    "Request",
                    "Granted",
                    "Granted",
                ]

            def it_takes_a_namespace_domain_event_as_the_seed():
                store = MemoryEventStore()
                graph = EventGraph([shipped])
                EventStream(graph, store, EventCodec(namespaces=[Order])).invoke(
                    Order.Shipped(tracking="T-1")
                )
                log = EventStream(graph, store, EventCodec(namespaces=[Order])).log
                assert log[0] == Order.Shipped(tracking="T-1")

            def when_a_handler_receives_live_causes():
                def without_persisting_them():
                    def it_supplies_aligned_causes():
                        supplied: list[int] = []

                        @on(sc.Tock)
                        def inspect_causes(event: sc.Tock, log: EventLog) -> None:
                            assert log.causes is not None
                            supplied.append(len(log.causes))

                        store = MemoryEventStore()
                        EventStream(
                            EventGraph([sc.guard], reducers=[sc.ticks()]),
                            store,
                            EventCodec(),
                        ).invoke(sc.Tick())
                        stream = EventStream(
                            EventGraph(
                                [sc.guard, inspect_causes], reducers=[sc.ticks()]
                            ),
                            store,
                            EventCodec(),
                        )
                        stream.invoke(sc.Tick())
                        assert supplied == [4]
                        assert stream.log.causes is None

        def when_a_later_handler_crashes_the_process():
            def it_keeps_the_events_stored_before_the_crash(tmp_path):
                path = tmp_path / "book.jsonl"
                marker = tmp_path / "marker"
                crashed = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-c",
                        "import _store_scenarios as s; s.main()",
                        "crash_after_side_effect",
                        str(path),
                        str(marker),
                    ],
                    env={**os.environ, "PYTHONPATH": str(TESTS)},
                    check=False,
                )
                assert crashed.returncode == 1
                assert marker.exists()
                with JsonlEventStore(path) as store:
                    rows = store.load()
                assert [r.type for r in rows] == ["Start", "Approved"]

        def when_a_handler_raises():
            def it_keeps_the_stored_events_and_folds_them():
                store = MemoryEventStore()
                graph = EventGraph([sc.guard, explode], reducers=[sc.ticks()])
                stream = EventStream(graph, store, EventCodec())
                with pytest.raises(RuntimeError, match="handler failed"):
                    stream.invoke(sc.Tick())
                assert [r.type for r in store.load()] == ["Tick", "Tock"]
                assert stream.state() == {"ticks": 1}
                assert len(stream.log) == 2

        def when_a_handler_returns_an_interrupt():
            def it_raises_and_stores_the_events_before_it():
                store = MemoryEventStore()
                stream = EventStream(EventGraph([ask]), store, EventCodec())
                with pytest.raises(InterruptWithoutCheckpointerError):
                    stream.invoke(Ask())
                assert [r.type for r in store.load()] == ["Ask"]

    def when_a_stored_record_does_not_decode():
        def it_raises_naming_the_record():
            store = MemoryEventStore()
            store.append([Record(__name__, "Granted", {"request": {"$ref": 4}})])
            with pytest.raises(ValueError, match=r"record #0 .*\$ref 4"):
                EventStream(EventGraph([grant]), store, EventCodec())

    def describe_state():
        def it_folds_the_history_once_for_many_reads():
            store = _committed(5)
            sc.FOLDS.clear()
            stream = _reopen(store)
            for _ in range(10):
                assert stream.state() == {"ticks": 5}
            assert len(sc.FOLDS) == 5

        def it_folds_a_turn_at_a_cost_independent_of_history():
            folds = []
            for history in (5, 200):
                stream = _reopen(_committed(history))
                sc.FOLDS.clear()
                run = stream.invoke(sc.Tick())
                assert [e.seen for e in run.filter(sc.Tock)] == [history + 1]
                assert stream.state() == {"ticks": history + 1}
                folds.append(len(sc.FOLDS))
            assert folds[0] == folds[1]

    def describe_log():
        def it_revives_a_link_as_the_same_object(tmp_path):
            path = tmp_path / "book.jsonl"
            with JsonlEventStore(path) as store:
                EventStream(EventGraph([grant]), store, EventCodec()).invoke(
                    Request(what="leave")
                )
            with JsonlEventStore(path) as store:
                log = EventStream(EventGraph([grant]), store, EventCodec()).log
            assert log[1].request is log[0]

        def it_links_an_event_of_an_earlier_turn_by_its_position():
            store = MemoryEventStore()
            graph = EventGraph([grant, recall])
            EventStream(graph, store, EventCodec()).invoke(Request(what="a"))
            EventStream(graph, store, EventCodec()).invoke(Recall())
            assert store.load()[-1].fields == {"first": {"$ref": 0}}
            log = EventStream(graph, store, EventCodec()).log
            assert log[-1].first is log[0]

        def it_repairs_a_torn_tail_before_the_next_append(tmp_path):
            path = tmp_path / "book.jsonl"
            whole = Record(__name__, "Request", {"what": "a"}).to_json()
            path.write_text(whole + "\n" + whole[:9])
            with JsonlEventStore(path) as store:
                EventStream(EventGraph([grant]), store, EventCodec()).invoke(
                    Request(what="b")
                )
                assert [r.type for r in store.load()] == [
                    "Request",
                    "Request",
                    "Granted",
                ]

    def when_a_reader_polls_the_file_during_appends():
        def it_reads_only_whole_records_in_growing_order(tmp_path):
            path = tmp_path / "book.jsonl"
            errors: list[BaseException] = []
            sizes: list[int] = []
            done = threading.Event()

            def read() -> None:
                while not done.is_set():
                    try:
                        data = path.read_bytes() if path.exists() else b""
                        rows = [Record.from_json(x) for x in data.split(b"\n")[:-1]]
                        sizes.append(len(rows))
                    except ValueError as exc:
                        errors.append(exc)

            reader = threading.Thread(target=read)
            reader.start()
            with JsonlEventStore(path) as store:
                stream = EventStream(EventGraph([grant]), store, EventCodec())
                for i in range(200):
                    stream.invoke(Request(what=str(i) * 50))
            done.set()
            reader.join()
            assert errors == []
            assert sizes == sorted(sizes)
