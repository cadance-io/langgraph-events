"""EventTopology, EventGraph.of() and EventStore: a graph without its services.

A client that renders a graph, runs the migration gate or maintains a store
never runs a handler. These suites pin that such a client needs no service
value: the handlers below need a ``_Mailer`` that is never built.
"""

import warnings
from pathlib import Path
from typing import Any

import pytest
from conftest import Ended, Started
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Abandoned,
    Event,
    EventGraph,
    EventStore,
    EventTopology,
    IntegrationEvent,
    Interrupted,
    OrphanedEventWarning,
    ScalarReducer,
    ServiceRequirement,
    Unannotated,
    UnresolvedAnnotation,
    on,
)
from langgraph_events.serde import NamespaceAwareSerde, migrate_from
from langgraph_events.serde.migrations import (
    MigrationTarget,
    check,
    update_baseline,
)


class _Mailer:
    """A service that a maintenance client cannot build."""

    def send(self, text: str) -> str:
        return f"sent {text}"


class _Approval(Interrupted):
    pass


class _Approved(IntegrationEvent):
    pass


class _Counted(IntegrationEvent):
    pass


@on(Started)
def _ask(event: Started, mailer: _Mailer) -> _Approval:
    mailer.send(event.data)
    return _Approval()


@on(_Approved)
def _finish(event: _Approved, mailer: _Mailer) -> Ended:
    return Ended(result=mailer.send("done"))


@on(Started)
def _count(event: Started, total: int | None) -> _Counted:
    return _Counted()


_TOTAL = ScalarReducer(name="total", event_type=_Counted, fn=lambda e: 1)

_HANDLERS = [_ask, _finish]


def _cfg(tid: str) -> Any:
    return {"configurable": {"thread_id": tid}}


def _paused(saver: MemorySaver, tid: str) -> EventTopology:
    """Run *tid* until ``_ask`` pauses it. Return the topology of the run."""
    topology = EventTopology(_HANDLERS)
    graph = EventGraph.of(topology, services=[_Mailer()], checkpointer=saver)
    graph.invoke(Started(data="x"), config=_cfg(tid))
    return topology


def describe_EventTopology():
    def when_a_handler_needs_a_service():
        def it_lists_the_parameter_as_a_requirement():
            topology = EventTopology(_HANDLERS)

            assert topology.requirements == (
                ServiceRequirement("_ask", "mailer", _Mailer),
                ServiceRequirement("_finish", "mailer", _Mailer),
            )

        def it_renders_the_model_of_the_graph():
            graph = EventGraph(_HANDLERS, services=[_Mailer()])

            model = EventTopology(_HANDLERS).namespaces()

            assert model.text() == graph.namespaces().text()
            assert model.mermaid() == graph.namespaces().mermaid()
            assert model.json() == graph.namespaces().json()

        def it_lists_the_node_names_of_the_graph():
            graph = EventGraph(_HANDLERS, services=[_Mailer()])

            topology = EventTopology(_HANDLERS)

            assert topology.handler_names == graph.handler_names
            assert topology.resumable_node_names == graph.resumable_node_names

    def when_a_parameter_has_no_annotation():
        def it_lists_the_parameter_as_unannotated():
            @on(Started)
            def untyped(event: Started, client) -> None:
                return None

            (requirement,) = EventTopology([untyped]).requirements

            assert requirement.annotation == Unannotated()

    def when_an_annotation_does_not_resolve():
        def it_lists_the_reason():
            @on(Started)
            def broken(event: Started, client: int) -> None:
                return None

            broken.__annotations__["client"] = "MissingClient"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (requirement,) = EventTopology([broken]).requirements

            assert requirement.annotation == UnresolvedAnnotation(
                "NameError: name 'MissingClient' is not defined"
            )

    def when_a_parameter_is_variadic():
        def it_lists_no_requirement():
            @on(Started)
            def catch_all(event: Started, *args: Any, **kwargs: Any) -> None:
                return None

            assert EventTopology([catch_all]).requirements == ()

    def when_a_reducer_claims_the_parameter():
        def it_lists_no_requirement():
            topology = EventTopology([_count], reducers=[_TOTAL])

            assert topology.requirements == ()
            assert topology.reducer_names == frozenset({"total"})

    def when_no_handler_is_given():
        def it_raises_a_value_error():
            with pytest.raises(ValueError, match="at least one handler"):
                EventTopology([])

    def when_a_loose_event_declares_a_migration():
        def it_builds_a_serde_that_revives_the_old_identity():
            class _OldNote(IntegrationEvent):
                pass

            stored = NamespaceAwareSerde(events=(_OldNote,)).dumps_typed(_OldNote())

            @migrate_from(_OldNote.__qualname__, in_module=_OldNote.__module__)
            class _NewNote(IntegrationEvent):
                pass

            @on(_NewNote)
            def react(event) -> None:  # type: ignore[no-untyped-def]
                return None

            serde = EventTopology([react]).serde()

            assert serde.loads_typed(stored) == _NewNote()


def describe_EventGraph_of():
    def when_every_requirement_has_a_service():
        def it_injects_the_service():
            saver = MemorySaver()
            topology = EventTopology(_HANDLERS)
            graph = EventGraph.of(topology, services=[_Mailer()], checkpointer=saver)
            graph.invoke(Started(data="x"), config=_cfg("of-runs"))

            log = graph.resume(_Approved(), config=_cfg("of-runs"))

            assert log.latest(Ended) == Ended(result="sent done")

    def when_a_service_is_missing():
        def it_raises_the_type_error_of_the_constructor():
            with pytest.raises(TypeError) as direct:
                EventGraph(_HANDLERS)

            with pytest.raises(TypeError) as bound:
                EventGraph.of(EventTopology(_HANDLERS))

            assert "cannot inject" in str(bound.value)
            assert str(bound.value) == str(direct.value)

    def when_a_renamed_duplicate_misses_a_service():
        def it_names_the_handler_as_declared():
            @on(Started)
            def handle(event: Started) -> None:
                return None

            first = handle

            @on(Started)  # type: ignore[no-redef]
            def handle(event: Started, mailer: _Mailer) -> None:
                return None

            with pytest.raises(TypeError) as info:
                EventGraph([first, handle])

            assert str(info.value) == (
                "Handler 'handle' declares parameter(s) ['mailer'] that the "
                "framework cannot inject. For service injection, register a "
                "matching instance via EventGraph(services=[...]); for state, "
                "register a Reducer; otherwise remove the parameter."
            )


def _bad_return(event: Started) -> Event:
    return Ended(result="x")


def describe_EventGraph():
    def when_the_services_and_a_handler_are_both_invalid():
        def it_refuses_the_services_first():
            bad = on(Started)(_bad_return)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(TypeError) as info:
                    EventGraph([bad], services=[_Mailer(), _Mailer()])

            assert str(info.value) == (
                "EventGraph(services=...) has two instances of type '_Mailer' "
                "\u2014 same-type collision. Register only one instance, switch "
                "to the name-keyed mapping form (services={'primary': a, "
                "'backup': b}), or use a subclass to distinguish them."
            )
            assert not [w for w in caught if w.category is OrphanedEventWarning]


def describe_EventStore():
    def when_the_service_of_a_handler_is_never_built():
        def it_finds_the_paused_thread():
            saver = MemorySaver()
            topology = _paused(saver, "store-find")

            configs = EventStore(topology, saver).threads_paused_on(
                _Approval, thread_ids=["store-find"]
            )

            assert configs == [_cfg("store-find")]

        def it_abandons_the_paused_thread():
            saver = MemorySaver()
            topology = _paused(saver, "store-abandon")

            EventStore(topology, saver).abandon(_cfg("store-abandon"), reason="r")

            graph = EventGraph.of(topology, services=[_Mailer()], checkpointer=saver)
            last = graph.get_state(_cfg("store-abandon")).events[-1]
            assert last == Abandoned(reason="r", discarded="_Approval")

        async def it_abandons_the_paused_thread_async():
            saver = MemorySaver()
            topology = _paused(saver, "store-aabandon")
            store = EventStore(topology, saver)

            await store.aabandon(_cfg("store-aabandon"))

            assert await store.athreads_paused_on(thread_ids=["store-aabandon"]) == []

        def it_rewrites_a_renamed_identity():
            saver = MemorySaver()

            class _OldNote(IntegrationEvent):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _OldNote))
            _paused(saver, "store-rewrite")
            graph = EventGraph.of(
                EventTopology(_HANDLERS), services=[_Mailer()], checkpointer=saver
            )
            graph.pre_seed(_cfg("store-rewrite"), {"events": [_OldNote()]})

            @migrate_from(_OldNote.__qualname__, in_module=_OldNote.__module__)
            class _NewNote(IntegrationEvent):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _NewNote))
            store = EventStore(EventTopology(_HANDLERS), saver)

            planned = store.plan_rewrite(thread_ids=["store-rewrite"])
            written = store.rewrite_store(thread_ids=["store-rewrite"])

            assert [t.status for t in planned.threads] == ["rewrite"]
            assert written.applied
            assert store.unrevivable_threads(thread_ids=["store-rewrite"]) == {}
            assert [
                t.status
                for t in store.plan_rewrite(thread_ids=["store-rewrite"]).threads
            ] == ["unchanged"]

    def when_a_handler_runs_through_the_store():
        def it_raises_an_error_that_names_the_store():
            store = EventStore(EventTopology(_HANDLERS), MemorySaver())

            with pytest.raises(RuntimeError) as info:
                store._compile().invoke(
                    {"events": [Started(data="x")], "causes": [None]},
                    config=_cfg("store-runs"),
                )

            assert str(info.value) == (
                "EventStore cannot run handler '_ask'. A store has no services. "
                "Run the thread with an EventGraph."
            )

    def when_the_checkpointer_is_not_a_saver():
        def it_raises_a_type_error():
            with pytest.raises(TypeError, match="BaseCheckpointSaver"):
                EventStore(EventTopology(_HANDLERS), None)  # type: ignore[arg-type]


def describe_MigrationTarget():
    def when_the_factory_returns_a_store():
        def it_passes_against_the_baseline_of_the_graph(tmp_path: Path):
            def graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
                saver.serde = NamespaceAwareSerde(events=(Started, Ended))
                return EventGraph(_HANDLERS, services=[_Mailer()], checkpointer=saver)

            def store(saver: BaseCheckpointSaver[Any]) -> EventStore:
                saver.serde = NamespaceAwareSerde(events=(Started, Ended))
                return EventStore(EventTopology(_HANDLERS), saver)

            baseline = tmp_path / "baseline.json"
            update_baseline(MigrationTarget(graph=graph, baseline=baseline))

            report = check(MigrationTarget(graph=store, baseline=baseline))

            assert report.findings == ()

    def when_the_store_serde_comes_from_the_topology():
        def it_holds_the_migrations_of_the_graph_serde():
            class _OldPing(IntegrationEvent):
                pass

            @migrate_from(_OldPing.__qualname__, in_module=_OldPing.__module__)
            class _Ping(IntegrationEvent):
                pass

            @on(_Ping)
            def react(event) -> None:  # type: ignore[no-untyped-def]
                return None

            def graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
                return EventGraph.from_namespaces(handlers=[react], checkpointer=saver)

            def store(saver: BaseCheckpointSaver[Any]) -> EventStore:
                topology = EventTopology([react])
                saver.serde = topology.serde()
                return EventStore(topology, saver)

            from_graph = MigrationTarget(graph=graph, baseline=Path("unused"))
            from_store = MigrationTarget(graph=store, baseline=Path("unused"))

            assert len(from_store.snapshot().migrations.keys) == 1
            assert (
                from_store.snapshot().migrations.fingerprint
                == from_graph.snapshot().migrations.fingerprint
            )
