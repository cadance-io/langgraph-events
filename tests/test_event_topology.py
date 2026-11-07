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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph_events import (
    Abandoned,
    Command,
    DomainEvent,
    Event,
    EventGraph,
    EventStore,
    EventTopology,
    IntegrationEvent,
    Interrupted,
    Namespace,
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
    TargetError,
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


@on(Started)
def handle(event: Started) -> None:
    return None


_FIRST_HANDLE = handle


@on(Started)  # type: ignore[no-redef]
def handle(event: Started, mailer: _Mailer) -> None:
    return None


class _Shop(Namespace):
    class Unhandled(Command):
        class Done(DomainEvent):
            pass


_HANDLERS = [_ask, _finish]


def _cfg(tid: str) -> Any:
    return {"configurable": {"thread_id": tid}}


def _retired_gate() -> tuple[tuple[str, bytes], type[Interrupted]]:
    """A payload stored under a deleted interrupt, and the tombstone that a
    ``@migrate_from`` maps it onto. No handler touches the tombstone."""

    class _OldGate(Interrupted):
        pass

    stored = NamespaceAwareSerde(events=(_OldGate,)).dumps_typed(_OldGate())

    @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
    class _Tombstone(Interrupted):
        pass

    return stored, _Tombstone


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
                ServiceRequirement(
                    node="_ask",
                    qualname="test_event_topology._ask",
                    param="mailer",
                    annotation=_Mailer,
                ),
                ServiceRequirement(
                    node="_finish",
                    qualname="test_event_topology._finish",
                    param="mailer",
                    annotation=_Mailer,
                ),
            )

        def it_names_the_node_of_a_renamed_duplicate():
            topology = EventTopology([_FIRST_HANDLE, handle])

            (requirement,) = topology.requirements

            assert (requirement.node, requirement.qualname) == (
                "handle_2",
                "test_event_topology.handle",
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

    def when_an_event_is_retired():
        def it_builds_a_serde_that_revives_it():
            stored, tombstone = _retired_gate()

            serde = EventTopology(_HANDLERS, retired=[tombstone]).serde()

            assert serde.loads_typed(stored) == tombstone()

        def it_lists_each_retired_event_once():
            topology = EventTopology(_HANDLERS, retired=[_Counted, _Counted])

            assert topology.retired_events == (_Counted,)

        def when_the_entries_come_from_a_generator():
            def it_lists_each_one():
                topology = EventTopology(_HANDLERS, retired=(c for c in [_Counted]))

                assert topology.retired_events == (_Counted,)

    def when_no_event_is_retired():
        def it_lists_no_retired_event():
            assert EventTopology(_HANDLERS).retired_events == ()

    def when_a_retired_entry_is_not_an_event_class():
        def it_raises_a_type_error():
            with pytest.raises(TypeError) as info:
                EventTopology(_HANDLERS, retired=[_Mailer])  # type: ignore[list-item]

            assert str(info.value) == (
                f"retired= expects Event subclasses, got {_Mailer!r}"
            )

    def when_a_retired_entry_is_abstract():
        def it_raises_a_type_error():
            with pytest.raises(TypeError) as info:
                EventTopology(_HANDLERS, retired=[IntegrationEvent])

            assert str(info.value) == (
                "retired= expects a concrete event class, got abstract "
                "IntegrationEvent. A stored thread cannot hold it."
            )

    def when_a_handler_returns_a_retired_event():
        def it_raises_a_value_error():
            with pytest.raises(ValueError) as info:
                EventTopology(_HANDLERS, retired=[_Approval])

            assert str(info.value) == (
                "Handler 'test_event_topology._ask' (node '_ask') returns "
                "_Approval, which retired= lists. A "
                "retired event is one that no handler returns. Stop returning "
                "it, or remove it from retired=."
            )


class _Desk(Namespace):
    class Book(Command):
        class Booked(DomainEvent):
            pass

        def handle(self) -> "_Desk.Book.Booked":
            return _Desk.Book.Booked()


@on(_Desk.Book.Booked)
def _confirm(event: _Desk.Book.Booked, mailer: _Mailer) -> None:
    mailer.send("booked")


def describe_EventTopology_from_namespaces():
    def when_a_namespace_has_an_inline_handler():
        def it_lists_the_handlers_of_from_namespaces():
            topology = EventTopology.from_namespaces(_Desk, handlers=[_confirm])
            graph = EventGraph.from_namespaces(
                _Desk, handlers=[_confirm], services=[_Mailer()]
            )

            assert topology.handler_names == graph.topology.handler_names
            assert topology.handler_names == {"_Desk.Book", "_confirm"}

        def it_needs_no_service():
            topology = EventTopology.from_namespaces(_Desk, handlers=[_confirm])

            assert [r.param for r in topology.requirements] == ["mailer"]

    def when_an_event_is_retired():
        def it_lists_the_retired_event():
            topology = EventTopology.from_namespaces(
                _Desk, handlers=[_confirm], retired=[_Counted]
            )

            assert topology.retired_events == (_Counted,)

    def when_an_argument_is_not_a_namespace():
        def it_raises_a_type_error():
            with pytest.raises(TypeError) as info:
                EventTopology.from_namespaces(_Mailer)  # type: ignore[arg-type]

            assert str(info.value) == (
                f"from_namespaces expects Namespace subclasses, got {_Mailer!r}"
            )


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
        def it_names_the_missing_type():
            with pytest.raises(TypeError) as info:
                EventGraph.of(EventTopology(_HANDLERS))

            assert str(info.value) == (
                "Handler 'test_event_topology._ask' (node '_ask') declares "
                "parameter(s) ['mailer'] that the framework cannot inject: "
                "'mailer' (_Mailer). services= has no service. For service "
                "injection, register a matching instance via "
                "EventGraph.of(..., services=); for state, register a Reducer; "
                "otherwise remove the parameter."
            )

        def when_a_name_key_is_a_near_miss():
            def it_lists_the_name_keys_it_received():
                with pytest.raises(TypeError) as info:
                    EventGraph.of(EventTopology(_HANDLERS), services={"mailr": 1})

                assert "services= has the name keys ['mailr']." in str(info.value)

        def when_the_constructor_raises():
            def it_names_the_constructor():
                with pytest.raises(TypeError) as info:
                    EventGraph(_HANDLERS, services=[1])

                assert "services= has the types ['int']." in str(info.value)
                assert "via EventGraph(services=)" in str(info.value)

    def when_a_renamed_duplicate_misses_a_service():
        def it_names_the_function_and_the_node():
            with pytest.raises(TypeError) as info:
                EventGraph([_FIRST_HANDLE, handle])

            assert str(info.value) == (
                "Handler 'test_event_topology.handle' (node 'handle_2') declares "
                "parameter(s) ['mailer'] that the framework cannot inject: "
                "'mailer' (_Mailer). services= has no service. For service "
                "injection, register a matching instance via "
                "EventGraph(services=); for state, register a Reducer; "
                "otherwise remove the parameter."
            )


def _bad_return(event: Started) -> Event:
    return Ended(result="x")


_SERVICE_COLLISION = (
    "EventGraph(services=...) has two instances of type '_Mailer' "
    "\u2014 same-type collision. Register only one instance, switch "
    "to the name-keyed mapping form (services={'primary': a, "
    "'backup': b}), or use a subclass to distinguish them."
)


def describe_EventGraph():
    def when_the_services_and_a_handler_are_both_invalid():
        def it_refuses_the_services_first():
            bad = on(Started)(_bad_return)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(TypeError) as info:
                    EventGraph([bad], services=[_Mailer(), _Mailer()])

            assert str(info.value) == _SERVICE_COLLISION
            assert not [w for w in caught if w.category is OrphanedEventWarning]

    def when_the_services_and_a_reducer_are_both_invalid():
        def it_refuses_the_services_first():
            reserved = ScalarReducer(name="events", event_type=_Counted, fn=len)

            with pytest.raises(TypeError) as info:
                EventGraph([_ask], reducers=[reserved], services=[_Mailer(), _Mailer()])

            assert str(info.value) == _SERVICE_COLLISION

    def when_the_services_and_a_command_are_both_invalid():
        def it_refuses_the_services_first():
            with pytest.raises(TypeError) as info:
                EventGraph([_Shop.Unhandled], services=[_Mailer(), _Mailer()])

            assert str(info.value) == _SERVICE_COLLISION

    def when_no_handler_is_given():
        def when_on_unresumable_is_invalid():
            def it_refuses_on_unresumable_first():
                with pytest.raises(ValueError) as info:
                    EventGraph([], on_unresumable="bogus")  # type: ignore[arg-type]

                assert str(info.value) == (
                    "on_unresumable must be 'raise', 'halt', or 'warn', got 'bogus'"
                )

        def when_the_services_are_invalid():
            def it_refuses_the_services_first():
                with pytest.raises(TypeError) as info:
                    EventGraph([], services=[_Mailer(), _Mailer()])

                assert str(info.value) == _SERVICE_COLLISION

    def when_the_graph_has_no_checkpointer():
        def it_has_no_event_store():
            with pytest.raises(ValueError) as info:
                _ = EventGraph(_HANDLERS, services=[_Mailer()]).event_store

            assert str(info.value) == (
                "event_store requires a checkpointer. Pass checkpointer= to "
                "EventGraph(...), or use EventStore(topology, saver)."
            )

    def when_from_namespaces_retires_an_event():
        def it_wires_a_serde_that_revives_it():
            stored, tombstone = _retired_gate()
            saver = MemorySaver()

            EventGraph.from_namespaces(
                _Desk,
                handlers=[_confirm],
                services=[_Mailer()],
                checkpointer=saver,
                retired=[tombstone],
            )

            assert saver.serde.loads_typed(stored) == tombstone()


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

    def when_the_saver_holds_another_serde():
        def it_keeps_the_serde():
            saver = MemorySaver()

            EventStore(EventTopology(_HANDLERS), saver)

            assert type(saver.serde) is JsonPlusSerializer

        def it_names_the_topology_serde_in_the_error():
            store = EventStore(EventTopology(_HANDLERS), MemorySaver())

            with pytest.raises(ValueError) as info:
                store.plan_rewrite(thread_ids=["t"])

            assert str(info.value) == (
                "plan_rewrite() needs a NamespaceAwareSerde on the checkpointer, "
                "found JsonPlusSerializer. Only that serde can report an identity "
                "it cannot revive. Set checkpointer.serde = topology.serde()."
            )

    def when_a_handler_runs_through_the_store():
        def it_raises_an_error_that_names_the_store():
            store = EventStore(EventTopology(_HANDLERS), MemorySaver())

            with pytest.raises(RuntimeError) as info:
                store._compiled.invoke(
                    {"events": [Started(data="x")], "causes": [None]},
                    config=_cfg("store-runs"),
                )

            assert str(info.value) == (
                "Handler 'test_event_topology._ask' (node '_ask') cannot run in "
                "an EventStore. A store has no services. Run the thread with an "
                "EventGraph."
            )

    def when_the_checkpointer_is_not_a_saver():
        def it_raises_a_type_error():
            with pytest.raises(TypeError, match="BaseCheckpointSaver"):
                EventStore(EventTopology(_HANDLERS), None)  # type: ignore[arg-type]


def describe_MigrationTarget():
    def when_the_factory_ignores_the_checkpointer():
        def it_names_the_graph_factory():
            def graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
                return EventGraph(_HANDLERS, services=[_Mailer()])

            with pytest.raises(TargetError) as info:
                MigrationTarget(graph=graph, baseline=Path("unused")).snapshot()

            assert str(info.value) == (
                "MigrationTarget.graph returned a graph with no checkpointer. "
                "Build the graph on the checkpointer that the factory receives."
            )

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

        def when_the_checkpointer_holds_the_default_serde():
            def it_sets_the_serde_of_the_topology():
                stored, tombstone = _retired_gate()
                saver = MemorySaver()

                def store(saver: BaseCheckpointSaver[Any]) -> EventStore:
                    return EventStore(
                        EventTopology(_HANDLERS, retired=[tombstone]), saver
                    )

                MigrationTarget(graph=store, baseline=Path("unused")).build(saver)

                assert saver.serde.loads_typed(stored) == tombstone()

    def when_the_topology_retires_an_event():
        def it_records_the_event_as_live():
            def graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
                return EventGraph.from_namespaces(
                    handlers=_HANDLERS,
                    services=[_Mailer()],
                    checkpointer=saver,
                    retired=[_Counted],
                )

            snapshot = MigrationTarget(graph=graph, baseline=Path("unused")).snapshot()

            assert _Counted.__qualname__ in {
                identity.qualname for identity in snapshot.live_fields()
            }

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
