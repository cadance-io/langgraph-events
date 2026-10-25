"""lge owns the read migrations for its own framework events.

The fixtures hold bytes that real old releases wrote. Regenerate them with
``scripts/generate_framework_fixtures.py``.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import ormsgpack
import pytest
from langchain_core.messages import SystemMessage

import langgraph_events
from langgraph_events import (
    Cancelled,
    DomainEvent,
    Event,
    HandlerAbandonedOnDeadline,
    HandlerAbandonedOnSignal,
    HandlerRaised,
    Interrupted,
    Namespace,
    RunPausedOnDeadline,
    RunPausedOnSignal,
    SystemEvent,
    SystemPromptSet,
)
from langgraph_events._event import is_abstract_event
from langgraph_events.serde import Migration, NamespaceAwareSerde
from langgraph_events.serde._framework import FRAMEWORK_MIGRATIONS
from langgraph_events.serde._jsonplus import EXT_INTERRUPT

FIXTURES = Path(__file__).parent / "fixtures"


class ClientNS(Namespace):
    """A client serde passes its own namespaces only, never lge's events."""

    class Ping(DomainEvent):
        n: int = 0


def _releases() -> list[str]:
    folders = (FIXTURES / "framework_events").iterdir()
    return sorted(
        (f.name for f in folders),
        key=lambda name: tuple(int(n) for n in name[1:].split("_")),
    )


def _exported_framework_classes() -> set[str]:
    exported = (getattr(langgraph_events, name) for name in langgraph_events.__all__)
    return {
        cls.__qualname__
        for cls in exported
        if isinstance(cls, type)
        and (issubclass(cls, SystemEvent) or cls is SystemPromptSet)
        and not is_abstract_event(cls)
    }


def _kebab(qualname: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", qualname).lower()


def _stored(release: str) -> dict[str, tuple[str, bytes]]:
    folder = FIXTURES / "framework_events" / release
    return {path.stem: ("msgpack", path.read_bytes()) for path in folder.iterdir()}


def _revived(serde: NamespaceAwareSerde, release: str, label: str) -> Event:
    revived = serde.loads_typed(_stored(release)[label])
    assert isinstance(revived, Event)
    return revived


def _fields(event: Event) -> dict[str, Any]:
    return {f.name: getattr(event, f.name) for f in dataclasses.fields(event)}


def _identity(data: bytes) -> list[str]:
    """The identity of the outer stored event in *data*."""

    def outer(_code: int, payload: bytes) -> object:
        return ormsgpack.unpackb(payload, ext_hook=lambda _code, _payload: None)

    module, qualname, _ = ormsgpack.unpackb(data, ext_hook=outer)
    return [module, qualname]


def _written_identity(serde: NamespaceAwareSerde, event: Event) -> list[str]:
    return _identity(serde.dumps_typed(event)[1])


def _identities_in(data: bytes) -> list[list[str]]:
    """Every stored event identity in *data*, however deep it sits."""
    found: list[list[str]] = []

    def hook(_code: int, payload: bytes) -> object:
        unpacked = ormsgpack.unpackb(payload, ext_hook=hook)
        if isinstance(unpacked, list) and len(unpacked) == 3:
            found.append(unpacked[:2])
        return None

    ormsgpack.unpackb(data, ext_hook=hook)
    return found


def _ext(data: bytes) -> ormsgpack.Ext:
    return ormsgpack.unpackb(data, ext_hook=ormsgpack.Ext)


def _escalation_split() -> Migration:
    return Migration.split_event(
        source=HandlerRaised, select=lambda kw: None, targets=(HandlerRaised,)
    )


EXCEPTION = "ValueError('boom')"

RAISED = {"handler": "work", "source_event": Cancelled(), "exception": EXCEPTION}

PROMPT = {"message": SystemMessage(content="Be terse.")}

V0_40_0 = {
    "Abandoned": ("Abandoned", {"reason": "stale", "discarded": "Approval"}),
    "Cancelled": ("Cancelled", {}),
    "Halted": ("Halted", {}),
    "HandlerRaised-deadline": ("HandlerAbandonedOnDeadline", RAISED),
    "HandlerRaised-no-deadline": ("HandlerRaised", RAISED),
    "HandlerRetried": (
        "HandlerRetried",
        {
            "handler": "work",
            "source_event": Cancelled(),
            "exception": EXCEPTION,
            "attempt": 1,
            "delay_seconds": 0.5,
        },
    ),
    "Interrupted": ("Interrupted", {}),
    "InvariantViolated": (
        "InvariantViolated",
        {"invariant": None, "handler": "work", "source_event": None, "would_emit": []},
    ),
    "MaxRoundsExceeded": ("MaxRoundsExceeded", {"rounds": 7}),
    "Resumed": ("Resumed", {"value": Cancelled(), "interrupted": Interrupted()}),
    "RunPaused": ("RunPausedOnDeadline", {"elapsed_seconds": 2.5}),
    "SystemPromptSet": ("SystemPromptSet", PROMPT),
    "Unresumable": ("Unresumable", {"resume_value": "Approval"}),
}

# v0.24.0 predates the deadline flag, so its HandlerRaised has no key.
V0_24_0 = {
    **{
        label: V0_40_0[label]
        for label in (
            "Cancelled",
            "Halted",
            "Interrupted",
            "InvariantViolated",
            "MaxRoundsExceeded",
            "Resumed",
            "RunPaused",
            "SystemPromptSet",
            "Unresumable",
        )
    },
    "HandlerRaised": ("HandlerRaised", RAISED),
}

# The release this branch will be. Its own bytes revive unchanged.
V0_41_0 = {
    **{
        label: V0_40_0[label]
        for label in V0_40_0
        if label
        not in {"HandlerRaised-deadline", "HandlerRaised-no-deadline", "RunPaused"}
    },
    "HandlerAbandonedOnDeadline": ("HandlerAbandonedOnDeadline", RAISED),
    "HandlerAbandonedOnSignal": ("HandlerAbandonedOnSignal", RAISED),
    "HandlerRaised": ("HandlerRaised", RAISED),
    "RunPausedOnDeadline": ("RunPausedOnDeadline", {"elapsed_seconds": 2.5}),
    "RunPausedOnSignal": ("RunPausedOnSignal", {"elapsed_seconds": 2.5}),
}

EXPECTED = {"v0_41_0": V0_41_0, "v0_40_0": V0_40_0, "v0_24_0": V0_24_0}

STORED = [(release, label) for release in EXPECTED for label in EXPECTED[release]]


@pytest.fixture
def client_serde() -> NamespaceAwareSerde:
    return NamespaceAwareSerde(namespaces=[ClientNS])


def describe_framework_migrations():
    def describe_stored_framework_events():
        @pytest.mark.parametrize("release", sorted(EXPECTED))
        def it_expects_every_stored_entry(release):
            assert sorted(EXPECTED[release]) == sorted(_stored(release))

        @pytest.mark.parametrize(("release", "label"), STORED)
        def it_revives_each_one_in_a_client_serde(client_serde, release, label):
            revived = _revived(client_serde, release, label)
            name, fields = EXPECTED[release][label]
            assert type(revived).__name__ == name
            assert _fields(revived) == fields

    def describe_round_trip():
        @pytest.mark.parametrize(
            "event",
            [
                HandlerAbandonedOnDeadline(handler="work", source_event=Cancelled()),
                HandlerAbandonedOnSignal(handler="work", source_event=Cancelled()),
                HandlerRaised(handler="work", source_event=Cancelled()),
                RunPausedOnDeadline(elapsed_seconds=2.5),
                RunPausedOnSignal(elapsed_seconds=2.5),
            ],
            ids=lambda event: type(event).__name__,
        )
        def it_revives_what_it_writes(client_serde, event):
            assert client_serde.loads_typed(client_serde.dumps_typed(event)) == event

    def describe_writing():
        @pytest.mark.parametrize(
            "event",
            [
                HandlerAbandonedOnDeadline(handler="work"),
                HandlerAbandonedOnSignal(handler="work"),
                RunPausedOnDeadline(elapsed_seconds=2.5),
                RunPausedOnSignal(elapsed_seconds=2.5),
            ],
            ids=lambda event: type(event).__name__,
        )
        def it_keeps_the_live_identity(client_serde, event):
            assert _written_identity(client_serde, event) == [
                "langgraph_events._event",
                type(event).__qualname__,
            ]

    def describe_legacy_write():
        def when_only_lge_declares_a_split():
            def it_constructs():
                NamespaceAwareSerde(namespaces=[ClientNS], legacy_write=True)

            def it_keeps_the_live_identity():
                serde = NamespaceAwareSerde(namespaces=[ClientNS], legacy_write=True)
                event = HandlerAbandonedOnDeadline(handler="work")
                assert _written_identity(serde, event) == [
                    "langgraph_events._event",
                    "HandlerAbandonedOnDeadline",
                ]

        def when_the_client_declares_its_own_split():
            def it_refuses_to_construct():
                split = Migration.split_event(
                    source=ClientNS.Ping,
                    select=lambda kw: None,
                    targets=(ClientNS.Ping,),
                )
                with pytest.raises(ValueError, match="SplitEvent"):
                    NamespaceAwareSerde(
                        [split], namespaces=[ClientNS], legacy_write=True
                    )

    def describe_a_rewrite_of_a_0_40_0_checkpoint():
        # The events channel is written whole at each checkpoint. One step on
        # this release stores every old framework event of the thread again.
        def it_stores_each_old_event_under_its_new_identity(client_serde):
            stored = _stored("v0_40_0")
            history = [
                client_serde.loads_typed(stored["RunPaused"]),
                client_serde.loads_typed(stored["HandlerRaised-deadline"]),
            ]
            _, written = client_serde.dumps_typed(history)
            assert sorted(qualname for _, qualname in _identities_in(written)) == [
                "Cancelled",
                "HandlerAbandonedOnDeadline",
                "RunPausedOnDeadline",
            ]

        def it_writes_identities_that_0_40_0_never_stored(client_serde):
            # The v0_40_0 folder holds every concrete framework class that
            # 0.40.0 exports. An identity outside it has no class there.
            old = {tuple(_identity(data)) for _, data in _stored("v0_40_0").values()}
            history = [client_serde.loads_typed(_stored("v0_40_0")["RunPaused"])]
            _, written = client_serde.dumps_typed(history)
            new = {tuple(identity) for identity in _identities_in(written)}
            assert new.isdisjoint(old)

        def it_counts_the_old_event_as_touched_until_rewritten(client_serde):
            with client_serde._record_reads() as first:
                event = client_serde.loads_typed(_stored("v0_40_0")["RunPaused"])
            with client_serde._record_reads() as second:
                client_serde.loads_typed(client_serde.dumps_typed(event))
            assert [touched for *_, touched in first] == [True]
            assert [touched for *_, touched in second] == [False]

    def describe_inside_an_interrupt():
        def it_revives_an_old_HandlerRaised_in_the_value(client_serde):
            raised = _ext(_stored("v0_40_0")["HandlerRaised-deadline"][1])
            wrapped = ormsgpack.packb(
                ormsgpack.Ext(EXT_INTERRUPT, ormsgpack.packb((raised, "id-1")))
            )
            interrupt = client_serde.loads_typed(("msgpack", wrapped))
            assert type(interrupt.value) is HandlerAbandonedOnDeadline

    def describe_client_migrations():
        def when_the_client_passes_its_own_migrations():
            def it_still_applies_the_framework_split():
                serde = NamespaceAwareSerde(
                    [
                        Migration.rename(
                            old_module=__name__,
                            old_qualname="ClientNS.Gone",
                            to=ClientNS.Ping,
                        )
                    ],
                    namespaces=[ClientNS],
                )
                revived = serde.loads_typed(_stored("v0_40_0")["RunPaused"])
                assert type(revived) is RunPausedOnDeadline

        def when_the_client_splits_a_framework_identity():
            def it_refuses_to_construct():
                with pytest.raises(
                    ValueError, match=r"lge already migrates .*'HandlerRaised'"
                ):
                    NamespaceAwareSerde([_escalation_split()], namespaces=[ClientNS])

        def when_the_client_transforms_a_framework_split_source():
            # A class-stage transform runs before lge's split, so it could
            # drop the field that the split reads.
            def it_refuses_to_construct():
                transform = Migration.transform_fields(
                    target=HandlerRaised, transform=lambda kw: kw
                )
                with pytest.raises(
                    ValueError, match=r"lge already migrates .*'HandlerRaised'"
                ):
                    NamespaceAwareSerde([transform], namespaces=[ClientNS])

    def describe_fixture_gate():
        def it_stores_every_exported_framework_class_in_the_newest_release():
            stored = {
                _identity(data)[1] for _, data in _stored(_releases()[-1]).values()
            }
            assert _exported_framework_classes() - stored == set()

        @pytest.mark.parametrize(
            "operation",
            [op for m in FRAMEWORK_MIGRATIONS for op in m.operations],
            ids=lambda op: op.qualname,
        )
        def it_stores_the_source_of_each_framework_migration(operation):
            stored = {
                tuple(_identity(data))
                for release in _releases()
                for _, data in _stored(release).values()
            }
            assert (operation.module, operation.qualname) in stored

        @pytest.mark.parametrize(
            "migration", FRAMEWORK_MIGRATIONS, ids=lambda m: m.name
        )
        def it_names_each_framework_migration_by_the_rule(migration):
            (operation,) = migration.operations
            kind = {
                "RenameEvent": "rename",
                "AddField": "fill",
                "TransformFields": "transform",
                "SplitEvent": "split",
            }[type(operation).__name__]
            assert re.fullmatch(
                rf"lge-\d+\.\d+-{_kebab(operation.qualname)}-{kind}", migration.name
            )
