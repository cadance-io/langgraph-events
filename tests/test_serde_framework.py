"""lge owns the read migrations for its own framework events.

The fixtures hold bytes that real old releases wrote. Regenerate them with
``scripts/generate_framework_fixtures.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import ormsgpack
import pytest

from langgraph_events import (
    Cancelled,
    DomainEvent,
    Event,
    HandlerAbandonedOnDeadline,
    HandlerRaised,
    Interrupted,
    Namespace,
    RunPausedOnDeadline,
)
from langgraph_events.serde import Migration, NamespaceAwareSerde
from langgraph_events.serde._jsonplus import EXT_INTERRUPT

FIXTURES = Path(__file__).parent / "fixtures"


class ClientNS(Namespace):
    """A client serde passes its own namespaces only, never lge's events."""

    class Ping(DomainEvent):
        n: int = 0


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
            "Unresumable",
        )
    },
    "HandlerRaised-no-key": ("HandlerRaised", RAISED),
}

EXPECTED = {"v0_40_0": V0_40_0, "v0_24_0": V0_24_0}

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
                HandlerRaised(handler="work", source_event=Cancelled()),
                RunPausedOnDeadline(elapsed_seconds=2.5),
            ],
            ids=lambda event: type(event).__name__,
        )
        def it_revives_what_it_writes(client_serde, event):
            assert client_serde.loads_typed(client_serde.dumps_typed(event)) == event

    def describe_writing():
        def it_keeps_the_live_identity(client_serde):
            event = HandlerAbandonedOnDeadline(handler="work")
            assert _written_identity(client_serde, event) == [
                "langgraph_events._event",
                "HandlerAbandonedOnDeadline",
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
