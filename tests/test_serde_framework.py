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
    HandlerAbandonedOnSignal,
    HandlerRaised,
    Interrupted,
    Namespace,
    RunPausedOnDeadline,
    RunPausedOnSignal,
)
from langgraph_events.serde import Migration, NamespaceAwareSerde

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


def _written_identity(serde: NamespaceAwareSerde, event: Event) -> list[str]:
    _, data = serde.dumps_typed(event)
    module, qualname, _ = ormsgpack.unpackb(
        data, ext_hook=lambda _code, payload: ormsgpack.unpackb(payload)
    )
    return [module, qualname]


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
