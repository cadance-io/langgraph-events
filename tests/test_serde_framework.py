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
    Interrupted,
    Namespace,
    RunPaused,
)
from langgraph_events.serde import Migration, NamespaceAwareSerde, _jsonplus

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

V0_40_0 = {
    "Abandoned": ("Abandoned", {"reason": "stale", "discarded": "Approval"}),
    "Cancelled": ("Cancelled", {}),
    "Halted": ("Halted", {}),
    "HandlerRaised-deadline": (
        "HandlerRaised",
        {
            "handler": "work",
            "source_event": Cancelled(),
            "exception": EXCEPTION,
            "abandoned_for_deadline": True,
        },
    ),
    "HandlerRaised-no-deadline": (
        "HandlerRaised",
        {
            "handler": "work",
            "source_event": Cancelled(),
            "exception": EXCEPTION,
            "abandoned_for_deadline": False,
        },
    ),
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
    "RunPaused": ("RunPaused", {"elapsed_seconds": 2.5}),
    "Unresumable": ("Unresumable", {"resume_value": "Approval"}),
}


@pytest.fixture
def client_serde() -> NamespaceAwareSerde:
    return NamespaceAwareSerde(namespaces=[ClientNS])


@pytest.fixture
def a_framework_split(monkeypatch):
    def bump(kw: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return RunPaused, {**kw, "elapsed_seconds": 99.0}

    split = Migration.split_event(
        "test-stand-in", source=RunPaused, select=bump, targets=(RunPaused,)
    )
    monkeypatch.setattr(_jsonplus, "FRAMEWORK_MIGRATIONS", (split,), raising=False)


def describe_framework_migrations():
    def describe_stored_framework_events():
        def when_v0_40_0_wrote_them():
            def it_expects_every_stored_entry():
                assert sorted(V0_40_0) == sorted(_stored("v0_40_0"))

            @pytest.mark.parametrize("label", sorted(V0_40_0))
            def it_revives_each_one_in_a_client_serde(client_serde, label):
                revived = _revived(client_serde, "v0_40_0", label)
                name, fields = V0_40_0[label]
                assert type(revived).__name__ == name
                assert _fields(revived) == fields

    def describe_reading():
        def when_lge_declares_a_split():
            def it_applies_it_in_a_client_serde(a_framework_split, client_serde):
                revived = _revived(client_serde, "v0_40_0", "RunPaused")
                assert revived == RunPaused(elapsed_seconds=99.0)

    def describe_writing():
        def when_lge_declares_a_split():
            def it_keeps_the_live_identity(a_framework_split, client_serde):
                assert _written_identity(client_serde, RunPaused()) == [
                    "langgraph_events._event",
                    "RunPaused",
                ]

    def describe_legacy_write():
        def when_only_lge_declares_a_split():
            def it_constructs(a_framework_split):
                NamespaceAwareSerde(namespaces=[ClientNS], legacy_write=True)

            def it_keeps_the_live_identity(a_framework_split):
                serde = NamespaceAwareSerde(namespaces=[ClientNS], legacy_write=True)
                assert _written_identity(serde, RunPaused()) == [
                    "langgraph_events._event",
                    "RunPaused",
                ]

        def when_the_client_declares_its_own_split():
            def it_refuses_to_construct(a_framework_split):
                split = Migration.split_event(
                    source=ClientNS.Ping,
                    select=lambda kw: None,
                    targets=(ClientNS.Ping,),
                )
                with pytest.raises(ValueError, match="SplitEvent"):
                    NamespaceAwareSerde(
                        [split], namespaces=[ClientNS], legacy_write=True
                    )
