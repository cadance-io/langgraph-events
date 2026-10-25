"""Write the stored bytes of each framework event, as an old release wrote them.

Run it with the old release's ``src`` first on ``PYTHONPATH``::

    git archive v0.40.0 src | tar -x -C /tmp/v0.40.0
    PYTHONPATH=/tmp/v0.40.0/src uv run python scripts/generate_framework_fixtures.py \
        tests/fixtures/framework_events/v0_40_0

The output feeds ``tests/test_serde_framework.py``. Each file holds the
msgpack bytes that ``dumps_typed`` wrote for one event. The file stem is
the event's label. The bytes stay binary so that no text scanner reads
them as a secret.

``tests/fixtures/framework_events/v0_24_0`` comes from v0.24.0 in the same
way. That release predates ``abandoned_for_deadline``, so it holds a
``HandlerRaised`` payload without the key.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import langgraph_events as lge
from langgraph_events import Cancelled, HandlerRaised
from langgraph_events.serde import NamespaceAwareSerde


def _handler_raised(**flag: bool) -> HandlerRaised:
    return HandlerRaised(
        handler="work", source_event=Cancelled(), exception=ValueError("boom"), **flag
    )


_FIELDS: dict[str, dict[str, object]] = {
    "Halted": {},
    "MaxRoundsExceeded": {"rounds": 7},
    "Cancelled": {},
    "Abandoned": {"reason": "stale", "discarded": "Approval"},
    "Unresumable": {"resume_value": "Approval"},
    "RunPaused": {"elapsed_seconds": 2.5},
    "Interrupted": {},
    "Resumed": {"value": Cancelled(), "interrupted": lge.Interrupted()},
    "HandlerRetried": {
        "handler": "work",
        "source_event": Cancelled(),
        "exception": ValueError("boom"),
        "attempt": 1,
        "delay_seconds": 0.5,
    },
    "InvariantViolated": {"handler": "work"},
}


def _events() -> dict[str, object]:
    # An old release can lack a class. The fixture then has no entry for it.
    events: dict[str, object] = {
        name: getattr(lge, name)(**fields)
        for name, fields in _FIELDS.items()
        if hasattr(lge, name)
    }
    names = {f.name for f in dataclasses.fields(HandlerRaised)}
    if "abandoned_for_deadline" in names:
        events["HandlerRaised-deadline"] = _handler_raised(abandoned_for_deadline=True)
        events["HandlerRaised-no-deadline"] = _handler_raised(
            abandoned_for_deadline=False
        )
    else:
        events["HandlerRaised-no-key"] = _handler_raised()
    return events


def main(out: Path) -> None:
    serde = NamespaceAwareSerde()
    out.mkdir(parents=True, exist_ok=True)
    events = _events()
    for label, event in events.items():
        type_, data = serde.dumps_typed(event)
        if type_ != "msgpack":
            raise SystemExit(f"{label} was written as {type_!r}, not msgpack.")
        (out / f"{label}.msgpack").write_bytes(data)
    version = Path(lge.__file__).resolve().parents[1]
    print(f"{len(events)} events from {version}", file=sys.stderr)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
