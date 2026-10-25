"""Write the stored bytes of each framework event, as one release wrote them.

Run it against a released source tree, with that tree first on
``PYTHONPATH``::

    git archive v0.40.0 src pyproject.toml | tar -x -C /tmp/v0.40.0
    PYTHONPATH=/tmp/v0.40.0/src uv run python \
        scripts/generate_framework_fixtures.py 0.40.0

The version argument must match the tree's ``pyproject.toml``. Without
``PYTHONPATH`` the script reads this checkout, which is the unreleased next
version: the argument must then be greater than its ``pyproject.toml``
version.

The output is ``tests/fixtures/framework_events/v<version>``, with each dot
replaced by ``_``. It feeds ``tests/test_serde_framework.py``. Each file
holds the msgpack bytes that ``dumps_typed`` wrote for one event. The file
stem is the event's label. The bytes stay binary so that no text scanner
reads them as a secret.

The class list is every concrete ``SystemEvent`` subclass and
``SystemPromptSet`` in ``langgraph_events.__all__``. A class with no entry in
``BUILDERS`` stops the script.
"""

from __future__ import annotations

import dataclasses
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage

import langgraph_events as lge
from langgraph_events import Cancelled, SystemEvent, SystemPromptSet
from langgraph_events.serde import NamespaceAwareSerde

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "tests" / "fixtures" / "framework_events"


def _one(**fields: object) -> Callable[[type], dict[str, object]]:
    return lambda cls: {cls.__name__: cls(**fields)}


def _raised(**extra: object) -> dict[str, object]:
    return {
        "handler": "work",
        "source_event": Cancelled(),
        "exception": ValueError("boom"),
        **extra,
    }


def _handler_raised(cls: type) -> dict[str, object]:
    # 0.25.0 to 0.40.0 stored abandoned_for_deadline on every HandlerRaised,
    # so the stored shape depends on the field set of the release.
    if "abandoned_for_deadline" in {f.name for f in dataclasses.fields(cls)}:
        return {
            "HandlerRaised-deadline": cls(**_raised(abandoned_for_deadline=True)),
            "HandlerRaised-no-deadline": cls(**_raised(abandoned_for_deadline=False)),
        }
    return {"HandlerRaised": cls(**_raised())}


BUILDERS: dict[str, Callable[[type], dict[str, object]]] = {
    "Abandoned": _one(reason="stale", discarded="Approval"),
    "Cancelled": _one(),
    "Halted": _one(),
    "HandlerAbandonedOnDeadline": _one(**_raised()),
    "HandlerAbandonedOnSignal": _one(**_raised()),
    "HandlerRaised": _handler_raised,
    "HandlerRetried": _one(**_raised(attempt=1, delay_seconds=0.5)),
    "Interrupted": _one(),
    "InvariantViolated": _one(handler="work"),
    "MaxRoundsExceeded": _one(rounds=7),
    "Resumed": _one(value=Cancelled(), interrupted=lge.Interrupted()),
    "RunPaused": _one(elapsed_seconds=2.5),
    "RunPausedOnDeadline": _one(elapsed_seconds=2.5),
    "RunPausedOnSignal": _one(elapsed_seconds=2.5),
    "SystemPromptSet": _one(message=SystemMessage(content="Be terse.")),
    "Unresumable": _one(resume_value="Approval"),
}


def _is_abstract(cls: type) -> bool:
    # A release before 0.41.0 has no abstract marker and no abstract
    # framework class below SystemEvent.
    return bool(vars(cls).get("__lge_abstract__", False))


def _framework_classes() -> list[type]:
    exported = (getattr(lge, name) for name in lge.__all__)
    return [
        cls
        for cls in exported
        if isinstance(cls, type)
        and (
            (issubclass(cls, SystemEvent) and cls is not SystemEvent)
            or cls is SystemPromptSet
        )
        and not _is_abstract(cls)
    ]


def _parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _check_version(expected: str, src: Path) -> None:
    pyproject = tomllib.loads((src.parent / "pyproject.toml").read_text())
    declared = pyproject["project"]["version"]
    if src == REPO / "src":
        if _parse(expected) <= _parse(declared):
            raise SystemExit(
                f"{src} is this checkout, at {declared}. Pass the unreleased "
                f"next version, or put a released tree first on PYTHONPATH."
            )
    elif expected != declared:
        raise SystemExit(f"{src} is release {declared}, not {expected}.")


def _events() -> dict[str, object]:
    classes = _framework_classes()
    missing = sorted(cls.__name__ for cls in classes if cls.__name__ not in BUILDERS)
    if missing:
        raise SystemExit(f"No builder for {', '.join(missing)}. Add one to BUILDERS.")
    return {
        label: event
        for cls in classes
        for label, event in BUILDERS[cls.__name__](cls).items()
    }


def main(version: str) -> None:
    src = Path(lge.__file__).resolve().parents[1]
    _check_version(version, src)
    out = OUT_ROOT / f"v{version.replace('.', '_')}"
    events = _events()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.msgpack"):
        stale.unlink()
    serde = NamespaceAwareSerde()
    for label, event in events.items():
        type_, data = serde.dumps_typed(event)
        if type_ != "msgpack":
            raise SystemExit(f"{label} was written as {type_!r}, not msgpack.")
        (out / f"{label}.msgpack").write_bytes(data)
    print(f"{len(events)} events from {src} into {out}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
