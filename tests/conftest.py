"""Shared test fixtures and event definitions."""

from dataclasses import dataclass

from langgraph_events import Event


# --- Test event hierarchy ---

@dataclass(frozen=True)
class Ping(Event):
    value: str = ""


@dataclass(frozen=True)
class Pong(Event):
    value: str = ""


@dataclass(frozen=True)
class Done(Event):
    result: str = ""


# --- For branching tests ---

@dataclass(frozen=True)
class Request(Event):
    kind: str = ""
    data: str = ""


@dataclass(frozen=True)
class FastResult(Event):
    data: str = ""


@dataclass(frozen=True)
class SlowResult(Event):
    data: str = ""


# --- For inheritance / fan-out tests ---

@dataclass(frozen=True)
class Auditable(Event):
    action: str = ""


@dataclass(frozen=True)
class Processable(Event):
    item: str = ""


@dataclass(frozen=True)
class AuditableProcessable(Auditable, Processable):
    action: str = ""
    item: str = ""
