"""Compiling a graph must not read handler source.

LangGraph inspects each node function's source to find nested graphs. An
EventGraph handler is a leaf, so that inspection is pure cost, paid again on
every new EventGraph over the same handlers.
"""

import time

from langgraph_events import Command, EventGraph, Namespace, on


class Compile(Namespace):
    class Do0(Command): ...

    class Do1(Command): ...

    class Do2(Command): ...

    class Do3(Command): ...

    class Do4(Command): ...

    class Do5(Command): ...

    class Do6(Command): ...

    class Do7(Command): ...

    class Do8(Command): ...

    class Do9(Command): ...


def _handlers() -> list:
    made = []
    for cmd in (getattr(Compile, f"Do{i}") for i in range(10)):

        @on(cmd)
        def handle(event) -> None:
            return None

        made.append(handle)
    return made


def _compile_seconds(handlers: list) -> float:
    started = time.perf_counter()
    _ = EventGraph(handlers).compiled
    return time.perf_counter() - started


def describe_event_graph_compile():
    def when_the_same_handlers_are_compiled_repeatedly():
        def it_does_not_pay_source_inspection_per_graph():
            handlers = _handlers()
            _ = EventGraph(handlers).compiled
            fastest = min(_compile_seconds(handlers) for _ in range(20))
            assert fastest < 0.006
