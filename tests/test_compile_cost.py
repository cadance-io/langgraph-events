"""Compiling a graph must not read handler source.

LangGraph inspects each node function's source to find nested graphs. An
EventGraph handler is a leaf, so that inspection is pure cost, paid again on
every new EventGraph over the same handlers.

The assertion counts source reads, not elapsed time. A wall-clock threshold
measured the same property, but coverage instrumentation roughly doubles the
time, so the pre-commit hook failed on a correct tree.
"""

import inspect

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


def describe_event_graph_compile():
    def when_the_same_handlers_are_compiled_repeatedly():
        def it_does_not_read_handler_source(monkeypatch):
            handlers = _handlers()
            # Warm up: the first compile of a process may read source for
            # reasons unrelated to the handler nodes.
            _ = EventGraph(handlers).compiled

            read: list[str] = []
            real = inspect.getsource

            def counting(obj):
                read.append(getattr(obj, "__qualname__", repr(obj)))
                return real(obj)

            monkeypatch.setattr(inspect, "getsource", counting)
            _ = EventGraph(handlers).compiled

            # Drop the leaf-node declaration and this list holds one entry per
            # handler, which is the regression the test exists to catch.
            assert read == []
