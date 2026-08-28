"""Defines a deprecated Namespace subclass so the warning's attributed
filename can be asserted from another module."""

from langgraph_events import Namespace


class Root(Namespace):
    pass


def define_child() -> type:
    class Child(Root):
        pass

    return Child
