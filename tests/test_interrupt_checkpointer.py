"""An ``Interrupted`` event needs a checkpointer to keep the pause."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_events import (
    EventGraph,
    IntegrationEvent,
    Interrupted,
    InterruptWithoutCheckpointerError,
    on,
)


class Asked(IntegrationEvent):
    pass


class ApprovalRequested(Interrupted):
    draft: str = ""


@on(Asked)
def ask(event: Asked) -> ApprovalRequested:
    return ApprovalRequested(draft="d")


def describe_Interrupted():
    def when_the_graph_has_no_checkpointer():
        def it_raises_naming_the_interrupt():
            with pytest.raises(
                InterruptWithoutCheckpointerError, match="ApprovalRequested"
            ):
                EventGraph([ask]).invoke(Asked())

        async def it_raises_on_the_async_path():
            with pytest.raises(InterruptWithoutCheckpointerError):
                await EventGraph([ask]).ainvoke(Asked())

    def when_the_graph_has_a_checkpointer():
        def it_pauses():
            graph = EventGraph([ask], checkpointer=InMemorySaver())
            config = {"configurable": {"thread_id": "t1"}}
            graph.invoke(Asked(), config=config)
            assert graph.get_state(config).interrupted == ApprovalRequested(draft="d")
