"""The Reflection facade — deterministic queries over an EventLog."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph_events._reflection import _evidence, _text, _tool

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langgraph_events._event import Event
    from langgraph_events._event_log import EventLog
    from langgraph_events._namespace import NamespaceModel
    from langgraph_events._reducer import BaseReducer
    from langgraph_events._reflection._tool import QueryTool


class Reflection:
    """Deterministic query surface over an event log, for harnessing by an agent.

    Offers only facts — event listings, field dumps, static topology, reducer
    projections, and verdict-free evidence joins. Reasoning and correlation are
    the querying agent's job. Obtain via ``EventGraph.reflect(log)`` or by
    annotating a handler parameter with ``Reflection``.
    """

    __slots__ = ("_log", "_model", "_reducers", "_tool")

    def __init__(
        self,
        log: EventLog,
        *,
        model: NamespaceModel,
        reducers: Mapping[str, BaseReducer],
    ) -> None:
        self._log = log
        self._model = model
        self._reducers = reducers
        self._tool: QueryTool | None = None

    @property
    def log(self) -> EventLog:
        """The underlying event log, with its full query surface."""
        return self._log

    def tool(self) -> QueryTool:
        """The ``query_log`` tool — this surface packaged for an LLM agent.

        Built once per Reflection (log and model are immutable) and cached.
        """
        if self._tool is None:
            self._tool = _tool.build_tool(self)
        return self._tool

    def context(self, *, tail: int = 5) -> str:
        """A bounded context card for a prompt: run shape + recent events."""
        return _text.render_context(self._log, self._model, tail=tail)

    def overview(self) -> str:
        """Totals, counts by kind/namespace, seeds, anomalies, run status."""
        return _text.render_overview(self._log, self._model)

    def event(self, index: int) -> str:
        """Full-field dump of one event, plus kind/namespace/owning command."""
        return _text.render_event_detail(self._resolve_index(index), self._log)

    def evidence(self, event: Event | int) -> str:
        """Every deterministic fact bearing on how one event came to be.

        A verdict-free join: explicit instance links, the owning command,
        matching static edges with candidate instances, and the forward face.
        No cause is chosen — correlation is the querying agent's job.
        """
        return _evidence.render_evidence(
            self._resolve_index(event), self._log, self._model
        )

    def schema(self) -> str:
        """The static topology — what can cause what. A fact about the code."""
        return self._model.text()

    def state(self) -> dict[str, Any]:
        """Project each registered reducer over the full log.

        Uses ``BaseReducer.seed`` — the framework-blessed from-scratch
        projection. Reducers with custom merge fns (e.g. ``message_reducer``)
        may differ from live channel values, which apply the merge.
        """
        events = list(self._log)
        return {name: r.seed(events) for name, r in self._reducers.items()}

    def _resolve_index(self, event: Event | int) -> int:
        """The single index gate: validate, normalize negatives, or locate.

        Every ``#index`` consumer routes through here, so range messages and
        canonical (non-negative) indices have one home.
        """
        n = len(self._log)
        if isinstance(event, int):
            if not -n <= event < n:
                raise IndexError(
                    f"index {event} out of range "
                    f"(log has {n} events, valid: 0..{n - 1})"
                )
            return event + n if event < 0 else event
        index = _evidence.find_index(self._log, event)
        if index is None:
            raise ValueError(f"event {type(event).__name__} not found in this log")
        return index
