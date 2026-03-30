"""MapperContext — shared state during one AG-UI stream."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ag_ui.core import RunAgentInput


@dataclasses.dataclass
class MapperContext:
    """Carries shared state across mappers for a single stream.

    Attributes:
        run_id: The AG-UI run identifier.
        thread_id: The AG-UI thread identifier.
        input_data: The original ``RunAgentInput`` for the request.
    """

    run_id: str
    thread_id: str
    input_data: RunAgentInput
    _message_counter: int = dataclasses.field(default=0, repr=False)
    _stream_message_ids: dict[str, str] = dataclasses.field(
        default_factory=dict, repr=False
    )
    _open_stream_runs: set[str] = dataclasses.field(default_factory=set, repr=False)
    _streamed_ai_message_ids: set[str] = dataclasses.field(
        default_factory=set,
        repr=False,
    )
    _lc_to_stream_id: dict[str, str] = dataclasses.field(
        default_factory=dict,
        repr=False,
    )
    _emitted_message_ids: set[str] = dataclasses.field(
        default_factory=set,
        repr=False,
    )

    def next_message_id(self) -> str:
        """Return a monotonically increasing message ID for this stream."""
        self._message_counter += 1
        return f"msg-{self._message_counter}"

    def ensure_stream_message_id(self, llm_run_id: str) -> tuple[str, bool]:
        """Get or create message_id for an LLM stream run.

        Returns (message_id, is_new).
        """
        existing = self._stream_message_ids.get(llm_run_id)
        if existing is not None:
            return existing, False
        message_id = self.next_message_id()
        self._stream_message_ids[llm_run_id] = message_id
        self._open_stream_runs.add(llm_run_id)
        return message_id, True

    def close_stream_message_id(self, llm_run_id: str) -> str | None:
        """Mark a streamed message as closed and return its message_id."""
        message_id = self._stream_message_ids.pop(llm_run_id, None)
        if message_id is None:
            return None
        self._open_stream_runs.discard(llm_run_id)
        return message_id

    def drain_open_stream_message_ids(self) -> list[str]:
        """Close and return all message_ids still open."""
        open_run_ids = list(self._open_stream_runs)
        message_ids = [
            self._stream_message_ids[run_id]
            for run_id in open_run_ids
            if run_id in self._stream_message_ids
        ]
        for run_id in open_run_ids:
            self._stream_message_ids.pop(run_id, None)
        self._open_stream_runs.clear()
        return message_ids

    def record_lc_to_stream_id(self, lc_message_id: str, stream_id: str) -> None:
        """Map a LangChain message ID to its AG-UI streaming message ID."""
        self._lc_to_stream_id[lc_message_id] = stream_id

    @property
    def lc_id_overrides(self) -> Mapping[str, str]:
        """LangChain message ID -> AG-UI streaming ID overrides (read-only)."""
        return MappingProxyType(self._lc_to_stream_id)

    def mark_emitted_message(self, message_id: str) -> None:
        """Remember a LangChain message id emitted by MessageEventMapper."""
        self._emitted_message_ids.add(message_id)

    def was_emitted_message(self, message_id: str | None) -> bool:
        """Whether this LangChain message id was already emitted by a mapper."""
        if not message_id:
            return False
        return message_id in self._emitted_message_ids

    def mark_streamed_ai_message(self, message_id: str) -> None:
        """Remember a LangChain AI message id that was token-streamed."""
        self._streamed_ai_message_ids.add(message_id)

    def was_streamed_ai_message(self, message_id: str | None) -> bool:
        """Whether this LangChain AI message id already streamed via deltas."""
        if not message_id:
            return False
        return message_id in self._streamed_ai_message_ids
