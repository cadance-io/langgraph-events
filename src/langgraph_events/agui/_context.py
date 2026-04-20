"""MapperContext — shared state during one AG-UI stream."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    _open_tool_calls: dict[str, dict[int, str]] = dataclasses.field(
        default_factory=dict, repr=False
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

    def ensure_tool_call_id(
        self,
        run_id: str,
        index: int,
        tool_call_id: str,
        name: str,
    ) -> tuple[str, bool]:
        """Track a streaming tool-call chunk and report whether it is new.

        Returns ``(resolved_tool_call_id, is_new)``. The first chunk for a
        given ``(run_id, index)`` records the id; later chunks look it up
        instead of overwriting with the empty string that continuation
        chunks typically carry.

        Raises ``ValueError`` if the first chunk for an ``(run_id, index)``
        carries an empty ``tool_call_id`` or ``name`` — the OpenAI streaming
        contract requires both on the first chunk of each call.
        """
        per_run = self._open_tool_calls.setdefault(run_id, {})
        existing = per_run.get(index)
        if existing is not None:
            return existing, False
        if not tool_call_id:
            raise ValueError(
                f"First tool_call_chunk for (run_id={run_id!r}, "
                f"index={index}) carries no 'id'. The OpenAI streaming "
                "contract requires the id in the first chunk of each "
                "tool call."
            )
        if not name:
            raise ValueError(
                f"First tool_call_chunk for (run_id={run_id!r}, "
                f"index={index}, id={tool_call_id!r}) carries no 'name'. "
                "Required on the first chunk per the OpenAI streaming "
                "contract."
            )
        per_run[index] = tool_call_id
        return tool_call_id, True

    def close_tool_calls_for_run(self, run_id: str) -> list[str]:
        """Close and return tool_call_ids opened under *run_id*."""
        per_run = self._open_tool_calls.pop(run_id, {})
        return list(per_run.values())

    def drain_open_tool_call_ids(self) -> list[str]:
        """Close and return every open tool_call_id, for error/drain paths."""
        ids: list[str] = []
        for per_run in self._open_tool_calls.values():
            ids.extend(per_run.values())
        self._open_tool_calls.clear()
        return ids

    def current_stream_message_id(self, run_id: str) -> str | None:
        """Return the currently-open text message id for *run_id*, if any."""
        return self._stream_message_ids.get(run_id)
