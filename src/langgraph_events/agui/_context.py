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

    def next_message_id(self) -> str:
        """Return a monotonically increasing message ID for this stream."""
        self._message_counter += 1
        return f"msg-{self._message_counter}"
