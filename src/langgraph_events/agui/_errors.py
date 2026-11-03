"""Exceptions that a caller of the AG-UI adapter can raise."""

from __future__ import annotations


class RunRefusedError(Exception):
    """Refuse an AG-UI run from a seed factory or a resume factory.

    The adapter ends the run with ``RUN_ERROR`` that carries *code* and
    *message*. The graph does not start, so the run stores nothing.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
