"""A stand-in for library code: emits a warning from varying call depths.

Lives outside the package, so tests monkeypatch the package root to point
here — that is the only way to exercise "anchor outside the library" without
adding throwaway warnings to real library modules.
"""

from typing import Any


def shallow(emit: Any) -> None:
    emit("boom")


def deep(emit: Any) -> None:
    _deeper(emit)


def _deeper(emit: Any) -> None:
    emit("boom")
