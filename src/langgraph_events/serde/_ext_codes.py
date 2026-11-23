"""The msgpack ext codes that this library owns."""

# Unique among LangGraph's existing ext codes (currently 0..6).
EXT_NAMESPACE_AWARE_EVENT = 100
# Dedicated code so we can recurse via our own ``_default`` when re-encoding
# the wrapped value — LangGraph's generic dataclass path uses ``_msgpack_enc``
# which is hardcoded to ``default=_msgpack_default`` and would bypass us.
EXT_INTERRUPT = 101
# A ``FoldReducer`` channel write. LangGraph stores it as a pending write when
# a sibling task interrupts or fails in the same superstep. Upstream's
# dataclass path would encode its events by ``__name__``, so a nested event
# would not revive.
EXT_FOLD_CONTRIBUTIONS = 102
