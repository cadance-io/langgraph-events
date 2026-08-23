"""The reserved ``additional_kwargs`` slot for AG-UI passthrough fields.

Stdlib only. The sibling modules defer their ``ag_ui`` and ``langchain``
imports into the functions that need them, so the constants and the inbound
collector live here rather than in ``_mappers``, which imports ``ag_ui`` at
module level.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

AGUI_EXTRAS_KEY = "langgraph_events.agui"
"""Reserved ``BaseMessage.additional_kwargs`` key for AG-UI passthrough fields.

A consumer puts a mapping under this key on a LangChain message. Every entry
becomes an extra field on the AG-UI message the adapter sends to the client.
The same key receives the client's extra fields on the way back in. Nothing
else in ``additional_kwargs`` crosses the wire.

``additional_kwargs`` is a shared namespace — LangChain provider integrations
write their own keys into it. The key is therefore package-qualified and holds
a dot, so it is not a Python identifier and no integration can produce it as a
keyword argument. Read the constant rather than the literal.
"""

AGUI_EXTRAS_MAX_BYTES = 8192
"""Size cap on the extra fields one inbound AG-UI message may carry.

Measured as the length of the JSON encoding of the whole extras mapping.
An inbound message over the cap loses its extras, with a WARNING naming the
message. The message itself still converts.

8 KiB holds roughly a hundred small entries or a few paragraphs of text, which
is the scale of the hint the field exists for. It also bounds the damage: a
thousand-message thread with every message at the cap is 8 MB of checkpoint,
which a checkpointer can still write. There is no lower bound that a legitimate
hint would breach and no higher one that keeps a thread's growth bounded.
"""


def collect_inbound_extras(message: Any) -> dict[str, Any]:
    """Return the ``additional_kwargs`` payload for an inbound AG-UI message.

    AG-UI models allow extra fields. Whatever the client sent beyond the
    declared schema is kept under :data:`AGUI_EXTRAS_KEY`, which is the same
    slot the outbound mapper reads. A message with no extra fields gets an
    empty ``additional_kwargs``.

    **This is a trust boundary.** The value is whatever the client put on the
    wire. It enters ``additional_kwargs``, flows through ``add_messages`` into
    the checkpoint, and is served back out on the next ``MessagesSnapshot`` —
    to every client on that thread, not only the one that sent it. The adapter
    neither filters the keys nor inspects the values. Two consequences:

    - Do not treat anything read out of this slot as trusted input.
    - On a thread more than one person can reach, one person's data is stored
      and served to the others. Strip or scope the slot yourself if that
      matters.

    The cap in :data:`AGUI_EXTRAS_MAX_BYTES` bounds how much a client can add
    per message. Extras that exceed it, or that cannot be measured at all, are
    dropped with a WARNING. Dropping rather than raising is deliberate: a raise
    would let a client break its own resume, and the same value would arrive
    again on every retry.
    """
    extra = getattr(message, "model_extra", None)
    if not extra:
        return {}
    payload = dict(extra)
    message_id = getattr(message, "id", None)
    try:
        size = len(json.dumps(payload))
    except (TypeError, ValueError, RecursionError) as exc:
        logger.warning(
            "Dropping the extra fields on inbound AG-UI message %r — they do "
            "not encode as JSON (%s).",
            message_id,
            exc,
        )
        return {}
    if size > AGUI_EXTRAS_MAX_BYTES:
        logger.warning(
            "Dropping the extra fields on inbound AG-UI message %r — %d bytes "
            "of JSON exceeds the %d byte cap.",
            message_id,
            size,
            AGUI_EXTRAS_MAX_BYTES,
        )
        return {}
    return {AGUI_EXTRAS_KEY: payload}
