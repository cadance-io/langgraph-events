"""The reserved ``additional_kwargs`` slots for AG-UI passthrough data.

Stdlib only. The sibling modules defer their ``ag_ui`` and ``langchain``
imports into the functions that need them, so the constants and the inbound
collector live here rather than in ``_mappers``, which imports ``ag_ui`` at
module level.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

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

AGUI_EXTRAS_OPEN_FIELDS = frozenset({"metadata"})
"""Declared AG-UI message fields that use the extras slot.

AG-UI 1.0 declares each one as an open ``Metadata`` mapping, and 0.x carried
it as an extra field. Inbound, the field goes to the slot under its own name.
Outbound, a mapping entry with that name sets the declared field.
"""

AGUI_HELD_PARTS_KEY = "langgraph_events.agui.held_parts"
"""Reserved ``additional_kwargs`` key for the AG-UI content parts held off the model.

A held part is a part that the model does not receive: a media part with a
``file`` source. That handle names a file in a provider account, and the
server must not send it with its own credentials.

The value is a list. Each entry is ``{"position": i, "part": p}``. ``i`` is
the position of the part in the AG-UI content list. ``p`` is the part as JSON,
by alias, with its ``metadata``. Inbound, the converter writes the entries.
Outbound, each part goes back into the content list at its position. Read the
parts with :func:`~langgraph_events.agui.held_parts`.

The key is not ``index``. LangChain's ``merge_lists`` merges two list entries
that carry the same int ``index`` as one stream chunk, so
``merge_message_runs`` would fuse the held parts of two messages.

The value is client input. The trust note of :func:`collect_inbound_extras`
applies. The entries are measured against :data:`AGUI_EXTRAS_MAX_BYTES` on
their own. Over the cap, the held parts are dropped with a WARNING, and the
message still converts.
"""

AGUI_EXTRAS_MAX_BYTES = 8192
"""Size cap on the extra fields one inbound AG-UI message may carry.

Measured as the length of the JSON encoding of the whole extras mapping.
An inbound message over the cap loses its extras, with a WARNING naming the
message. The message itself still converts.

8 KiB holds roughly a hundred small entries or a few paragraphs of text, which
is the scale of the hint the field exists for. It also bounds the damage: a
thousand-message thread with every message at the cap is 8 MB of checkpoint,
which a checkpointer can still write. The held parts of
:data:`AGUI_HELD_PARTS_KEY` have a separate cap of the same size, so one
message can store up to 16 KiB, and that thread up to 16 MB. There is no lower
bound that a legitimate hint would breach and no higher one that keeps a
thread's growth bounded.
"""


def collect_inbound_extras(message: BaseModel) -> dict[str, Any]:
    """Return the ``additional_kwargs`` payload for an inbound AG-UI message.

    AG-UI models allow extra fields. Whatever the client sent beyond the
    declared schema is kept under :data:`AGUI_EXTRAS_KEY`, which is the same
    slot the outbound mapper reads. Each field in :data:`AGUI_EXTRAS_OPEN_FIELDS`
    goes to the same slot. A message with neither gets an empty
    ``additional_kwargs``.

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
    payload = dict(getattr(message, "model_extra", None) or {})
    for name in AGUI_EXTRAS_OPEN_FIELDS:
        value = getattr(message, name, None)
        if value is not None:
            payload[name] = value
    return capped_slot(
        AGUI_EXTRAS_KEY, payload, "extra fields", getattr(message, "id", None)
    )


def capped_slot(
    key: str,
    payload: dict[str, Any] | list[dict[str, Any]],
    what: str,
    message_id: str | None,
) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        size = len(json.dumps(payload))
    except (TypeError, ValueError, RecursionError) as exc:
        logger.warning(
            "Dropping the %s of inbound AG-UI message %r — "
            "additional_kwargs[%r] must encode as JSON (%s).",
            what,
            message_id,
            key,
            exc,
        )
        return {}
    if size > AGUI_EXTRAS_MAX_BYTES:
        logger.warning(
            "Dropping the %s of inbound AG-UI message %r — "
            "additional_kwargs[%r] holds %d bytes of JSON, over the "
            "AGUI_EXTRAS_MAX_BYTES cap of %d.",
            what,
            message_id,
            key,
            size,
            AGUI_EXTRAS_MAX_BYTES,
        )
        return {}
    return {key: payload}
