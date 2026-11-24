"""A decorated value class, in its own module: a serde that reads another
module of the tests must not warn about it."""

import enum

from langgraph_events import IntegrationEvent
from langgraph_events.serde import migrate_value_from

GONE = "gone_value_module"


@migrate_value_from(GONE)
class Moved(enum.StrEnum):
    ON = "on"


class HoldsMoved(IntegrationEvent):
    moved: Moved | None = None
