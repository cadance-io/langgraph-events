"""A custom checkpointer with no SQL and no ``storage`` attribute.

It stands for a saver that a client writes against ``BaseCheckpointSaver``.
The base class does not promise that ``put`` replaces a stored checkpoint
of the same id, or that ``put_writes`` replaces an ``__interrupt__`` write.
The two flags turn each promise off.
"""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)


def _config(thread_id: str, ns: str, checkpoint_id: str) -> Any:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": ns,
            "checkpoint_id": checkpoint_id,
        }
    }


class DictSaver(BaseCheckpointSaver[int]):
    def __init__(
        self, *, put_overwrites: bool = True, writes_overwrite: bool = True
    ) -> None:
        super().__init__()
        self.put_overwrites = put_overwrites
        self.writes_overwrite = writes_overwrite
        self.rows: dict[tuple[str, str], dict[str, tuple[Any, Any, str | None]]] = {}
        self.writes: dict[tuple[str, str, str], dict[tuple[str, int], Any]] = {}

    def _tuple(self, thread_id: str, ns: str, checkpoint_id: str) -> CheckpointTuple:
        checkpoint, metadata, parent = self.rows[(thread_id, ns)][checkpoint_id]
        writes = self.writes.get((thread_id, ns, checkpoint_id), {})
        return CheckpointTuple(
            _config(thread_id, ns, checkpoint_id),
            self.serde.loads_typed(checkpoint),
            self.serde.loads_typed(metadata),
            _config(thread_id, ns, parent) if parent else None,
            [
                (task_id, channel, self.serde.loads_typed(value))
                for task_id, channel, value in writes.values()
            ],
        )

    def get_tuple(self, config: Any) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        rows = self.rows.get((thread_id, ns))
        if not rows:
            return None
        checkpoint_id = get_checkpoint_id(config) or max(rows)
        if checkpoint_id not in rows:
            return None
        return self._tuple(thread_id, ns, checkpoint_id)

    def list(self, config: Any, **_kwargs: Any) -> Iterator[CheckpointTuple]:
        wanted = (config or {}).get("configurable", {}).get("thread_id")
        for (thread_id, ns), rows in list(self.rows.items()):
            if wanted in (None, thread_id):
                for checkpoint_id in sorted(rows, reverse=True):
                    yield self._tuple(thread_id, ns, checkpoint_id)

    def put(
        self,
        config: Any,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Any:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        rows = self.rows.setdefault((thread_id, ns), {})
        if self.put_overwrites or checkpoint["id"] not in rows:
            rows[checkpoint["id"]] = (
                self.serde.dumps_typed(checkpoint),
                self.serde.dumps_typed(metadata),
                config["configurable"].get("checkpoint_id"),
            )
        return _config(thread_id, ns, checkpoint["id"])

    def put_writes(
        self,
        config: Any,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        c = config["configurable"]
        stored = self.writes.setdefault(
            (c["thread_id"], c.get("checkpoint_ns", ""), c["checkpoint_id"]), {}
        )
        for idx, (channel, value) in enumerate(writes):
            key = (task_id, WRITES_IDX_MAP.get(channel, idx))
            if key in stored and not (
                self.writes_overwrite and channel in WRITES_IDX_MAP
            ):
                continue
            stored[key] = (task_id, channel, self.serde.dumps_typed(value))

    async def aget_tuple(self, config: Any) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(self, config: Any, **kwargs: Any) -> AsyncIterator[CheckpointTuple]:
        for tup in self.list(config, **kwargs):
            yield tup

    async def aput(
        self,
        config: Any,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Any:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: Any,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)
