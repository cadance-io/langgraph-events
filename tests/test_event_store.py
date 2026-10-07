"""``Record``, ``MemoryEventStore`` and ``JsonlEventStore``: the store port."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys

import pytest

from langgraph_events.store import (
    JsonlEventStore,
    MemoryEventStore,
    Record,
    StoreLockedError,
)

FIRST = Record("shop", "Order.Placed", {"order": "A-1", "lines": [1, 2]})
SECOND = Record("shop", "Order.Shipped", {"order": {"$ref": 0}})


def describe_Record():
    def it_round_trips_through_one_json_line():
        line = SECOND.to_json()
        assert "\n" not in line
        assert Record.from_json(line) == SECOND

    def it_rejects_a_line_that_is_not_a_record():
        with pytest.raises(ValueError, match="not an event record"):
            Record.from_json('{"module": "shop", "fields": {}}')

    def it_rejects_non_finite_field_values():
        record = Record("shop", "Order.Placed", {"amount": float("nan")})
        with pytest.raises(ValueError):
            record.to_json()

    def it_rejects_a_nonstandard_json_constant():
        with pytest.raises(ValueError):
            Record.from_json(
                '{"module": "shop", "type": "Order.Placed", '
                '"fields": {"amount": Infinity}}'
            )


def describe_MemoryEventStore():
    def it_loads_what_it_appended_in_order():
        store = MemoryEventStore()
        store.append([FIRST])
        store.append([SECOND])
        assert store.load() == [FIRST, SECOND]


def describe_JsonlEventStore():
    def when_the_file_does_not_exist():
        def it_loads_nothing(tmp_path):
            with JsonlEventStore(tmp_path / "book.jsonl") as store:
                assert store.load() == []

        def it_creates_the_lock_file_only(tmp_path):
            path = tmp_path / "book.jsonl"
            with JsonlEventStore(path) as store:
                store.load()
            assert sorted(p.name for p in tmp_path.iterdir()) == ["book.jsonl.lock"]

        def it_creates_no_file_for_an_empty_append(tmp_path):
            path = tmp_path / "book.jsonl"
            with JsonlEventStore(path) as store:
                store.append([])
            assert not path.exists()

    def when_records_are_appended():
        def it_writes_one_stdlib_json_object_per_line(tmp_path):
            path = tmp_path / "book.jsonl"
            with JsonlEventStore(path) as store:
                store.append([FIRST, SECOND])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert rows[1] == {
                "module": "shop",
                "type": "Order.Shipped",
                "fields": {"order": {"$ref": 0}},
            }

        def it_gives_the_log_the_default_mode(tmp_path):
            path = tmp_path / "book.jsonl"
            with JsonlEventStore(path) as store:
                store.append([FIRST])
            umask = os.umask(0)
            os.umask(umask)
            assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask

        def it_loads_them_in_a_later_store(tmp_path):
            path = tmp_path / "book.jsonl"
            with JsonlEventStore(path) as store:
                store.append([FIRST])
                store.append([SECOND])
            with JsonlEventStore(path) as store:
                assert store.load() == [FIRST, SECOND]

    def when_the_last_line_is_torn():
        def it_ignores_the_torn_line(tmp_path):
            path = tmp_path / "book.jsonl"
            path.write_text(FIRST.to_json() + "\n" + SECOND.to_json()[:9])
            with JsonlEventStore(path) as store:
                assert store.load() == [FIRST]

        def it_truncates_the_torn_line_on_the_next_append(tmp_path):
            path = tmp_path / "book.jsonl"
            path.write_text(FIRST.to_json() + "\n" + SECOND.to_json()[:9])
            with JsonlEventStore(path) as store:
                store.append([SECOND])
            assert path.read_text() == FIRST.to_json() + "\n" + SECOND.to_json() + "\n"

    def when_a_complete_line_is_not_a_record():
        def it_raises_naming_the_line(tmp_path):
            path = tmp_path / "book.jsonl"
            path.write_text(FIRST.to_json() + "\n" + "[1, 2]\n")
            with (
                JsonlEventStore(path) as store,
                pytest.raises(ValueError, match=r"book\.jsonl:2: "),
            ):
                store.load()

    def when_the_append_fails():
        def it_raises_the_os_error_unwrapped(tmp_path):
            path = tmp_path / "book.jsonl"
            path.mkdir()
            with JsonlEventStore(path) as store, pytest.raises(IsADirectoryError):
                store.append([FIRST])

    def when_another_store_holds_the_lock():
        def it_raises_store_locked(tmp_path):
            path = tmp_path / "book.jsonl"
            with (
                JsonlEventStore(path),
                pytest.raises(StoreLockedError, match=r"book\.jsonl\.lock is locked"),
            ):
                JsonlEventStore(path)

        def it_opens_once_the_holder_closes(tmp_path):
            path = tmp_path / "book.jsonl"
            JsonlEventStore(path).close()
            with JsonlEventStore(path) as store:
                assert store.load() == []

        def it_closes_more_than_once(tmp_path):
            path = tmp_path / "book.jsonl"
            store = JsonlEventStore(path)
            store.close()
            store.close()
            with JsonlEventStore(path) as replacement:
                assert replacement.load() == []

        def it_raises_store_locked_in_a_second_process(tmp_path):
            path = tmp_path / "book.jsonl"
            probe = (
                "import sys\n"
                "from pathlib import Path\n"
                "from langgraph_events.store import JsonlEventStore, StoreLockedError\n"
                "try:\n"
                "    JsonlEventStore(Path(sys.argv[1]))\n"
                "except StoreLockedError:\n"
                "    sys.exit(3)\n"
            )
            with JsonlEventStore(path):
                locked = subprocess.run(  # noqa: S603
                    [sys.executable, "-c", probe, str(path)], check=False
                )
            assert locked.returncode == 3
