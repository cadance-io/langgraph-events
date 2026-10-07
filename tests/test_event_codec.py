"""``EventCodec``: events to records and back, with links and migrations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import _store_scenarios as sc
import pytest
from _note_fixture import Note
from conftest import Order

from langgraph_events import Event, HandlerRaised, IntegrationEvent, RunPaused
from langgraph_events.serde import UnrevivedIdentity
from langgraph_events.store import EventCodec, Record

TESTS = Path(__file__).parent


class Request(IntegrationEvent):
    what: str = ""


class Granted(IntegrationEvent):
    request: Event | None = None


class Tagged(IntegrationEvent):
    data: object = None


def _scenario(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(TESTS)}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", "import _store_scenarios as s; s.main()", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _row(cls: type[Event], **fields: object) -> Record:
    return Record(cls.__module__, cls.__qualname__, fields)


def describe_EventCodec():
    def describe_encode():
        def it_stores_a_link_to_an_earlier_event_as_a_ref():
            request = Request(what="leave")
            records = EventCodec().encode([request, Granted(request=request)])
            assert records[1].fields == {"request": {"$ref": 0}}

        def it_numbers_a_later_batch_after_the_first():
            codec = EventCodec()
            request = Request(what="leave")
            codec.encode([Request(), request])
            assert codec.encode([Granted(request=request)])[0].fields == {
                "request": {"$ref": 1}
            }

        def it_escapes_a_dict_keyed_by_a_dollar_name():
            record = EventCodec().encode([Tagged(data={"$ref": 7, "a": (1, 2)})])[0]
            assert record.fields == {
                "data": {"$dict": {"$ref": 7, "a": {"$tuple": [1, 2]}}}
            }

        def it_stores_a_value_outside_json_as_its_repr():
            raised = HandlerRaised(handler="h", exception=KeyError("k"))
            record = EventCodec().encode([raised])[0]
            assert record.fields["exception"] == {"$repr": "KeyError('k')"}

    def describe_decode():
        def when_every_record_revives():
            def it_revives_a_ref_as_the_same_object():
                request = Request(what="leave")
                records = EventCodec().encode([request, Granted(request=request)])
                log = EventCodec().decode(records)
                assert log[1].request is log[0]

            def it_round_trips_the_markers():
                event = Tagged(data={"$ref": 7, "a": (1, 2), "b": [{"c": None}]})
                log = EventCodec().decode(EventCodec().encode([event]))
                assert log[0] == event

            def it_applies_the_migrations():
                log = EventCodec(events=[Note]).decode([_row(Note, text="hi")])
                assert log[0] == Note(text="hi", tag="legacy")
                old = Record(Note.__module__, "OldNote", {"text": "hi"})
                assert EventCodec(events=[Note]).decode([old])[0] == Note(
                    text="hi", tag="legacy"
                )

            def it_resolves_a_system_event_stored_under_the_package():
                row = Record("langgraph_events", "RunPaused", {"elapsed_seconds": 1.0})
                assert EventCodec().decode([row])[0] == RunPaused(elapsed_seconds=1.0)

            def it_round_trips_a_system_event_under_its_own_module():
                records = EventCodec().encode([RunPaused(elapsed_seconds=2.0)])
                assert records[0].module == "langgraph_events._event"
                log = EventCodec().decode(records)
                assert log[0] == RunPaused(elapsed_seconds=2.0)

            def it_round_trips_a_namespace_domain_event():
                records = EventCodec().encode([Order.Shipped(tracking="T-1")])
                assert records[0].type == "Order.Shipped"
                log = EventCodec(namespaces=[Order]).decode(records)
                assert log[0] == Order.Shipped(tracking="T-1")

            def it_resolves_a_ref_into_an_earlier_decode_call():
                codec = EventCodec()
                codec.decode([_row(Request, what="a")])
                log = codec.decode([_row(Granted, request={"$ref": 0})])
                assert log[0] == Granted(request=Request(what="a"))

        def when_a_ref_does_not_point_to_an_earlier_event():
            @pytest.mark.parametrize("pointer", [1, 2, -1])
            def it_raises_naming_the_record(pointer):
                rows = [_row(Request), _row(Granted, request={"$ref": pointer})]
                with pytest.raises(ValueError, match=r"record #1 .*must point"):
                    EventCodec().decode(rows)

            @pytest.mark.parametrize("pointer", ["0", True, 0.0])
            def it_raises_on_a_pointer_that_is_not_an_integer(pointer):
                rows = [_row(Request), _row(Granted, request={"$ref": pointer})]
                with pytest.raises(ValueError, match="is not an integer"):
                    EventCodec().decode(rows)

            def it_keeps_no_event_of_the_failed_call():
                codec = EventCodec()
                with pytest.raises(ValueError):
                    codec.decode([_row(Request), _row(Granted, request={"$ref": 5})])
                rows = [_row(Request, what="b"), _row(Granted, request={"$ref": 0})]
                assert codec.decode(rows)[1].request == Request(what="b")

        def when_the_class_is_unknown():
            def it_raises_naming_the_record():
                rows = [_row(Request), Record("gone.module", "Gone", {})]
                with pytest.raises(
                    ValueError, match=r"record #1 gone\.module\.Gone: Cannot revive"
                ):
                    EventCodec().decode(rows)

            def it_degrades_inside_tolerate_unresolved():
                codec = EventCodec()
                rows = [_row(Request), Record("gone.module", "Gone", {"a": 1})]
                with codec.tolerate_unresolved() as missing:
                    log = codec.decode(rows)
                gone = UnrevivedIdentity(module="gone.module", qualname="Gone")
                assert list(log) == [Request(), gone]
                assert missing == [gone]

        def when_a_replay_function_is_registered():
            def it_revives_a_runtime_class_in_a_fresh_process(tmp_path):
                path = str(tmp_path / "book.jsonl")
                assert _scenario("write_runtime_class", path).returncode == 0
                read = _scenario("read_runtime_class", path)
                assert read.returncode == 0, read.stderr
                assert json.loads(read.stdout) == {
                    "types": ["Define", "Defined", "Runtime.Widget"],
                    "size": 3,
                    "importable": False,
                }

            def it_registers_the_classes_before_the_next_record():
                rows = [
                    _row(sc.Defined, name="Gadget", field="size"),
                    Record(sc.__name__, "Runtime.Gadget", {"size": 2}),
                ]
                log = sc.runtime_codec().decode(rows)
                assert type(log[1]).__qualname__ == "Runtime.Gadget"

        def when_a_replay_function_raises():
            def it_raises_naming_the_record():
                def broken(event: Event) -> list[type[Event]]:
                    raise RuntimeError("factory failed")

                codec = EventCodec(replay={sc.Defined: broken})
                rows = [_row(sc.Defined, name="W", field="size")]
                with pytest.raises(ValueError, match=r"record #0 .*factory failed"):
                    codec.decode(rows)

        def when_a_replay_function_exits():
            def it_lets_system_exit_through():
                def leave(event: Event) -> list[type[Event]]:
                    raise SystemExit(4)

                codec = EventCodec(replay={sc.Defined: leave})
                with pytest.raises(SystemExit):
                    codec.decode([_row(sc.Defined, name="W", field="size")])

        def when_a_marker_holds_the_wrong_shape():
            def it_raises_naming_the_marker():
                with pytest.raises(ValueError, match=r"record #0 .*\$tuple holds int"):
                    EventCodec().decode([_row(Tagged, data={"$tuple": 3})])

    def describe_register():
        def it_revives_a_registered_class_ahead_of_the_import_walk():
            widget = sc.make_runtime_class("Sprocket", "teeth")
            codec = EventCodec()
            codec.register(widget)
            log = codec.decode([Record(sc.__name__, "Runtime.Sprocket", {"teeth": 9})])
            assert type(log[0]) is widget
