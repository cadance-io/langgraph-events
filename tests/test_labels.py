"""Two classes named in one diagnostic must be told apart."""

from langgraph_events._labels import distinct_labels


class Alpha:
    pass


class Holder:
    class Alpha:
        pass


def describe_distinct_labels():

    def when_the_names_differ():

        def it_uses_the_bare_names():
            here, there = distinct_labels(Alpha, Holder)

            assert (here, there) == ("Alpha", "Holder")

    def when_only_the_qualnames_differ():

        def it_escalates_to_qualnames_and_no_further():
            here, there = distinct_labels(Alpha, Holder.Alpha)

            # Qualnames already separate them, so no identity noise.
            assert here == f"{Alpha.__module__}.{Alpha.__qualname__}"
            assert there == f"{Holder.__module__}.{Holder.Alpha.__qualname__}"

    def when_even_the_qualnames_match():

        # Two engine lifetimes of one module: same module, same qualname,
        # different objects. Nothing textual separates them.
        def it_escalates_to_identity():
            twin = type("Alpha", (), {"__qualname__": Alpha.__qualname__})
            twin.__module__ = Alpha.__module__

            here, there = distinct_labels(Alpha, twin)

            assert here != there

    def when_the_same_class_is_passed_twice():

        def it_does_not_invent_a_difference():
            here, there = distinct_labels(Alpha, Alpha)

            # Equal, and stated plainly — not "mod.Alpha (0x..) and
            # mod.Alpha (0x..)", which reads as two things that are one.
            assert here == there == "Alpha"
