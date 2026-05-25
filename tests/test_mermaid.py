"""Unit tests for the internal mermaid flowchart builder.

Scope: the smallest unit that exercises the actual ``_render_edge`` →
``_quote`` path. Snapshot/integration coverage lives in
``test_mermaid_sync.py`` and ``test_patterns_docs_sync.py``; this file
pins the pipe-label quoting rules independently of any caller's label
format, so causation-suffix renames or new emitters can't silently re-
introduce the v11 parse error.
"""

from __future__ import annotations

from langgraph_events._mermaid import MermaidFlowchart


def describe_mermaid_flowchart():
    def describe_pipe_edge_label_quoting():
        def when_label_has_no_special_chars():
            def it_renders_the_label_bare():
                out = MermaidFlowchart().edge("A", "B", label="plain").render()
                assert "|plain|" in out

        def when_label_contains_brackets_or_braces():
            # Regression for #85: ``|x [y]|`` is parsed by mermaid@11 as a
            # square node-shape literal; ``|x {y}|`` would be a diamond.
            # Both must be wrapped so the pipe-label parser sees text.
            def it_double_quotes_the_label():
                for ch_open, ch_close in ("[]", "{}"):
                    label = f"x {ch_open}y{ch_close}"
                    out = MermaidFlowchart().edge("A", "B", label=label).render()
                    assert f'|"{label}"|' in out, label

        def when_label_contains_other_trip_chars():
            def it_double_quotes_the_label():
                for ch in '()|"#':
                    label = f"x{ch}y"
                    out = MermaidFlowchart().edge("A", "B", label=label).render()
                    assert f'|"{label}"|' in out, ch
