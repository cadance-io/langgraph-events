"""Verify docs/patterns.md + docs/index.md + docs/concepts.md autogen blocks
stay in sync with the example EventGraphs.

This is the counterpart of ``test_mermaid_sync.py`` — runs the same script
with ``--check``, which also validates the `<!-- autogen:* -->` blocks in
the docs tree.
"""

import subprocess
import sys


def describe_patterns_docs_sync():
    def it_keeps_docs_autogen_blocks_current():
        result = subprocess.run(  # noqa: S603
            [sys.executable, "scripts/generate_mermaid.py", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "docs/patterns.md / docs/index.md / docs/concepts.md autogen "
            "blocks are stale. Run 'uv run python scripts/generate_mermaid.py' "
            f"to update.\n{result.stdout}{result.stderr}"
        )
