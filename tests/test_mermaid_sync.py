"""Verify .graph.md files stay in sync with example handlers."""

import subprocess
import sys


def describe_mermaid_sync():
    def it_keeps_graph_files_current():
        result = subprocess.run(  # noqa: S603
            [sys.executable, "scripts/generate_mermaid.py", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"Mermaid graph files are stale. "
            f"Run 'uv run python scripts/generate_mermaid.py' to update.\n"
            f"{result.stdout}{result.stderr}"
        )
