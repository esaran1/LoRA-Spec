from __future__ import annotations

import sys
from pathlib import Path


def add_source_checkout_to_path() -> None:
    """Make script modules importable from a source checkout during dynamic loading."""
    source_path = Path(__file__).resolve().parent / "src"
    if not source_path.exists():
        return
    source_entry = str(source_path)
    if source_entry not in sys.path:
        sys.path.insert(0, source_entry)


add_source_checkout_to_path()
