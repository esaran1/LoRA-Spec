from __future__ import annotations

import os
import tempfile
from pathlib import Path


matplotlib_cache = Path(tempfile.gettempdir()) / "lora-spec-matplotlib-cache"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
