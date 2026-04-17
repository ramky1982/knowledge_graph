from __future__ import annotations

import importlib
from typing import Any


def require_networkx() -> Any:
    """Load networkx lazily to keep import errors user-friendly."""
    return importlib.import_module("networkx")
