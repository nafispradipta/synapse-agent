"""
Shared storage constants and the optional NumPy dependency.

Kept separate from the storage facade so the bounded-context mixins can import
them without creating an import cycle back into ``synapse.storage``.
"""

from __future__ import annotations

from typing import Set

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

VALID_CATEGORIES: Set[str] = {
    "preference",
    "decision",
    "architecture",
    "incident",
    "sop",
    "general",
}

VALID_STATUSES: Set[str] = {
    "active",
    "superseded",
    "archived",
    "flagged",
}
