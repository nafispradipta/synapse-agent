"""
Shared storage constants and the optional NumPy dependency.

Kept separate from the storage facade so the bounded-context mixins can import
them without creating an import cycle back into ``synapse.storage``.
"""

from __future__ import annotations

import math
from typing import Any, Set

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


def bounded_unit(value: Any, default: float = 0.5) -> float:
    """Clamp a unit-interval score; reject non-finite or non-numeric input."""
    if value is None:
        return float(default)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return max(0.0, min(1.0, number))


def bounded_strength(value: Any, default: float = 1.0) -> float:
    """Clamp a Hebbian strength score into ``[0.0, 5.0]``."""
    if value is None:
        return float(default)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return max(0.0, min(5.0, number))
