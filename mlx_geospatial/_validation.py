"""Small shared validators for public API resource limits."""

from __future__ import annotations

import operator

import numpy as np


def check_positive_int(value, name: str) -> int:
    """Return a positive integer parameter; reject bools and coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer, got bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(
            f"{name} must be a positive integer, got {type(value).__name__}"
        ) from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def check_chunk_bytes(value) -> int:
    """Return a positive integer byte budget; reject bools and coercion."""
    return check_positive_int(value, "chunk_bytes")
