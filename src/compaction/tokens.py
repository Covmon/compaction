"""Deterministic token estimation (no API call)."""

from __future__ import annotations

import json
import math
from typing import Any

CHARS_PER_TOKEN = 3.5


def estimate_tokens(obj: Any) -> int:
    """Estimate tokens of a str or any JSON-able object as ceil(len/3.5)."""
    if obj is None:
        return 0
    text = (
        obj
        if isinstance(obj, str)
        else json.dumps(obj, ensure_ascii=False, default=str)
    )
    return math.ceil(len(text) / CHARS_PER_TOKEN)
