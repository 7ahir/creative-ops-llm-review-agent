from __future__ import annotations

import hashlib


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[: max(0, limit - 1)].rstrip()
    return clipped + "…"


def slug(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

