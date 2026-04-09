"""Format retrieved document metadata into the fixed source response shape."""

from __future__ import annotations

from typing import Any

from .schemas import Source


def format_sources(documents: list[Any]) -> list[Source]:
    seen: set[tuple[str, int]] = set()
    sources: list[Source] = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        filename = str(metadata.get("filename") or metadata.get("document") or "").strip()
        if not filename:
            continue
        page = _safe_page(metadata.get("page_num", metadata.get("page", 0)))
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(document=filename, page=page))
    return sources


def _safe_page(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

