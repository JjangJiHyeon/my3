from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Iterable


TEXT_TYPES = {"title", "text"}
VISUAL_TYPES = {"chart", "image"}
TABLE_TYPES = {"table"}
EXCLUDED_ROLES = {"footer", "page_number", "decorative_noise"}
EXCLUDED_KINDS = {"decorative_visual", "background", "logo"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_id(*parts: Any, length: int = 12) -> str:
    raw = "::".join("" if part is None else str(part) for part in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:length]


def page_num(page: dict[str, Any]) -> int | None:
    value = page.get("page_num", page.get("page_number"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def block_id(block: dict[str, Any]) -> str:
    return str(block.get("id") or stable_id(block.get("type"), block.get("bbox"), block.get("text")))


def block_meta(block: dict[str, Any]) -> dict[str, Any]:
    meta = block.get("meta")
    return meta if isinstance(meta, dict) else {}


def is_excluded_block(block: dict[str, Any], *, for_support: bool = False) -> bool:
    meta = block_meta(block)
    role = str(meta.get("slide_role") or meta.get("summary_role") or meta.get("dashboard_role") or "").lower()
    kind = str(meta.get("slide_panel_kind") or meta.get("slide_visual_role") or "").lower()
    reason = str(meta.get("summary_exclude_reason") or meta.get("rag_exclude_reason") or "").lower()
    if meta.get("summary_exclude") or meta.get("rag_exclude"):
        if for_support and any(key in reason for key in ("empty_table", "non_rectangular")):
            return True
        return True
    if block.get("type") == "footer" or role in EXCLUDED_ROLES:
        return True
    if kind in EXCLUDED_KINDS or "decorative" in kind:
        return True
    if any(key in reason for key in ("decorative", "footer", "page_number", "empty_table_geometry")):
        return True
    return False


def summary_priority(blocks: Iterable[dict[str, Any]]) -> str | None:
    rank = {"high": 3, "medium": 2, "low": 1}
    best = None
    best_score = 0
    for block in blocks:
        value = block_meta(block).get("summary_priority")
        score = rank.get(str(value).lower(), 0)
        if score > best_score:
            best = str(value)
            best_score = score
    return best


def block_order(block: dict[str, Any]) -> tuple:
    meta = block_meta(block)
    bbox = block.get("bbox") or [0, 0, 0, 0]
    return (
        meta.get("rag_order", 9999),
        meta.get("slide_panel_order", 9999),
        meta.get("slide_reading_order", 999999),
        bbox[1] if len(bbox) > 1 else 0,
        bbox[0] if bbox else 0,
    )


def panel_key(block: dict[str, Any]) -> str | None:
    meta = block_meta(block)
    for key in ("slide_panel_id", "slide_region_id", "dashboard_region_id"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


def associated_title(blocks: Iterable[dict[str, Any]], page_title: str | None = None) -> str | None:
    for block in blocks:
        meta = block_meta(block)
        for key in ("associated_title", "page_title"):
            value = clean_text(meta.get(key))
            if value:
                return value
        if block.get("type") == "title":
            value = clean_text(block.get("text"))
            if value:
                return value
    return page_title


def split_long_text(text: str, *, target: int = 1100, hard: int = 1400, overlap: int = 120) -> list[str]:
    text = clean_text(text)
    if len(text) <= hard:
        return [text] if text else []
    parts: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + hard)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n\n", 0, target), window.rfind("\n", 0, target), window.rfind(". ", 0, target))
            if cut > target * 0.45:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return parts


def support_text_from_block(block: dict[str, Any]) -> list[str]:
    meta = block_meta(block)
    out: list[str] = []
    seen_values: set[str] = set()
    for key, label in (
        ("table_summary", "TABLE SUMMARY"),
        ("table_markdown", "TABLE"),
        ("chart_summary", "CHART SUMMARY"),
        ("visual_summary", "VISUAL SUMMARY"),
        ("caption_text", "CAPTION"),
    ):
        value = clean_text(meta.get(key) or block.get(key))
        compact = re.sub(r"\s+", "", value)
        if value and compact not in seen_values:
            seen_values.add(compact)
            out.append(f"[{label}]\n{value}")
    rows = meta.get("key_value_rows") or block.get("key_value_rows")
    if isinstance(rows, list) and rows:
        lines = []
        for row in rows[:20]:
            if isinstance(row, dict):
                item = clean_text(row.get("item"))
                values = clean_text(row.get("values"))
                if item or values:
                    lines.append(f"- {item}: {values}".strip())
            elif isinstance(row, (list, tuple)):
                lines.append("- " + " | ".join(clean_text(x) for x in row if clean_text(x)))
        if lines:
            out.append("[KEY VALUES]\n" + "\n".join(lines))
    return out


def compact_join(parts: Iterable[str]) -> str:
    cleaned = [clean_text(part) for part in parts if clean_text(part)]
    return "\n\n".join(cleaned)


def type_counts(blocks: Iterable[dict[str, Any]]) -> Counter:
    return Counter(block.get("type") for block in blocks)
