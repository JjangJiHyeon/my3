"""Metadata schema and normalization helpers for retrieval records."""

from __future__ import annotations

import hashlib
import re
from typing import Any

GPT_MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-large"
EXPECTED_EMBEDDING_DIMENSIONS = 3072
EVIDENCE_PREVIEW_MAX_CHARS = 600

METADATA_FIELDS = [
    "vector_record_id",
    "chunk_id",
    "strategy_name",
    "doc_id",
    "filename",
    "filepath",
    "page_num",
    "chunk_index",
    "chunk_type",
    "document_type",
    "pipeline_used",
    "quality_grade",
    "source_block_ids",
    "section_title",
    "page_title",
    "associated_title",
    "summary_priority",
    "chosen_text_source",
    "has_text",
    "has_table",
    "has_chart",
    "has_image",
    "table_count",
    "image_count",
    "retrieval_text",
    "display_text",
    "evidence_preview",
    "char_len",
    "token_estimate",
]

STRING_FIELDS = {
    "chunk_id",
    "strategy_name",
    "doc_id",
    "filename",
    "filepath",
    "chunk_type",
    "document_type",
    "pipeline_used",
    "quality_grade",
    "section_title",
    "page_title",
    "associated_title",
    "summary_priority",
    "chosen_text_source",
    "retrieval_text",
    "display_text",
    "evidence_preview",
}

BOOL_FIELDS = {"has_text", "has_table", "has_chart", "has_image"}
INT_FIELDS = {"page_num", "chunk_index", "table_count", "image_count", "char_len", "token_estimate"}


def normalize_text(value: Any) -> str:
    """Return a compact string without assuming any document-specific content."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", value).strip()


def estimate_tokens(text: str) -> int:
    """Lightweight token estimate used for metadata only."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def make_evidence_preview(text: str, max_chars: int = EVIDENCE_PREVIEW_MAX_CHARS) -> str:
    compact = normalize_text(text)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def stable_vector_record_id(chunk: dict[str, Any], fallback_index: int) -> str:
    base = "|".join(
        normalize_text(chunk.get(key))
        for key in ("doc_id", "strategy_name", "chunk_id", "page_num", "chunk_index", "chunk_type")
    )
    if not base.strip("|"):
        base = f"fallback|{fallback_index}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]
    return f"vec_{digest}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_source_block_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    text = normalize_text(value)
    return [text] if text else []


def normalize_metadata_record(
    chunk: dict[str, Any],
    document_defaults: dict[str, Any] | None = None,
    fallback_index: int = 0,
) -> dict[str, Any]:
    """Normalize one chunk into the retrieval metadata contract."""
    defaults = document_defaults or {}
    merged = {**defaults, **chunk}

    retrieval_text = normalize_text(merged.get("retrieval_text"))
    display_text = normalize_text(merged.get("display_text")) or retrieval_text
    evidence_preview = make_evidence_preview(merged.get("evidence_preview") or display_text or retrieval_text)

    record: dict[str, Any] = {field: None for field in METADATA_FIELDS}
    record["vector_record_id"] = normalize_text(merged.get("vector_record_id")) or stable_vector_record_id(merged, fallback_index)
    record["source_block_ids"] = _as_source_block_ids(merged.get("source_block_ids"))

    for field in STRING_FIELDS:
        record[field] = normalize_text(merged.get(field))

    for field in BOOL_FIELDS:
        record[field] = _as_bool(merged.get(field))

    for field in INT_FIELDS:
        record[field] = _as_int(merged.get(field))

    record["retrieval_text"] = retrieval_text
    record["display_text"] = display_text
    record["evidence_preview"] = evidence_preview
    record["char_len"] = record["char_len"] if record["char_len"] is not None else len(retrieval_text)
    record["token_estimate"] = (
        record["token_estimate"] if record["token_estimate"] is not None else estimate_tokens(retrieval_text)
    )
    record["table_count"] = record["table_count"] if record["table_count"] is not None else 0
    record["image_count"] = record["image_count"] if record["image_count"] is not None else 0
    record["has_text"] = bool(record["has_text"] or retrieval_text)
    record["has_table"] = bool(record["has_table"] or record["table_count"] > 0)
    record["has_image"] = bool(record["has_image"] or record["image_count"] > 0)

    return {field: record[field] for field in METADATA_FIELDS}
