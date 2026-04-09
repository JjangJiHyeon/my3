"""Shared Chroma ingestion contract and constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

GPT_MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-large"
COLLECTION_NAME = "document_chunks"
CHROMA_DB_DIRNAME = "chroma"

CHROMA_METADATA_FIELDS = [
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
    "char_len",
    "token_estimate",
    "evidence_preview",
    "visual_type",
    "visual_index",
    "visual_confidence",
    "source_bbox",
    "extraction_method",
    "evidence_image_path",
]


def coerce_chroma_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep Chroma metadata filterable and scalar-only."""
    metadata: dict[str, str | int | float | bool] = {}
    for field in CHROMA_METADATA_FIELDS:
        value = record.get(field)
        if value is None:
            metadata[field] = ""
        elif isinstance(value, (str, int, float, bool)):
            metadata[field] = value
        else:
            metadata[field] = str(value)
    return metadata


def resolve_project_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == project_root.name:
        return (project_root.parent / path).resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (project_root / path).resolve()
