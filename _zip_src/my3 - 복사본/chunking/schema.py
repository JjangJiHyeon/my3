from __future__ import annotations

from typing import Any

CHUNK_SCHEMA_VERSION = 1


def empty_chunk(
    *,
    chunk_id: str,
    strategy_name: str,
    doc_id: str,
    filename: str,
    filepath: str | None,
    document_type: str | None,
    pipeline_used: str | None,
    quality_grade: str | None,
    page_num: int | None,
    chunk_index: int,
    chunk_type: str,
    retrieval_text: str,
    display_text: str | None = None,
    section_title: str | None = None,
    page_title: str | None = None,
    associated_title: str | None = None,
    source_block_ids: list[str] | None = None,
    chosen_text_source: str | None = None,
    summary_priority: str | None = None,
    has_table: bool = False,
    has_chart: bool = False,
    has_image: bool = False,
    table_count: int = 0,
    image_count: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval_text = retrieval_text or ""
    display_text = display_text if display_text is not None else retrieval_text
    return {
        "chunk_id": chunk_id,
        "strategy_name": strategy_name,
        "doc_id": doc_id,
        "filename": filename,
        "filepath": filepath,
        "document_type": document_type,
        "pipeline_used": pipeline_used,
        "quality_grade": quality_grade,
        "page_num": page_num,
        "chunk_index": chunk_index,
        "chunk_type": chunk_type,
        "section_title": section_title,
        "page_title": page_title,
        "associated_title": associated_title,
        "source_block_ids": source_block_ids or [],
        "chosen_text_source": chosen_text_source,
        "summary_priority": summary_priority,
        "has_text": bool(retrieval_text.strip()),
        "has_table": bool(has_table),
        "has_chart": bool(has_chart),
        "has_image": bool(has_image),
        "table_count": int(table_count or 0),
        "image_count": int(image_count or 0),
        "retrieval_text": retrieval_text,
        "display_text": display_text or "",
        "evidence_preview": (display_text or retrieval_text or "")[:500],
        "char_len": len(retrieval_text),
        "token_estimate": max(1, len(retrieval_text) // 4) if retrieval_text.strip() else 0,
        "metadata": metadata or {},
    }
