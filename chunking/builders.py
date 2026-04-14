from __future__ import annotations

from collections import Counter
from copy import deepcopy
from statistics import mean
from typing import Any

from .io import chunk_output_payload
from .strategies import (
    BASELINE_STRATEGY,
    MAIN_STRATEGY,
    build_llm_ready_native_chunks,
    build_text_first_chunks,
)


def summarize_chunks(doc_id: str, filename: str, strategy_name: str, chunks: list[dict[str, Any]], issues: list[str]) -> dict[str, Any]:
    lengths = [chunk.get("char_len", 0) for chunk in chunks]
    pages = sorted({chunk.get("page_num") for chunk in chunks if chunk.get("page_num") is not None})
    sparse = sum(1 for chunk in chunks if chunk.get("metadata", {}).get("sparse_page"))
    return {
        "doc_id": doc_id,
        "filename": filename,
        "strategy_name": strategy_name,
        "total_chunks": len(chunks),
        "chunk_type_distribution": dict(Counter(chunk.get("chunk_type") for chunk in chunks)),
        "avg_char_len": round(mean(lengths), 2) if lengths else 0,
        "min_char_len": min(lengths) if lengths else 0,
        "max_char_len": max(lengths) if lengths else 0,
        "page_coverage": pages,
        "page_coverage_count": len(pages),
        "sparse_page_chunk_count": sparse,
        "sparse_page_ratio": round(sparse / max(1, len(chunks)), 4),
        "empty_retrieval_text_chunks": sum(1 for chunk in chunks if not str(chunk.get("retrieval_text") or "").strip()),
        "top_issues": issues[:50],
    }


def build_all_for_document(
    *,
    parsed_doc: dict[str, Any],
    llm_ready: dict[str, Any] | None,
    doc_id: str,
) -> dict[str, dict[str, Any]]:
    prepared_doc = _prepare_parsed_doc_for_chunking(parsed_doc)
    main_chunks, main_issues = build_text_first_chunks(prepared_doc, llm_ready, doc_id)
    baseline_chunks, baseline_issues = build_llm_ready_native_chunks(prepared_doc, llm_ready, doc_id)
    filename = str(prepared_doc.get("filename") or (llm_ready or {}).get("filename") or "")
    return {
        MAIN_STRATEGY: chunk_output_payload(
            doc=prepared_doc,
            doc_id=doc_id,
            strategy_name=MAIN_STRATEGY,
            chunks=main_chunks,
            issues=main_issues,
        ),
        BASELINE_STRATEGY: chunk_output_payload(
            doc=prepared_doc,
            doc_id=doc_id,
            strategy_name=BASELINE_STRATEGY,
            chunks=baseline_chunks,
            issues=baseline_issues,
        ),
        "summary": {
            "doc_id": doc_id,
            "filename": filename,
            "strategies": [
                summarize_chunks(doc_id, filename, MAIN_STRATEGY, main_chunks, main_issues),
                summarize_chunks(doc_id, filename, BASELINE_STRATEGY, baseline_chunks, baseline_issues),
            ],
        },
    }


def _prepare_parsed_doc_for_chunking(parsed_doc: dict[str, Any]) -> dict[str, Any]:
    prepared = deepcopy(parsed_doc)
    pages = prepared.get("pages")
    if not isinstance(pages, list):
        return prepared

    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            continue
        from parsers.pdf_parser import _enrich_table_blocks

        blocks, _ = _enrich_table_blocks(
            blocks,
            float(page.get("page_width") or 0.0),
            float(page.get("page_height") or 0.0),
            [],
        )
        page["blocks"] = blocks
    return prepared
