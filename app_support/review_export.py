"""
Raw final review JSON — debugging / human review baseline (not LLM chunk export).

Saved separately from *_llm_ready* artifacts. Content reflects the same final
`parse_document()` output the UI uses: finalized blocks, summary_blocks, parser_debug.

Output path: {RESULT_DIR}/review/{doc_id}.json

Top-level keys:
  export_kind, export_schema_version — discriminator / version
  doc_id, source_file, filename, filepath
  parser_version, document_type, exported_at, status
  metadata — copy of parse_document metadata (routing, quality, etc.)
  pages[] — per-page review records (see build_review_payload)

Each page includes page_number, dimensions, preview_image path, blocks[], summary_blocks[],
parser_debug, dropped_blocks (duplicate of parser_debug.dropped_blocks when present), plus
rag_text / text / ocr_applied / etc. when present on the source page.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REVIEW_SUBDIR = "review"
EXPORT_KIND = "raw_final_review"
EXPORT_SCHEMA_VERSION = 1


def build_review_payload(
    parsed: dict[str, Any],
    *,
    source_file: str | None = None,
    exported_at: str | None = None,
) -> dict[str, Any]:
    meta = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
    doc_id = parsed.get("id", "") or ""
    src = source_file or parsed.get("filepath") or parsed.get("filename")
    doc_type = meta.get("refined_document_type") or meta.get("document_type") or meta.get("doc_type")
    ts = exported_at or datetime.now(timezone.utc).isoformat()

    pages_out: list[dict[str, Any]] = []
    for p in parsed.get("pages") or []:
        if not isinstance(p, dict):
            continue
        dbg = p.get("parser_debug")
        if not isinstance(dbg, dict):
            dbg = {}
        dropped = dbg.get("dropped_blocks")
        page_review: dict[str, Any] = {
            "page_number": p.get("page_num"),
            "page_width": p.get("page_width"),
            "page_height": p.get("page_height"),
            "preview_image": p.get("preview_image"),
            "preview_width": p.get("preview_width"),
            "preview_height": p.get("preview_height"),
            "blocks": copy.deepcopy(p.get("blocks", [])),
            "summary_blocks": copy.deepcopy(p.get("summary_blocks", [])),
            "parser_debug": copy.deepcopy(dbg),
            "dropped_blocks": copy.deepcopy(dropped) if dropped is not None else [],
        }
        for extra in (
            "rag_text",
            "text",
            "tables",
            "ocr_applied",
            "coord_space",
            "image_count",
            "error",
            "preview_scale_x",
            "preview_scale_y",
        ):
            if extra in p:
                page_review[extra] = copy.deepcopy(p[extra])
        pages_out.append(page_review)

    return {
        "export_kind": EXPORT_KIND,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "doc_id": doc_id,
        "source_file": src,
        "filename": parsed.get("filename"),
        "filepath": parsed.get("filepath"),
        "parser_version": parsed.get("parser_version"),
        "document_type": doc_type,
        "exported_at": ts,
        "status": parsed.get("status"),
        "metadata": copy.deepcopy(meta),
        "pages": pages_out,
    }


def save_review_json(
    parsed: dict[str, Any],
    doc_id: str,
    result_dir: str,
    *,
    source_file: str | None = None,
) -> Path:
    payload = build_review_payload(parsed, source_file=source_file)
    review_dir = Path(result_dir) / REVIEW_SUBDIR
    review_dir.mkdir(parents=True, exist_ok=True)
    out = review_dir / f"{doc_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def persist_main_cache_json(
    doc_id: str,
    data: dict[str, Any],
    result_dir: str,
    parser_version: str,
) -> Path:
    """Write the same document shape app.py has always used: parsed_results/{doc_id}.json"""
    out_data = dict(data)
    out_data["parser_version"] = parser_version
    rdir = Path(result_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    out = rdir / f"{doc_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    return out


def save_all_parse_outputs(
    doc_id: str,
    parsed: dict[str, Any],
    result_dir: str,
    parser_version: str,
) -> tuple[Path, Path]:
    """
    Persist UI cache JSON and side-by-side review JSON.
    Does not change the in-memory `parsed` dict structure returned to callers.
    """
    main = persist_main_cache_json(doc_id, parsed, result_dir, parser_version)
    review = save_review_json(
        parsed,
        doc_id,
        result_dir,
        source_file=parsed.get("filepath"),
    )
    return main, review
