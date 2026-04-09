"""Build normalized retrieval metadata from chunk JSON files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .metadata_schema import normalize_metadata_record

CHUNK_FILE_PATTERNS = (
    "*.text_first_with_visual_support.json",
    "*.llm_ready_native.json",
)


def discover_chunk_files(chunks_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in CHUNK_FILE_PATTERNS:
        files.extend(chunks_dir.glob(pattern))
    discovered = sorted(set(files))
    if not discovered:
        raise FileNotFoundError(f"No supported chunk JSON files found in: {chunks_dir}")
    return discovered


def _document_defaults(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    fallback_parts = source_path.name.split(".")
    fallback_strategy = fallback_parts[1] if len(fallback_parts) > 2 else ""
    return {
        "strategy_name": payload.get("strategy_name") or fallback_strategy,
        "doc_id": payload.get("doc_id") or source_path.name.split(".")[0],
        "filename": payload.get("filename"),
        "filepath": payload.get("filepath"),
        "document_type": payload.get("document_type"),
        "pipeline_used": payload.get("pipeline_used"),
        "quality_grade": payload.get("quality_grade"),
    }


def load_chunk_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Chunk file must contain a JSON object: {path}")
    if not isinstance(payload.get("chunks"), list):
        raise ValueError(f"Chunk file is missing a list-valued 'chunks' field: {path}")
    return payload


def build_metadata_records(chunks_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_files = discover_chunk_files(chunks_dir)
    records: list[dict[str, Any]] = []
    skipped_empty_ids: list[str] = []
    strategy_distribution: Counter[str] = Counter()
    doc_distribution: Counter[str] = Counter()

    for chunk_file in chunk_files:
        payload = load_chunk_payload(chunk_file)
        defaults = _document_defaults(payload, chunk_file)
        for chunk in payload["chunks"]:
            if not isinstance(chunk, dict):
                continue
            record = normalize_metadata_record(chunk, defaults, len(records))
            records.append(record)
            strategy_distribution[record["strategy_name"] or "unknown"] += 1
            doc_distribution[record["doc_id"] or "unknown"] += 1
            if not record["retrieval_text"]:
                skipped_empty_ids.append(record["vector_record_id"])

    stats = {
        "input_chunk_files": [str(path) for path in chunk_files],
        "total_input_chunks": len(records),
        "skipped_empty_chunks": len(skipped_empty_ids),
        "skipped_empty_vector_record_ids": skipped_empty_ids,
        "strategy_distribution": dict(sorted(strategy_distribution.items())),
        "doc_distribution": dict(sorted(doc_distribution.items())),
    }
    return records, stats
