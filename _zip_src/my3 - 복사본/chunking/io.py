from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_parsed_results(parsed_dir: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    if not parsed_dir.exists():
        return docs
    for path in sorted(parsed_dir.glob("*.json")):
        try:
            data = read_json(path)
        except Exception as exc:
            docs[path.stem] = {"id": path.stem, "status": "error", "error": str(exc), "pages": []}
            continue
        doc_id = str(data.get("id") or path.stem)
        docs[doc_id] = data
    return docs


def load_review_results(review_dir: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    if not review_dir.exists():
        return docs
    for path in sorted(review_dir.glob("*.json")):
        try:
            docs[path.stem] = read_json(path)
        except Exception as exc:
            docs[path.stem] = {"doc_id": path.stem, "status": "error", "error": str(exc), "pages": []}
    return docs


def load_llm_ready(root: Path) -> dict[str, dict[str, Any]]:
    by_filename: dict[str, dict[str, Any]] = {}
    search_dirs = [root / "exports" / "latest_sidecars", root]
    seen_paths: set[Path] = set()
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.glob("*_llm_ready.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                data = read_json(path)
            except Exception:
                continue
            filename = data.get("filename")
            if filename:
                by_filename[str(filename)] = data
    return by_filename


def chunk_output_payload(
    *,
    doc: dict[str, Any],
    doc_id: str,
    strategy_name: str,
    chunks: list[dict[str, Any]],
    issues: list[str] | None = None,
) -> dict[str, Any]:
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return {
        "schema_version": 1,
        "strategy_name": strategy_name,
        "doc_id": doc_id,
        "filename": doc.get("filename"),
        "filepath": doc.get("filepath"),
        "document_type": meta.get("refined_document_type") or meta.get("document_type") or doc.get("document_type"),
        "pipeline_used": meta.get("pipeline_used") or doc.get("pipeline_used"),
        "quality_grade": meta.get("quality_grade") or doc.get("quality_grade"),
        "total_chunks": len(chunks),
        "issues": issues or [],
        "chunks": chunks,
    }
