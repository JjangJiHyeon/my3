"""Create file-based vector preparation artifacts for evaluation."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .embedding_runner import embed_texts
from .metadata_builder import build_metadata_records
from .metadata_schema import EMBEDDING_MODEL, GPT_MODEL


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("run_%Y%m%d_%H%M%S")


def _portable_path(path: Path | str, project_root: Path) -> str:
    resolved_root = project_root.resolve()
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(path)


def _portable_path_list(paths: list[str], project_root: Path) -> list[str]:
    return [_portable_path(path, project_root) for path in paths]


def prepare_vectors(
    chunks_dir: Path,
    output_dir: Path,
    project_root: Path,
    embedding_batch_size: int = 64,
) -> dict[str, Any]:
    records, build_stats = build_metadata_records(chunks_dir)
    embeddable_records = [record for record in records if record["retrieval_text"]]
    embedding_texts = [record["retrieval_text"] for record in embeddable_records]

    vectors = embed_texts(
        embedding_texts,
        project_root=project_root,
        model=EMBEDDING_MODEL,
        batch_size=embedding_batch_size,
    )
    if len(vectors) != len(embeddable_records):
        raise RuntimeError(f"Embedding count mismatch: got {len(vectors)}, expected {len(embeddable_records)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    run_dir = output_dir / run_id
    while run_dir.exists():
        time.sleep(1)
        run_id = _run_id()
        run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    id_map = {
        str(index): {
            "vector_record_id": record["vector_record_id"],
            "chunk_id": record["chunk_id"],
            "doc_id": record["doc_id"],
            "strategy_name": record["strategy_name"],
        }
        for index, record in enumerate(embeddable_records)
    }

    embedding_dimensions = int(vectors.shape[1]) if vectors.ndim == 2 else 0
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpt_model": GPT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": embedding_dimensions,
        "input_chunk_files": _portable_path_list(build_stats["input_chunk_files"], project_root),
        "total_input_chunks": build_stats["total_input_chunks"],
        "embedded_chunks": len(embeddable_records),
        "skipped_empty_chunks": build_stats["skipped_empty_chunks"],
        "strategy_distribution": build_stats["strategy_distribution"],
        "doc_distribution": build_stats["doc_distribution"],
    }

    embedding_stats = {
        **manifest,
        "embedding_batch_size": embedding_batch_size,
        "skipped_empty_vector_record_ids": build_stats["skipped_empty_vector_record_ids"],
    }

    _json_dump(run_dir / "run_manifest.json", manifest)
    _write_jsonl(run_dir / "chunk_metadata_records.jsonl", records)
    _json_dump(run_dir / "embedding_stats.json", embedding_stats)
    np.savez_compressed(
        run_dir / "vectors.npz",
        vectors=vectors,
        vector_record_ids=np.asarray([record["vector_record_id"] for record in embeddable_records], dtype=object),
    )
    _json_dump(run_dir / "id_map.json", id_map)
    _json_dump(
        output_dir / "latest.json",
        {
            "run_id": run_id,
            "run_dir": _portable_path(run_dir, project_root),
            "manifest_path": _portable_path(run_dir / "run_manifest.json", project_root),
            "metadata_path": _portable_path(run_dir / "chunk_metadata_records.jsonl", project_root),
            "vectors_path": _portable_path(run_dir / "vectors.npz", project_root),
            "id_map_path": _portable_path(run_dir / "id_map.json", project_root),
        },
    )

    return manifest
