"""Ingest file-based vector artifacts into a Chroma persistent collection."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .collection_builder import add_records_to_collection, get_or_create_collection, make_persistent_client
from .config import CHROMA_DB_DIRNAME, COLLECTION_NAME, EMBEDDING_MODEL, GPT_MODEL, coerce_chroma_metadata


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(payload)
    return records


def _run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("run_%Y%m%d_%H%M%S")


def _make_run_dir(output_dir: Path) -> tuple[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    run_dir = output_dir / run_id
    while run_dir.exists():
        time.sleep(1)
        run_id = _run_id()
        run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _portable_path(path: Path | str, project_root: Path) -> str:
    resolved_root = project_root.resolve()
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(path)


def _portable_payload(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _portable_payload(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_payload(item, project_root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            return _portable_path(path, project_root)
    return value


def _resolve_artifact_path(project_root: Path, artifact_root: Path, raw_path: Any, run_id: str | None = None) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        if path.parts and path.parts[0] == project_root.name:
            candidate = (project_root.parent / path).resolve()
        else:
            candidate = (project_root / path).resolve()
        if candidate.exists():
            return candidate
        return candidate

    if path.exists():
        return path

    parts = path.parts
    artifact_name = artifact_root.name
    if artifact_name in parts:
        tail = Path(*parts[parts.index(artifact_name) + 1 :])
        candidate = artifact_root / tail
        if candidate.exists():
            return candidate

    fallback_run_id = run_id or next((part for part in reversed(parts) if part.startswith("run_")), None)
    if fallback_run_id:
        if fallback_run_id in parts:
            run_index = parts.index(fallback_run_id)
            tail = Path(*parts[run_index + 1 :])
        else:
            tail = Path(path.name) if path.name else Path()
        candidate = artifact_root / fallback_run_id / tail
        if candidate.exists():
            return candidate

    return path


def resolve_vector_run(vector_indexes_dir: Path, vector_run: str) -> dict[str, Path | str]:
    vector_indexes_dir = vector_indexes_dir.resolve()
    project_root = vector_indexes_dir.parent
    if vector_run == "latest":
        latest = _read_json(vector_indexes_dir / "latest.json")
        run_id = str(latest["run_id"])
        run_dir = _resolve_artifact_path(project_root, vector_indexes_dir, latest["run_dir"], run_id)
        return {
            "vector_run_id": run_id,
            "run_dir": run_dir,
            "manifest_path": _resolve_artifact_path(project_root, vector_indexes_dir, latest["manifest_path"], run_id),
            "metadata_path": _resolve_artifact_path(project_root, vector_indexes_dir, latest["metadata_path"], run_id),
            "vectors_path": _resolve_artifact_path(project_root, vector_indexes_dir, latest["vectors_path"], run_id),
            "id_map_path": _resolve_artifact_path(project_root, vector_indexes_dir, latest["id_map_path"], run_id),
        }

    run_dir = Path(vector_run)
    if not run_dir.is_absolute():
        run_dir = vector_indexes_dir / vector_run
    elif not run_dir.exists():
        run_dir = vector_indexes_dir / run_dir.name
    return {
        "vector_run_id": run_dir.name,
        "run_dir": run_dir,
        "manifest_path": run_dir / "run_manifest.json",
        "metadata_path": run_dir / "chunk_metadata_records.jsonl",
        "vectors_path": run_dir / "vectors.npz",
        "id_map_path": run_dir / "id_map.json",
    }


def _build_ingest_payload(
    metadata_records: list[dict[str, Any]],
    vector_ids: list[str],
    vectors: np.ndarray,
    id_map: dict[str, Any],
) -> tuple[list[str], list[str], list[dict[str, str | int | float | bool]], list[list[float]], list[dict[str, Any]], list[str]]:
    records_by_id = {str(record.get("vector_record_id")): record for record in metadata_records}
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    embeddings: list[list[float]] = []
    id_manifest: list[dict[str, Any]] = []
    failed_ids: list[str] = []

    for index, vector_id in enumerate(vector_ids):
        mapped = id_map.get(str(index))
        if isinstance(mapped, dict) and str(mapped.get("vector_record_id")) != vector_id:
            raise RuntimeError(f"id_map mismatch at vector index {index}: {mapped.get('vector_record_id')} != {vector_id}")
        record = records_by_id.get(vector_id)
        if record is None:
            failed_ids.append(vector_id)
            continue
        retrieval_text = str(record.get("retrieval_text") or "")
        if not retrieval_text:
            failed_ids.append(vector_id)
            continue

        ids.append(vector_id)
        documents.append(retrieval_text)
        metadatas.append(coerce_chroma_metadata(record))
        embeddings.append(vectors[index].astype(float).tolist())
        id_manifest.append(
            {
                "chroma_id": vector_id,
                "vector_record_id": vector_id,
                "chunk_id": record.get("chunk_id", ""),
                "doc_id": record.get("doc_id", ""),
                "strategy_name": record.get("strategy_name", ""),
                "filename": record.get("filename", ""),
                "page_num": record.get("page_num"),
                "evidence_preview": record.get("evidence_preview", ""),
            }
        )

    return ids, documents, metadatas, embeddings, id_manifest, failed_ids


def ingest_vector_run(
    vector_indexes_dir: Path,
    output_dir: Path,
    vector_run: str = "latest",
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 256,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    project_root = output_dir.parent
    vector_paths = resolve_vector_run(vector_indexes_dir, vector_run)
    vector_manifest = _read_json(Path(vector_paths["manifest_path"]))
    id_map = _read_json(Path(vector_paths["id_map_path"]))
    metadata_records = _read_jsonl(Path(vector_paths["metadata_path"]))

    vector_data = np.load(Path(vector_paths["vectors_path"]), allow_pickle=True)
    vectors = vector_data["vectors"]
    vector_ids = [str(item) for item in vector_data["vector_record_ids"].tolist()]
    if len(vector_ids) != len(vectors):
        raise RuntimeError(f"Vector id count mismatch: {len(vector_ids)} ids for {len(vectors)} vectors.")

    ids, documents, metadatas, embeddings, id_manifest, failed_ids = _build_ingest_payload(
        metadata_records=metadata_records,
        vector_ids=vector_ids,
        vectors=vectors,
        id_map=id_map,
    )

    run_id, run_dir = _make_run_dir(output_dir)
    chroma_path = run_dir / CHROMA_DB_DIRNAME
    client = make_persistent_client(chroma_path)
    collection = get_or_create_collection(client, collection_name)

    inserted = add_records_to_collection(
        collection=collection,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        batch_size=batch_size,
    )

    collection_count = collection.count()
    timestamp = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "gpt_model": GPT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": collection_name,
        "persist_path": _portable_path(chroma_path, project_root),
        "source_vector_run_id": vector_paths["vector_run_id"],
        "source_vector_run_dir": _portable_path(Path(vector_paths["run_dir"]), project_root),
        "source_vector_manifest": _portable_payload(vector_manifest, project_root),
        "source_id_map_count": len(id_map),
        "input_vectors": len(vector_ids),
        "attempted_records": len(ids),
        "inserted_records": inserted,
        "failed_records": len(failed_ids),
        "collection_count": collection_count,
    }
    ingest_summary = {
        **manifest,
        "success_count": inserted,
        "failure_count": len(failed_ids),
        "failed_vector_record_ids": failed_ids,
    }

    _json_dump(run_dir / "run_manifest.json", manifest)
    _json_dump(run_dir / "ingest_summary.json", ingest_summary)
    _json_dump(run_dir / "id_manifest.json", id_manifest)
    _json_dump(
        output_dir / "latest.json",
        {
            "run_id": run_id,
            "run_dir": _portable_path(run_dir, project_root),
            "collection_name": collection_name,
            "persist_path": _portable_path(chroma_path, project_root),
            "manifest_path": _portable_path(run_dir / "run_manifest.json", project_root),
            "ingest_summary_path": _portable_path(run_dir / "ingest_summary.json", project_root),
            "id_manifest_path": _portable_path(run_dir / "id_manifest.json", project_root),
            "source_vector_run_id": vector_paths["vector_run_id"],
        },
    )

    return ingest_summary
