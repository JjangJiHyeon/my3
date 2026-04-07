"""Cosine-similarity retrieval evaluation over existing vector indexes."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


GPT_MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-large"
STRATEGIES = ("text_first_with_visual_support", "llm_ready_native")
RESULT_FIELDS = (
    "query_id",
    "query_text",
    "strategy_name",
    "rank",
    "score",
    "vector_record_id",
    "chunk_id",
    "doc_id",
    "filename",
    "page_num",
    "chunk_type",
    "section_title",
    "page_title",
    "evidence_preview",
    "retrieval_text_preview",
)


def load_env_file(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def make_run_dir(project_root: Path) -> Path:
    base = project_root / "eval_runs"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def resolve_vector_run(project_root: Path, vector_run: str) -> Path:
    vector_root = project_root / "vector_indexes"
    if not vector_root.exists():
        raise FileNotFoundError(f"Vector index directory does not exist: {vector_root}")

    if vector_run == "latest":
        latest_path = vector_root / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"latest vector manifest does not exist: {latest_path}")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        candidates = [
            latest.get("run_dir"),
            latest.get("latest_run_dir"),
            latest.get("path"),
            latest.get("run_id"),
            latest.get("latest_run_id"),
        ]
        for candidate in candidates:
            if candidate:
                path = Path(str(candidate))
                if not path.is_absolute():
                    path = vector_root / path
                if path.exists():
                    return path
        run_dirs = sorted(vector_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_dirs:
            return run_dirs[0]
        raise FileNotFoundError(f"Could not resolve latest vector run from {latest_path}")

    path = Path(vector_run)
    if not path.is_absolute():
        path = vector_root / vector_run
    if not path.exists():
        raise FileNotFoundError(f"Vector run does not exist: {path}")
    return path


def load_vectors(vector_run_dir: Path) -> np.ndarray:
    npz_path = vector_run_dir / "vectors.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing vectors file: {npz_path}")

    with np.load(npz_path) as data:
        preferred = [key for key in ("embeddings", "vectors", "matrix") if key in data]
        keys = preferred or list(data.files)
        for key in keys:
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.ndim == 2:
                return arr
    raise ValueError(f"No 2D vector array found in {npz_path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {exc}") from exc
    return rows


def load_id_map(vector_run_dir: Path, row_count: int) -> list[str | None]:
    path = vector_run_dir / "id_map.json"
    if not path.exists():
        return [None] * row_count
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids: list[str | None] = [None] * row_count

    if isinstance(payload, list):
        for idx, value in enumerate(payload[:row_count]):
            ids[idx] = str(value)
        return ids

    if isinstance(payload, dict):
        for key, value in payload.items():
            try:
                idx = int(key)
                if 0 <= idx < row_count:
                    ids[idx] = str(value)
            except ValueError:
                continue
    return ids


def _record_id(record: dict[str, Any], fallback: str | None = None) -> str:
    for key in ("vector_record_id", "record_id", "id", "vector_id"):
        if record.get(key) is not None:
            return str(record[key])
    if fallback is not None:
        return fallback
    chunk_id = str(record.get("chunk_id", ""))
    return chunk_id or ""


def load_metadata(vector_run_dir: Path, row_count: int) -> list[dict[str, Any]]:
    metadata_path = vector_run_dir / "chunk_metadata_records.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    rows = load_jsonl(metadata_path)
    id_map = load_id_map(vector_run_dir, row_count)
    by_id = {_record_id(row): row for row in rows if _record_id(row)}
    metadata: list[dict[str, Any]] = []

    for idx in range(row_count):
        mapped_id = id_map[idx]
        if mapped_id and mapped_id in by_id:
            record = dict(by_id[mapped_id])
        elif idx < len(rows):
            record = dict(rows[idx])
        else:
            record = {}
        record["vector_record_id"] = _record_id(record, mapped_id or str(idx))
        record["_vector_index"] = idx
        metadata.append(record)
    return metadata


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_queries(queries: list[dict[str, Any]], project_root: Path) -> np.ndarray:
    load_env_file(project_root)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in .env or the environment.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required to embed evaluation queries.") from exc

    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query["query_text"] for query in queries],
    )
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype=np.float32)


def text_preview(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _strategy_name(record: dict[str, Any]) -> str:
    strategy = str(record.get("strategy_name", ""))
    if strategy:
        return strategy
    chunk_id = str(record.get("chunk_id", ""))
    for candidate in STRATEGIES:
        if candidate in chunk_id:
            return candidate
    return ""


def build_result_row(
    query: dict[str, Any],
    record: dict[str, Any],
    strategy: str,
    rank: int,
    score: float,
) -> dict[str, Any]:
    return {
        "query_id": query["query_id"],
        "query_text": query["query_text"],
        "strategy_name": strategy,
        "rank": rank,
        "score": round(float(score), 8),
        "vector_record_id": str(record.get("vector_record_id", "")),
        "chunk_id": str(record.get("chunk_id", "")),
        "doc_id": str(record.get("doc_id", "")),
        "filename": str(record.get("filename", "")),
        "page_num": record.get("page_num", ""),
        "chunk_type": str(record.get("chunk_type", "")),
        "section_title": str(record.get("section_title", "")),
        "page_title": str(record.get("page_title", "")),
        "evidence_preview": text_preview(record.get("evidence_preview") or record.get("display_text")),
        "retrieval_text_preview": text_preview(record.get("retrieval_text")),
    }


def retrieve_topk(
    queries: list[dict[str, Any]],
    query_vectors: np.ndarray,
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    normalized_vectors = normalize_matrix(vectors)
    normalized_queries = normalize_matrix(query_vectors)
    rows: list[dict[str, Any]] = []

    strategy_indices = {
        strategy: [idx for idx, record in enumerate(metadata) if _strategy_name(record) == strategy]
        for strategy in STRATEGIES
    }
    for strategy, indices in strategy_indices.items():
        if not indices:
            raise ValueError(f"No vector metadata records found for strategy: {strategy}")

    for query_idx, query in enumerate(queries):
        query_vector = normalized_queries[query_idx]
        for strategy, indices in strategy_indices.items():
            scores = normalized_vectors[indices] @ query_vector
            top_count = min(top_k, len(indices))
            ordered_local = np.argsort(scores)[::-1][:top_count]
            for rank, local_idx in enumerate(ordered_local, start=1):
                vector_idx = indices[int(local_idx)]
                rows.append(
                    build_result_row(
                        query=query,
                        record=metadata[vector_idx],
                        strategy=strategy,
                        rank=rank,
                        score=float(scores[int(local_idx)]),
                    )
                )
    return rows


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 8) if values else 0.0


def compute_metrics(rows: list[dict[str, Any]], queries: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    by_strategy_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy_query[(row["strategy_name"], row["query_id"])].append(row)

    strategy_summary: dict[str, Any] = {}
    for strategy in STRATEGIES:
        top1_scores: list[float] = []
        top3_scores: list[float] = []
        docs: set[str] = set()
        strategy_rows = [row for row in rows if row["strategy_name"] == strategy]

        for query in queries:
            hits = sorted(by_strategy_query.get((strategy, query["query_id"]), []), key=lambda x: int(x["rank"]))
            if hits:
                top1_scores.append(float(hits[0]["score"]))
                top3_scores.append(_avg([float(hit["score"]) for hit in hits[:3]]))
                docs.update(str(hit["doc_id"]) for hit in hits if hit.get("doc_id"))

        strategy_summary[strategy] = {
            "total_hits": len(strategy_rows),
            "avg_top1_score": _avg(top1_scores),
            "avg_top3_score": _avg(top3_scores),
            "doc_coverage": len(docs),
            "hit_distribution": dict(Counter(row["doc_id"] or "(missing)" for row in strategy_rows)),
            "chunk_type_hit_distribution": dict(Counter(row["chunk_type"] or "(missing)" for row in strategy_rows)),
        }

    doc_scores: dict[str, dict[str, Any]] = {}
    doc_ids = sorted({str(row["doc_id"]) for row in rows if row.get("doc_id")})
    for doc_id in doc_ids:
        doc_rows = [row for row in rows if row.get("doc_id") == doc_id]
        by_strategy: dict[str, Any] = {}
        for strategy in STRATEGIES:
            strategy_doc_rows = [row for row in doc_rows if row["strategy_name"] == strategy]
            by_strategy[strategy] = {
                "avg_score": _avg([float(row["score"]) for row in strategy_doc_rows]),
                "hit_count": len(strategy_doc_rows),
            }
        doc_scores[doc_id] = {
            "filename": next((row["filename"] for row in doc_rows if row.get("filename")), ""),
            "avg_score": _avg([float(row["score"]) for row in doc_rows]),
            "hit_count": len(doc_rows),
            "by_strategy": by_strategy,
        }

    query_best: list[dict[str, Any]] = []
    score_diffs: list[dict[str, Any]] = []
    for query in queries:
        top1: dict[str, float] = {}
        for strategy in STRATEGIES:
            hits = sorted(by_strategy_query.get((strategy, query["query_id"]), []), key=lambda x: int(x["rank"]))
            top1[strategy] = float(hits[0]["score"]) if hits else 0.0
        diff = top1["text_first_with_visual_support"] - top1["llm_ready_native"]
        best_strategy = "tie"
        if diff > 0:
            best_strategy = "text_first_with_visual_support"
        elif diff < 0:
            best_strategy = "llm_ready_native"
        query_best.append(
            {
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "best_strategy": best_strategy,
                "top1_scores": {key: round(value, 8) for key, value in top1.items()},
            }
        )
        score_diffs.append({"query_id": query["query_id"], "text_first_minus_llm_ready_top1": round(diff, 8)})

    return {
        "total_queries": len(queries),
        "top_k": top_k,
        "strategies": strategy_summary,
        "doc_coverage": {strategy: data["doc_coverage"] for strategy, data in strategy_summary.items()},
        "strategy_hit_distribution": {strategy: data["hit_distribution"] for strategy, data in strategy_summary.items()},
        "chunk_type_hit_distribution": {
            strategy: data["chunk_type_hit_distribution"] for strategy, data in strategy_summary.items()
        },
        "document_average_scores": doc_scores,
        "query_best_strategy": query_best,
        "strategy_score_differences": score_diffs,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps({field: row.get(field, "") for field in RESULT_FIELDS}, ensure_ascii=False) + "\n")


def write_topk_preview(path: Path, rows: list[dict[str, Any]], queries: list[dict[str, Any]]) -> None:
    lines = ["# Retrieval Top-K Preview", ""]
    for query in queries:
        lines.extend([f"## {query['query_id']}: {query['query_text']}", ""])
        for strategy in STRATEGIES:
            lines.extend([f"### {strategy}", ""])
            hits = [row for row in rows if row["query_id"] == query["query_id"] and row["strategy_name"] == strategy]
            hits = sorted(hits, key=lambda row: int(row["rank"]))
            for hit in hits:
                lines.append(
                    f"- rank {hit['rank']} | score {hit['score']:.6f} | "
                    f"doc `{hit['doc_id']}` | page {hit['page_num']} | type `{hit['chunk_type']}`"
                )
                preview = hit.get("retrieval_text_preview") or hit.get("evidence_preview") or ""
                if preview:
                    lines.append(f"  - preview: {preview}")
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_by_doc_csv(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "doc_id",
                "filename",
                "strategy_name",
                "avg_score",
                "hit_count",
                "overall_avg_score",
                "overall_hit_count",
            ],
        )
        writer.writeheader()
        for doc_id, data in metrics["document_average_scores"].items():
            for strategy in STRATEGIES:
                strategy_data = data["by_strategy"][strategy]
                writer.writerow(
                    {
                        "doc_id": doc_id,
                        "filename": data["filename"],
                        "strategy_name": strategy,
                        "avg_score": strategy_data["avg_score"],
                        "hit_count": strategy_data["hit_count"],
                        "overall_avg_score": data["avg_score"],
                        "overall_hit_count": data["hit_count"],
                    }
                )


def write_manifest(
    path: Path,
    *,
    run_dir: Path,
    vector_run_dir: Path,
    query_paths: list[Path],
    top_k: int,
    total_queries: int,
) -> None:
    manifest = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "vector_run_dir": str(vector_run_dir),
        "query_files": [str(path) for path in query_paths],
        "top_k": top_k,
        "strategies": list(STRATEGIES),
        "gpt_model_for_report_only": GPT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "total_queries": total_queries,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_retrieval_evaluation(
    *,
    project_root: Path,
    vector_run: str,
    queries: list[dict[str, Any]],
    query_paths: list[Path],
    top_k: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    vector_run_dir = resolve_vector_run(project_root, vector_run)
    vectors = load_vectors(vector_run_dir)
    metadata = load_metadata(vector_run_dir, vectors.shape[0])
    if len(metadata) != vectors.shape[0]:
        raise ValueError("Metadata record count does not match vector row count.")

    query_vectors = embed_queries(queries, project_root)
    if query_vectors.shape[1] != vectors.shape[1]:
        raise ValueError(
            f"Query embedding dimension {query_vectors.shape[1]} does not match vector dimension {vectors.shape[1]}."
        )

    run_dir = make_run_dir(project_root)
    rows = retrieve_topk(queries, query_vectors, vectors, metadata, top_k)
    metrics = compute_metrics(rows, queries, top_k)

    write_manifest(
        run_dir / "run_manifest.json",
        run_dir=run_dir,
        vector_run_dir=vector_run_dir,
        query_paths=query_paths,
        top_k=top_k,
        total_queries=len(queries),
    )
    write_jsonl(run_dir / "retrieval_results.jsonl", rows)
    write_topk_preview(run_dir / "retrieval_topk_preview.md", rows, queries)
    write_by_doc_csv(run_dir / "retrieval_by_doc.csv", metrics)
    return run_dir, rows, metrics
