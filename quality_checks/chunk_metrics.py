"""Metrics for comparing chunking strategies before retrieval."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRATEGIES = ("text_first_with_visual_support", "llm_ready_native")
LONG_CHUNK_CHAR_THRESHOLD = 1500
SHORT_CHUNK_CHAR_THRESHOLD = 120


@dataclass(frozen=True)
class StrategyInput:
    """Loaded chunk file for one document and one strategy."""

    doc_id: str
    strategy_name: str
    path: Path
    payload: dict[str, Any]


def load_strategy_inputs(chunks_dir: Path) -> list[StrategyInput]:
    """Load read-only chunk files for the supported comparison strategies."""
    loaded: list[StrategyInput] = []
    for strategy in STRATEGIES:
        for path in sorted(chunks_dir.glob(f"*.{strategy}.json")):
            payload = _read_json(path)
            doc_id = str(payload.get("doc_id") or path.name.split(".")[0])
            loaded.append(
                StrategyInput(
                    doc_id=doc_id,
                    strategy_name=str(payload.get("strategy_name") or strategy),
                    path=path,
                    payload=payload,
                )
            )
    return loaded


def build_chunk_quality_metrics(chunks_dir: Path) -> dict[str, Any]:
    """Build strategy-level and document-level chunk quality metrics."""
    inputs = load_strategy_inputs(chunks_dir)
    summary_files = sorted(chunks_dir.glob("*.chunk_summary.json"))
    expected_doc_ids = sorted({item.doc_id for item in inputs})
    by_strategy: dict[str, list[StrategyInput]] = {
        strategy: [item for item in inputs if item.strategy_name == strategy]
        for strategy in STRATEGIES
    }

    strategy_metrics = {
        strategy: _summarize_strategy(strategy, items, expected_doc_ids)
        for strategy, items in by_strategy.items()
    }
    doc_summaries = _build_doc_level_summary(inputs)

    return {
        "schema_version": 1,
        "strategies": list(STRATEGIES),
        "thresholds": {
            "long_chunk_char_len_gt": LONG_CHUNK_CHAR_THRESHOLD,
            "short_chunk_char_len_lt": SHORT_CHUNK_CHAR_THRESHOLD,
        },
        "input": {
            "chunks_dir": str(chunks_dir),
            "expected_doc_count": len(expected_doc_ids),
            "expected_doc_ids": expected_doc_ids,
            "strategy_file_count": {strategy: len(items) for strategy, items in by_strategy.items()},
            "chunk_summary_file_count": len(summary_files),
        },
        "strategy_metrics": strategy_metrics,
        "doc_level_summary": doc_summaries,
    }


def write_metrics_outputs(metrics: dict[str, Any], run_dir: Path) -> None:
    """Write JSON and CSV metric outputs."""
    _write_json(run_dir / "chunk_strategy_metrics.json", metrics)
    _write_json(run_dir / "doc_level_chunk_summary.json", metrics["doc_level_summary"])
    _write_strategy_csv(metrics, run_dir / "chunk_strategy_metrics.csv")


def _summarize_strategy(
    strategy_name: str, inputs: list[StrategyInput], expected_doc_ids: list[str]
) -> dict[str, Any]:
    chunks = [chunk for item in inputs for chunk in _chunks(item.payload)]
    char_lengths = [_int_or_len(chunk, "char_len", "retrieval_text") for chunk in chunks]
    token_estimates = [_int_or_estimate_tokens(chunk) for chunk in chunks]
    pages = sorted({_safe_int(chunk.get("page_num")) for chunk in chunks if chunk.get("page_num") is not None})
    doc_page_pairs = {
        (str(chunk.get("doc_id") or ""), _safe_int(chunk.get("page_num")))
        for chunk in chunks
        if chunk.get("page_num") is not None
    }
    doc_chunk_counts = {item.doc_id: len(_chunks(item.payload)) for item in inputs}
    doc_count_values = [doc_chunk_counts.get(doc_id, 0) for doc_id in expected_doc_ids]

    chunk_type_distribution = Counter(str(chunk.get("chunk_type") or "unknown") for chunk in chunks)
    flag_counts = {
        flag: sum(1 for chunk in chunks if bool(chunk.get(flag)))
        for flag in ("has_table", "has_chart", "has_image", "has_text")
    }
    source_block_ids_empty_count = sum(
        1 for chunk in chunks if not chunk.get("source_block_ids")
    )
    evidence_preview_count = sum(
        1 for chunk in chunks if str(chunk.get("evidence_preview") or "").strip()
    )
    sparse_page_related_chunk_count = sum(1 for chunk in chunks if _is_sparse_related(chunk))
    empty_retrieval_text_count = sum(
        1 for chunk in chunks if not str(chunk.get("retrieval_text") or "").strip()
    )

    total_chunks = len(chunks)
    total_docs = len({item.doc_id for item in inputs})
    expected_docs = len(expected_doc_ids)

    return {
        "strategy_name": strategy_name,
        "total_chunks": total_chunks,
        "total_docs": total_docs,
        "doc_coverage": _ratio(total_docs, expected_docs),
        "covered_doc_ids": sorted({item.doc_id for item in inputs}),
        "page_coverage": {
            "covered_pages": pages,
            "covered_page_count": len(pages),
            "covered_doc_page_count": len(doc_page_pairs),
            "per_doc_covered_page_count": {
                item.doc_id: len(
                    {
                        _safe_int(chunk.get("page_num"))
                        for chunk in _chunks(item.payload)
                        if chunk.get("page_num") is not None
                    }
                )
                for item in inputs
            },
        },
        "empty_retrieval_text_count": empty_retrieval_text_count,
        "avg_char_len": _mean(char_lengths),
        "median_char_len": _median(char_lengths),
        "avg_token_estimate": _mean(token_estimates),
        "chunk_type_distribution": dict(sorted(chunk_type_distribution.items())),
        "source_block_ids_empty_count": source_block_ids_empty_count,
        "content_flag_ratios": {
            flag: _ratio(count, total_chunks) for flag, count in flag_counts.items()
        },
        "content_flag_counts": flag_counts,
        "evidence_preview_presence_ratio": _ratio(evidence_preview_count, total_chunks),
        "evidence_preview_count": evidence_preview_count,
        "sparse_page_related_chunk_ratio": _ratio(sparse_page_related_chunk_count, total_chunks),
        "sparse_page_related_chunk_count": sparse_page_related_chunk_count,
        "chunk_count_by_doc": doc_chunk_counts,
        "chunk_count_variance_by_doc": {
            "min": min(doc_count_values) if doc_count_values else 0,
            "max": max(doc_count_values) if doc_count_values else 0,
            "range": (max(doc_count_values) - min(doc_count_values)) if doc_count_values else 0,
            "population_stdev": round(statistics.pstdev(doc_count_values), 4)
            if len(doc_count_values) > 1
            else 0.0,
        },
        "long_chunk_count": sum(1 for value in char_lengths if value > LONG_CHUNK_CHAR_THRESHOLD),
        "short_chunk_count": sum(1 for value in char_lengths if value < SHORT_CHUNK_CHAR_THRESHOLD),
        "min_char_len": min(char_lengths) if char_lengths else 0,
        "max_char_len": max(char_lengths) if char_lengths else 0,
    }


def _build_doc_level_summary(inputs: list[StrategyInput]) -> list[dict[str, Any]]:
    by_doc: dict[str, list[StrategyInput]] = defaultdict(list)
    for item in inputs:
        by_doc[item.doc_id].append(item)

    summaries: list[dict[str, Any]] = []
    for doc_id in sorted(by_doc):
        strategy_rows = {}
        filename = ""
        for item in sorted(by_doc[doc_id], key=lambda loaded: loaded.strategy_name):
            chunks = _chunks(item.payload)
            filename = filename or str(item.payload.get("filename") or "")
            char_lengths = [_int_or_len(chunk, "char_len", "retrieval_text") for chunk in chunks]
            pages = sorted(
                {_safe_int(chunk.get("page_num")) for chunk in chunks if chunk.get("page_num") is not None}
            )
            strategy_rows[item.strategy_name] = {
                "total_chunks": len(chunks),
                "page_coverage_count": len(pages),
                "doc_page_coverage_count": len(
                    {
                        (str(chunk.get("doc_id") or doc_id), _safe_int(chunk.get("page_num")))
                        for chunk in chunks
                        if chunk.get("page_num") is not None
                    }
                ),
                "covered_pages": pages,
                "avg_char_len": _mean(char_lengths),
                "median_char_len": _median(char_lengths),
                "chunk_type_distribution": dict(
                    sorted(Counter(str(chunk.get("chunk_type") or "unknown") for chunk in chunks).items())
                ),
                "empty_retrieval_text_count": sum(
                    1 for chunk in chunks if not str(chunk.get("retrieval_text") or "").strip()
                ),
                "source_block_ids_empty_count": sum(
                    1 for chunk in chunks if not chunk.get("source_block_ids")
                ),
                "sparse_page_related_chunk_count": sum(1 for chunk in chunks if _is_sparse_related(chunk)),
                "long_chunk_count": sum(1 for value in char_lengths if value > LONG_CHUNK_CHAR_THRESHOLD),
                "short_chunk_count": sum(1 for value in char_lengths if value < SHORT_CHUNK_CHAR_THRESHOLD),
                "content_flag_counts": {
                    flag: sum(1 for chunk in chunks if bool(chunk.get(flag)))
                    for flag in ("has_table", "has_chart", "has_image", "has_text")
                },
            }
        summaries.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "strategies": strategy_rows,
                "strategy_difference": _strategy_difference(strategy_rows),
            }
        )
    return summaries


def _strategy_difference(strategy_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    text_first = strategy_rows.get("text_first_with_visual_support", {})
    native = strategy_rows.get("llm_ready_native", {})
    return {
        "chunk_count_delta_text_first_minus_llm_ready_native": int(text_first.get("total_chunks", 0))
        - int(native.get("total_chunks", 0)),
        "page_coverage_delta_text_first_minus_llm_ready_native": int(
            text_first.get("page_coverage_count", 0)
        )
        - int(native.get("page_coverage_count", 0)),
        "avg_char_len_delta_text_first_minus_llm_ready_native": round(
            float(text_first.get("avg_char_len", 0.0)) - float(native.get("avg_char_len", 0.0)),
            4,
        ),
        "short_chunk_delta_text_first_minus_llm_ready_native": int(text_first.get("short_chunk_count", 0))
        - int(native.get("short_chunk_count", 0)),
        "long_chunk_delta_text_first_minus_llm_ready_native": int(text_first.get("long_chunk_count", 0))
        - int(native.get("long_chunk_count", 0)),
    }


def _write_strategy_csv(metrics: dict[str, Any], path: Path) -> None:
    rows = []
    for strategy, row in metrics["strategy_metrics"].items():
        rows.append(
            {
                "strategy_name": strategy,
                "total_chunks": row["total_chunks"],
                "total_docs": row["total_docs"],
                "doc_coverage": row["doc_coverage"],
                "page_coverage_count": row["page_coverage"]["covered_page_count"],
                "doc_page_coverage_count": row["page_coverage"]["covered_doc_page_count"],
                "empty_retrieval_text_count": row["empty_retrieval_text_count"],
                "avg_char_len": row["avg_char_len"],
                "median_char_len": row["median_char_len"],
                "avg_token_estimate": row["avg_token_estimate"],
                "source_block_ids_empty_count": row["source_block_ids_empty_count"],
                "has_table_ratio": row["content_flag_ratios"]["has_table"],
                "has_chart_ratio": row["content_flag_ratios"]["has_chart"],
                "has_image_ratio": row["content_flag_ratios"]["has_image"],
                "has_text_ratio": row["content_flag_ratios"]["has_text"],
                "evidence_preview_presence_ratio": row["evidence_preview_presence_ratio"],
                "sparse_page_related_chunk_ratio": row["sparse_page_related_chunk_ratio"],
                "doc_chunk_count_stdev": row["chunk_count_variance_by_doc"]["population_stdev"],
                "long_chunk_count": row["long_chunk_count"],
                "short_chunk_count": row["short_chunk_count"],
                "chunk_type_distribution_json": json.dumps(
                    row["chunk_type_distribution"], ensure_ascii=False, sort_keys=True
                ),
            }
        )

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _chunks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = payload.get("chunks")
    return chunks if isinstance(chunks, list) else []


def _is_sparse_related(chunk: dict[str, Any]) -> bool:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    parser_debug = metadata.get("parser_debug") if isinstance(metadata.get("parser_debug"), dict) else {}
    return bool(
        metadata.get("sparse_page")
        or parser_debug.get("sparse_page")
        or parser_debug.get("near_empty_reason")
    )


def _int_or_len(chunk: dict[str, Any], key: str, fallback_text_key: str) -> int:
    value = chunk.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return len(str(chunk.get(fallback_text_key) or ""))


def _int_or_estimate_tokens(chunk: dict[str, Any]) -> int:
    value = chunk.get("token_estimate")
    if isinstance(value, (int, float)):
        return int(value)
    return max(1, _int_or_len(chunk, "char_len", "retrieval_text") // 4)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mean(values: list[int]) -> float:
    return round(float(statistics.fmean(values)), 4) if values else 0.0


def _median(values: list[int]) -> float:
    return round(float(statistics.median(values)), 4) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
