"""Run chunk strategy quality checks for retrieval-readiness review."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from quality_checks.chunk_metrics import build_chunk_quality_metrics, write_metrics_outputs
from quality_checks.chunk_report import write_chunk_strategy_report


GPT_MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-large"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    chunks_dir = resolve_project_path(args.chunks_dir, project_root, "chunks")
    output_dir = resolve_project_path(args.output_dir, project_root, "chunk_eval_runs")

    if not chunks_dir.exists():
        raise FileNotFoundError(f"chunks directory not found: {chunks_dir}")

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics = build_chunk_quality_metrics(chunks_dir)
    manifest = build_manifest(run_id, chunks_dir, output_dir, run_dir, metrics)

    write_json(run_dir / "run_manifest.json", manifest)
    write_metrics_outputs(metrics, run_dir)
    write_chunk_strategy_report(metrics, manifest, run_dir / "chunk_strategy_comparison.md")

    print(f"Chunk quality check run created: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare chunking strategy outputs before embedding/retrieval."
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=None,
        help="Read-only directory containing chunk JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where run_YYYYMMDD_HHMMSS outputs will be created.",
    )
    return parser.parse_args()


def resolve_project_path(value: Path | None, project_root: Path, default_name: str) -> Path:
    if value is None:
        return project_root / default_name
    if value.is_absolute():
        return value
    return (Path.cwd() / value).resolve()


def build_manifest(
    run_id: str,
    chunks_dir: Path,
    output_dir: Path,
    run_dir: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": {
            "gpt_model": GPT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        },
        "input": {
            "chunks_dir": str(chunks_dir),
            "read_only_patterns": [
                "*.text_first_with_visual_support.json",
                "*.llm_ready_native.json",
                "*.chunk_summary.json",
            ],
            "strategy_file_count": metrics["input"]["strategy_file_count"],
            "expected_doc_count": metrics["input"]["expected_doc_count"],
        },
        "output": {
            "output_dir": str(output_dir),
            "run_dir": str(run_dir),
            "files": [
                "run_manifest.json",
                "chunk_strategy_metrics.json",
                "chunk_strategy_metrics.csv",
                "chunk_strategy_comparison.md",
                "doc_level_chunk_summary.json",
            ],
        },
        "notes": [
            "OPENAI_API_KEY is not read by this offline quality check.",
            "Existing chunks/*.json files are treated as read-only inputs.",
            "No document, company, quarter, or test-document-specific logic is used.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
