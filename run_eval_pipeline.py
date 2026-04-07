"""Run retrieval strategy evaluation and report generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.query_loader import load_query_files
from evaluation.report_builder import write_reports
from evaluation.retrieval_eval import run_retrieval_evaluation


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies with top-k cosine similarity.")
    parser.add_argument("--vector-run", default="latest", help="Vector run id/path under vector_indexes, or latest.")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            str(project_root / "eval_queries" / "default_queries.json"),
            str(project_root / "eval_queries" / "finance_queries.json"),
        ],
        help="One or more query JSON files.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k hits per query and strategy.")
    return parser


def main() -> None:
    project_root = Path(__file__).resolve().parent
    args = build_parser().parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    query_paths = [Path(path).resolve() for path in args.queries]
    queries = load_query_files(query_paths)
    run_dir, rows, metrics = run_retrieval_evaluation(
        project_root=project_root,
        vector_run=args.vector_run,
        queries=queries,
        query_paths=query_paths,
        top_k=args.top_k,
    )
    write_reports(project_root, run_dir, rows, metrics)
    print(f"Evaluation run written to: {run_dir}")


if __name__ == "__main__":
    main()
