"""Run ChromaDB ingestion from file-based vector artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma_store.config import COLLECTION_NAME, resolve_project_path
from chroma_store.ingest import ingest_vector_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest prepared vectors into a Chroma persistent collection.")
    parser.add_argument("--vector-run", default="latest", help="Vector run id, run directory, or 'latest'.")
    parser.add_argument("--persist-dir", default="chroma_indexes", help="Directory for Chroma run artifacts.")
    parser.add_argument("--collection-name", default=COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--batch-size", type=int, default=256, help="Chroma upsert batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    vector_indexes_dir = project_root / "vector_indexes"
    persist_dir = resolve_project_path(args.persist_dir, project_root)

    summary = ingest_vector_run(
        vector_indexes_dir=vector_indexes_dir,
        output_dir=persist_dir,
        vector_run=args.vector_run,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
    )
    print(
        f"Created {summary['run_id']} in Chroma collection '{summary['collection_name']}' "
        f"with {summary['success_count']} successes and {summary['failure_count']} failures."
    )


if __name__ == "__main__":
    main()
