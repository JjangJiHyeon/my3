"""Run retrieval metadata normalization and embedding vector preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.vector_prep import prepare_vectors


def resolve_project_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == project_root.name:
        return (project_root.parent / path).resolve()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (project_root / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare retrieval metadata records and embedding vectors.")
    parser.add_argument("--chunks-dir", default="chunks", help="Directory containing chunk JSON files.")
    parser.add_argument("--output-dir", default="vector_indexes", help="Directory where vector artifacts are written.")
    parser.add_argument("--embedding-batch-size", type=int, default=64, help="OpenAI embedding batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    chunks_dir = resolve_project_path(args.chunks_dir, project_root)
    output_dir = resolve_project_path(args.output_dir, project_root)

    manifest = prepare_vectors(
        chunks_dir=chunks_dir,
        output_dir=output_dir,
        project_root=project_root,
        embedding_batch_size=args.embedding_batch_size,
    )
    print(
        f"Created {manifest['run_id']} with {manifest['embedded_chunks']} embeddings "
        f"from {manifest['total_input_chunks']} chunks."
    )


if __name__ == "__main__":
    main()
