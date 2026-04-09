"""Chroma persistent collection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import COLLECTION_NAME, EMBEDDING_MODEL


def make_persistent_client(persist_path: Path) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required for Chroma ingestion.") from exc
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


def get_or_create_collection(client: Any, collection_name: str = COLLECTION_NAME) -> Any:
    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "source_contract": "file_based_vector_indexes",
        },
    )


def add_records_to_collection(
    collection: Any,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
    embeddings: list[list[float]],
    batch_size: int = 256,
) -> int:
    inserted = 0
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )
        inserted += len(ids[start:end])
    return inserted
