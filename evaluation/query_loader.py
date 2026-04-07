"""Load fixed retrieval evaluation query sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_QUERY_FIELDS = ("query_id", "query_text")


def load_query_files(paths: list[Path]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("queries", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Query file must contain a list or a queries list: {path}")

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Query row must be an object in {path}: {row!r}")
            missing = [field for field in REQUIRED_QUERY_FIELDS if not row.get(field)]
            if missing:
                raise ValueError(f"Query row missing {missing} in {path}: {row!r}")

            query_id = str(row["query_id"])
            if query_id in seen_ids:
                raise ValueError(f"Duplicate query_id across query files: {query_id}")
            seen_ids.add(query_id)

            queries.append(
                {
                    "query_id": query_id,
                    "query_text": str(row["query_text"]),
                    "category": str(row.get("category", "")),
                    "intent": str(row.get("intent", "")),
                    "source_file": str(path),
                }
            )

    if not queries:
        raise ValueError("No evaluation queries were loaded.")
    return queries

