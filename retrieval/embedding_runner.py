"""OpenAI embedding runner for retrieval preparation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np

from .metadata_schema import EMBEDDING_MODEL, EXPECTED_EMBEDDING_DIMENSIONS


def load_openai_api_key(project_root: Path) -> str:
    """Read OPENAI_API_KEY from the environment or project .env only."""
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    env_path = project_root / ".env"
    if not env_path.exists():
        raise RuntimeError("OPENAI_API_KEY was not found in the environment or project .env file.")

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value

    raise RuntimeError("OPENAI_API_KEY was not found in the environment or project .env file.")


def _make_openai_client(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI(api_key=api_key)


def _embed_with_urllib(api_key: str, model: str, batch: list[str]) -> list[list[float]]:
    body = json.dumps({"model": model, "input": batch}).encode("utf-8")
    http_request = request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI embeddings request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI embeddings request failed: {exc.reason}") from exc

    data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in data]


def embed_texts(
    texts: list[str],
    project_root: Path,
    model: str = EMBEDDING_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, EXPECTED_EMBEDDING_DIMENSIONS), dtype=np.float32)

    api_key = load_openai_api_key(project_root)
    client = _make_openai_client(api_key)
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if client is not None:
            response = client.embeddings.create(model=model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        else:
            vectors.extend(_embed_with_urllib(api_key, model, batch))

    return np.asarray(vectors, dtype=np.float32)
