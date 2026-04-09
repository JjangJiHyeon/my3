"""Configuration helpers for the lightweight RAG API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .observability import is_langsmith_enabled, langsmith_project_name

GPT_MODEL = "gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
DEFAULT_STRATEGY_NAME = "llm_ready_native"


@dataclass(frozen=True)
class RagApiConfig:
    project_root: Path
    openai_api_key: str
    chat_model: str = GPT_MODEL
    embedding_model: str = EMBEDDING_MODEL
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
    chroma_root: Path | None = None
    latest_index_path: Path | None = None
    run_dir: Path | None = None
    langsmith_enabled: bool = False
    langsmith_project: str = "my3-rag"


class RagApiConfigError(RuntimeError):
    """Raised when the RAG API cannot be configured safely."""


def _read_dotenv(project_root: Path) -> dict[str, str]:
    dotenv_path = project_root / ".env"
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_value(key: str, dotenv_values: dict[str, str], default: str | None = None) -> str | None:
    return os.environ.get(key) or dotenv_values.get(key) or default


def load_config(project_root: Path | None = None, run_dir: Path | None = None) -> RagApiConfig:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    dotenv_values = _read_dotenv(root)
    api_key = _env_value("OPENAI_API_KEY", dotenv_values)
    if not api_key:
        raise RagApiConfigError("OPENAI_API_KEY must be set in .env or the environment.")

    chat_model = _env_value("OPENAI_CHAT_MODEL", dotenv_values, GPT_MODEL)
    embedding_model = _env_value("OPENAI_EMBEDDING_MODEL", dotenv_values, EMBEDDING_MODEL)
    dimensions_raw = _env_value("OPENAI_EMBEDDING_DIMENSIONS", dotenv_values, str(EMBEDDING_DIMENSIONS))

    if chat_model != GPT_MODEL:
        raise RagApiConfigError(f"OPENAI_CHAT_MODEL must be {GPT_MODEL}.")
    if embedding_model != EMBEDDING_MODEL:
        raise RagApiConfigError(f"OPENAI_EMBEDDING_MODEL must be {EMBEDDING_MODEL}.")

    try:
        embedding_dimensions = int(dimensions_raw or EMBEDDING_DIMENSIONS)
    except ValueError as exc:
        raise RagApiConfigError("OPENAI_EMBEDDING_DIMENSIONS must be an integer.") from exc

    if embedding_dimensions != EMBEDDING_DIMENSIONS:
        raise RagApiConfigError(f"OPENAI_EMBEDDING_DIMENSIONS must be {EMBEDDING_DIMENSIONS}.")

    chroma_root = root / "chroma_indexes"
    latest_index_path = chroma_root / "latest.json"
    return RagApiConfig(
        project_root=root,
        openai_api_key=api_key,
        chat_model=chat_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        chroma_root=chroma_root,
        latest_index_path=latest_index_path,
        run_dir=run_dir,
        langsmith_enabled=is_langsmith_enabled(),
        langsmith_project=langsmith_project_name(),
    )


def load_latest_chroma_index(config: RagApiConfig) -> dict[str, Any]:
    latest_path = config.latest_index_path
    if latest_path is None or not latest_path.exists():
        raise RagApiConfigError("Missing chroma_indexes/latest.json. Build or copy the Chroma index before starting API.")

    with latest_path.open("r", encoding="utf-8") as handle:
        latest = json.load(handle)

    manifest_path = _resolve_index_path(config.project_root, latest.get("manifest_path"), "chroma_indexes", latest.get("run_id"))
    if manifest_path and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            latest["manifest"] = json.load(handle)
    return latest


def resolve_chroma_persist_dir(config: RagApiConfig, latest: dict[str, Any]) -> Path:
    for key in ("persist_path", "chroma_persist_dir", "persist_directory", "chroma_dir", "run_dir"):
        path = _resolve_index_path(config.project_root, latest.get(key), "chroma_indexes", latest.get("run_id"))
        if path and path.exists():
            return path
    raise RagApiConfigError("Could not resolve Chroma persist directory from chroma_indexes/latest.json.")


def resolve_collection_name(latest: dict[str, Any]) -> str | None:
    manifest = latest.get("manifest") if isinstance(latest.get("manifest"), dict) else {}
    value = latest.get("collection_name") or manifest.get("collection_name") or manifest.get("chroma_collection_name")
    return str(value) if value else None


def _resolve_index_path(
    project_root: Path,
    raw_path: Any,
    artifact_root_name: str = "chroma_indexes",
    run_id: Any = None,
) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        if path.parts and path.parts[0] == project_root.name:
            return (project_root.parent / path).resolve()
        return (project_root / path).resolve()

    if path.exists():
        return path

    parts = path.parts
    artifact_root = project_root / artifact_root_name
    if artifact_root_name in parts:
        tail = Path(*parts[parts.index(artifact_root_name) + 1 :])
        candidate = artifact_root / tail
        if candidate.exists():
            return candidate

    fallback_run_id = str(run_id) if run_id else next((part for part in reversed(parts) if part.startswith("run_")), None)
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
