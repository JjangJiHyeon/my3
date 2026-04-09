"""Optional LangSmith tracing and local latency logging helpers."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def load_langsmith_env(project_root: Path) -> dict[str, str]:
    """Load LangSmith-related keys from .env into process env if needed."""

    dotenv_path = project_root / ".env"
    loaded: dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("LANGSMITH_"):
            continue
        value = value.strip().strip('"').strip("'")
        if key not in os.environ and value:
            os.environ[key] = value
            loaded[key] = value

    tracing = os.environ.get("LANGSMITH_TRACING", "").strip().lower()
    if tracing in {"1", "true", "yes", "on", "local"} and "LANGCHAIN_TRACING_V2" not in os.environ:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        loaded["LANGCHAIN_TRACING_V2"] = "true"

    return loaded


def optional_traceable(*args: Any, **kwargs: Any):
    """Return a no-op decorator when LangSmith is unavailable."""

    def decorator(func: Any) -> Any:
        try:
            from langsmith import traceable
        except Exception:
            return func
        try:
            return traceable(*args, **kwargs)(func)
        except Exception:
            return func

    return decorator


def is_langsmith_enabled() -> bool:
    raw = str(os.getenv("LANGSMITH_TRACING", "")).strip().lower()
    return raw in {"1", "true", "yes", "on", "local"}


def langsmith_project_name(default: str = "my3-rag") -> str:
    value = str(os.getenv("LANGSMITH_PROJECT", default)).strip()
    return value or default


def sanitize_metadata(**kwargs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def append_latency_event(run_dir: Path | None, payload: dict[str, Any]) -> None:
    if run_dir is None:
        return
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "latency_events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to write latency event: %s", exc)


@contextmanager
def stage_timer(run_dir: Path | None, stage: str, **metadata: Any) -> Iterator[dict[str, Any]]:
    event = sanitize_metadata(
        stage=stage,
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        **metadata,
    )
    started = time.perf_counter()
    try:
        yield event
    except Exception as exc:
        event["status"] = "error"
        event["error_type"] = exc.__class__.__name__
        event["error_message"] = str(exc)
        raise
    finally:
        event["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        append_latency_event(run_dir, event)
