"""Run the lightweight QA and summary RAG API."""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_api.observability import (
    is_langsmith_enabled,
    langsmith_project_name,
    load_langsmith_env,
    optional_traceable,
    stage_timer,
)

load_langsmith_env(PROJECT_ROOT)

from rag_api.config import RagApiConfigError, load_config
from rag_api.qa_chain import run_qa_chain, stream_qa_chain
from rag_api.retriever import ChromaRetriever
from rag_api.schemas import QARequest, RagResponse, SummaryRequest
from rag_api.summary_chain import run_summary_chain

app = FastAPI(title="LangChain-lite RAG API")
RUN_DIR = PROJECT_ROOT / "rag_api_runs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")


@app.on_event("startup")
def startup() -> None:
    _write_run_artifacts(RUN_DIR)


@app.post("/qa", response_model=RagResponse)
def qa(request: QARequest) -> RagResponse:
    try:
        return _handle_qa_request(request)
    except (RagApiConfigError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/qa/stream")
def qa_stream(request: QARequest) -> StreamingResponse:
    return StreamingResponse(
        _handle_qa_stream_request(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/summary", response_model=RagResponse)
def summary(request: SummaryRequest) -> RagResponse:
    try:
        return _handle_summary_request(request)
    except (RagApiConfigError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@optional_traceable(name="rag_api.qa_request", run_type="chain")
def _handle_qa_request(request: QARequest) -> RagResponse:
    request_id = f"qa-{uuid.uuid4().hex[:12]}"
    with stage_timer(
        RUN_DIR,
        "request.qa.total",
        request_id=request_id,
        strategy_name=request.strategy_name,
        top_k=request.top_k,
        filename_filter=request.filename_filter,
        query_len=len(request.query),
    ) as event:
        with stage_timer(RUN_DIR, "request.config.load", request_id=request_id):
            config = load_config(PROJECT_ROOT, run_dir=RUN_DIR)
        documents = ChromaRetriever(config).retrieve(
            query=request.query,
            strategy_name=request.strategy_name,
            top_k=request.top_k,
            filename_filter=request.filename_filter,
            request_id=request_id,
        )
        event["retrieved_docs"] = len(documents)
        response = run_qa_chain(config, request.query, documents, request_id=request_id)
        event["source_count"] = len(response.sources)
        event["answer_chars"] = len(response.answer)
        return response


@optional_traceable(name="rag_api.qa_stream_request", run_type="chain")
def _handle_qa_stream_request(request: QARequest):
    request_id = f"qa-stream-{uuid.uuid4().hex[:12]}"

    def _events():
        try:
            yield _sse_message("status", {"stage": "config", "message": "Loading QA configuration."})
            with stage_timer(
                RUN_DIR,
                "request.qa_stream.total",
                request_id=request_id,
                strategy_name=request.strategy_name,
                top_k=request.top_k,
                filename_filter=request.filename_filter,
                query_len=len(request.query),
            ) as event:
                with stage_timer(RUN_DIR, "request.config.load", request_id=request_id):
                    config = load_config(PROJECT_ROOT, run_dir=RUN_DIR)
                yield _sse_message("status", {"stage": "retrieve", "message": "Retrieving supporting chunks."})
                documents = ChromaRetriever(config).retrieve(
                    query=request.query,
                    strategy_name=request.strategy_name,
                    top_k=request.top_k,
                    filename_filter=request.filename_filter,
                    request_id=request_id,
                )
                event["retrieved_docs"] = len(documents)
                yield _sse_message(
                    "retrieved",
                    {
                        "count": len(documents),
                        "message": f"Retrieved {len(documents)} supporting chunks.",
                    },
                )
                yield _sse_message("status", {"stage": "generate", "message": "Generating answer."})
                chunks, sources = stream_qa_chain(config, request.query, documents, request_id=request_id)
                answer_parts: list[str] = []
                for chunk in chunks:
                    answer_parts.append(chunk)
                    yield _sse_message("token", {"delta": chunk})
                answer = "".join(answer_parts).strip()
                payload = _rag_response_payload(
                    RagResponse(
                        mode="qa",
                        title=request.query,
                        answer=answer,
                        sources=sources,
                    )
                )
                event["source_count"] = len(payload["sources"])
                event["answer_chars"] = len(answer)
                yield _sse_message("done", payload)
        except (RagApiConfigError, RuntimeError) as exc:
            yield _sse_message("error", {"detail": str(exc)})
        except Exception as exc:
            yield _sse_message("error", {"detail": f"QA streaming failed: {exc}"})

    return _events()


@optional_traceable(name="rag_api.summary_request", run_type="chain")
def _handle_summary_request(request: SummaryRequest) -> RagResponse:
    request_id = f"summary-{uuid.uuid4().hex[:12]}"
    with stage_timer(
        RUN_DIR,
        "request.summary.total",
        request_id=request_id,
        filename=request.filename,
        strategy_name=request.strategy_name,
        top_k=request.top_k,
    ) as event:
        with stage_timer(RUN_DIR, "request.config.load", request_id=request_id):
            config = load_config(PROJECT_ROOT, run_dir=RUN_DIR)
        documents = ChromaRetriever(config).retrieve_representative(
            filename=request.filename,
            strategy_name=request.strategy_name,
            top_k=request.top_k,
            request_id=request_id,
        )
        event["retrieved_docs"] = len(documents)
        response = run_summary_chain(config, request.filename, documents, request_id=request_id)
        event["source_count"] = len(response.sources)
        event["answer_chars"] = len(response.answer)
        return response


def _sse_message(event: str, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


def _rag_response_payload(response: RagResponse) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def _write_run_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "api": "LangChain-lite RAG API",
        "chat_model": "gpt-5.2",
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 3072,
        "endpoints": ["/qa", "/summary"],
        "langsmith_tracing_enabled": is_langsmith_enabled(),
        "langsmith_project": langsmith_project_name(),
        "latency_events_path": str(run_dir / "latency_events.jsonl"),
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_text(
        run_dir / "qa_samples.jsonl",
        json.dumps(
            {
                "endpoint": "/qa",
                "request": {
                    "query": "문서의 핵심 내용을 알려줘.",
                    "strategy_name": "text_first_with_visual_support",
                    "top_k": 5,
                },
                "response_shape": _response_shape("qa"),
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    _write_text(
        run_dir / "summary_samples.jsonl",
        json.dumps(
            {
                "endpoint": "/summary",
                "request": {"filename": "example.pdf", "strategy_name": "text_first_with_visual_support", "top_k": 8},
                "response_shape": _response_shape("summary"),
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    _write_text(run_dir / "api_demo_report.md", _demo_report(run_dir.name))


def _response_shape(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "title": "string",
        "answer": "string",
        "sources": [{"document": "string", "page": 0}],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _demo_report(run_id: str) -> str:
    return f"""# RAG API Demo Report

- Run: `{run_id}`
- Chat model: `gpt-5.2`
- Embedding model: `text-embedding-3-large`
- API key source: `.env` or environment variable only

## QA

`POST /qa`

```json
{{
  "query": "문서의 핵심 내용을 알려줘.",
  "strategy_name": "text_first_with_visual_support",
  "top_k": 5
}}
```

## Summary

`POST /summary`

```json
{{
  "filename": "example.pdf",
  "strategy_name": "text_first_with_visual_support",
  "top_k": 8
}}
```
"""


if __name__ == "__main__":
    import uvicorn

    while RUN_DIR.exists():
        time.sleep(1)
    uvicorn.run("scripts.run_rag_api:app", host="127.0.0.1", port=8000, reload=False)
