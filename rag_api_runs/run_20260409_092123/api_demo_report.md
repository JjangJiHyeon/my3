# RAG API Demo Report

- Run: `run_20260409_092123`
- Chat model: `gpt-5.2`
- Embedding model: `text-embedding-3-large`
- API key source: `.env` or environment variable only

## QA

`POST /qa`

```json
{
  "query": "문서의 핵심 내용을 알려줘",
  "strategy_name": "llm_ready_native",
  "top_k": 5
}
```

## Summary

`POST /summary`

```json
{
  "filename": "example.pdf",
  "strategy_name": "llm_ready_native",
  "top_k": 8
}
```
