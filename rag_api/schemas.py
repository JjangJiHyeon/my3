"""Request and response schemas for the RAG API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .config import DEFAULT_STRATEGY_NAME


class QARequest(BaseModel):
    query: str = Field(..., min_length=1)
    strategy_name: str = DEFAULT_STRATEGY_NAME
    top_k: int = Field(default=5, ge=1, le=30)
    filename_filter: str | None = None


class SummaryRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    strategy_name: str = DEFAULT_STRATEGY_NAME
    top_k: int = Field(default=8, ge=1, le=50)


class Source(BaseModel):
    document: str
    page: int = 0


class RagResponse(BaseModel):
    mode: Literal["qa", "summary"]
    title: str
    answer: str
    sources: list[Source]

