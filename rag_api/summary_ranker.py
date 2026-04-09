"""Representative chunk reranking for document-wide summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


TOC_TERMS = ("contents", "table of contents", "toc", "\ubaa9\ucc28")
APPENDIX_TERMS = ("appendix", "appendices", "\ubd80\ub85d", "\ucc38\uace0\uc790\ub8cc")
VISUAL_TYPES = {"chart", "image", "figure"}
TITLE_TYPES = {"title"}
MIN_EXPLANATORY_CHARS = 120


@dataclass(frozen=True)
class RankedDocument:
    document: Any
    score: float
    penalties: dict[str, float]


def rerank_summary_documents(documents: list[Any], top_k: int) -> list[Any]:
    """Favor broad explanatory chunks and diversify by page/section."""
    ranked = sorted(
        (_score_document(document) for document in documents),
        key=lambda item: item.score,
        reverse=True,
    )

    selected: list[RankedDocument] = []
    page_counts: dict[str, int] = {}
    section_counts: dict[str, int] = {}
    for item in ranked:
        if len(selected) >= top_k:
            break
        metadata = _metadata(item.document)
        page_key = str(metadata.get("page_num", metadata.get("page", "")))
        section_key = _section_key(metadata)
        if page_key and page_counts.get(page_key, 0) >= 1 and len(selected) < max(2, top_k // 2):
            continue
        if section_key and section_counts.get(section_key, 0) >= 2:
            continue
        selected.append(item)
        if page_key:
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
        if section_key:
            section_counts[section_key] = section_counts.get(section_key, 0) + 1

    if len(selected) < top_k:
        seen = {id(item.document) for item in selected}
        for item in ranked:
            if len(selected) >= top_k:
                break
            if id(item.document) in seen:
                continue
            selected.append(item)
            seen.add(id(item.document))

    for item in selected:
        metadata = _metadata(item.document)
        metadata["_summary_rank_score"] = round(item.score, 6)
        metadata["_summary_rank_penalties"] = item.penalties
    return [item.document for item in selected]


def _score_document(document: Any) -> RankedDocument:
    metadata = _metadata(document)
    text = _text(document)
    lower_text = text.lower()
    chunk_type = str(metadata.get("chunk_type") or "").lower()
    base_score = _base_score(metadata)
    penalties: dict[str, float] = {}

    if _has_any(lower_text, TOC_TERMS) or _has_any(_title_text(metadata).lower(), TOC_TERMS):
        penalties["toc_like"] = 0.28
    if _has_any(lower_text, APPENDIX_TERMS) or _has_any(_title_text(metadata).lower(), APPENDIX_TERMS):
        penalties["appendix_like"] = 0.22
    if chunk_type in TITLE_TYPES or _looks_title_only(text, metadata):
        penalties["title_only"] = 0.2
    if chunk_type in VISUAL_TYPES or _looks_visual_only(text, metadata):
        penalties["visual_only"] = 0.18
    if _looks_caption_or_numeric_only(text):
        penalties["caption_or_numeric_only"] = 0.16

    length_bonus = _length_bonus(text)
    explanatory_bonus = 0.08 if _looks_explanatory(text) else 0.0
    score = base_score + length_bonus + explanatory_bonus - sum(penalties.values())
    return RankedDocument(document=document, score=score, penalties=penalties)


def _metadata(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def _text(document: Any) -> str:
    return " ".join(str(getattr(document, "page_content", "") or "").split())


def _base_score(metadata: dict[str, Any]) -> float:
    for key in ("_summary_raw_score", "relevance_score", "score"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _length_bonus(text: str) -> float:
    length = len(text)
    if length >= 350:
        return 0.12
    if length >= MIN_EXPLANATORY_CHARS:
        return 0.08
    if length >= 60:
        return 0.02
    return -0.08


def _looks_explanatory(text: str) -> bool:
    if len(text) < MIN_EXPLANATORY_CHARS:
        return False
    words = re.findall(r"[A-Za-z0-9\u3131-\uD7A3]+", text)
    return len(words) >= 12


def _looks_title_only(text: str, metadata: dict[str, Any]) -> bool:
    title = _title_text(metadata)
    if len(text) > 90:
        return False
    if not title:
        return len(text.split()) <= 8
    return text.strip().lower() == title.strip().lower() or len(text.split()) <= 8


def _looks_visual_only(text: str, metadata: dict[str, Any]) -> bool:
    has_text = metadata.get("has_text")
    has_visual = bool(metadata.get("has_chart") or metadata.get("has_image"))
    if has_visual and has_text is False:
        return True
    if has_visual and len(text) < MIN_EXPLANATORY_CHARS:
        return True
    return False


def _looks_caption_or_numeric_only(text: str) -> bool:
    if not text:
        return True
    if len(text) >= MIN_EXPLANATORY_CHARS:
        return False
    alpha_tokens = re.findall(r"[A-Za-z\u3131-\uD7A3]{2,}", text)
    numeric_tokens = re.findall(r"\d+(?:[.,]\d+)?%?", text)
    return len(numeric_tokens) >= max(2, len(alpha_tokens) * 2)


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _title_text(metadata: dict[str, Any]) -> str:
    return " ".join(
        str(metadata.get(key) or "")
        for key in ("page_title", "section_title", "associated_title")
    ).strip()


def _section_key(metadata: dict[str, Any]) -> str:
    for key in ("section_title", "page_title", "associated_title", "chunk_type"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value.lower()
    return ""
