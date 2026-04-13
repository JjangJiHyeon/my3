"""Format retrieved document metadata into response source shapes."""

from __future__ import annotations

import re
from typing import Any

from retrieval.keyword_index import classify_keyword_token, tokenize_keyword_text

from .hwp_viewer_page_mapper import preferred_page_num
from .schemas import Source


def format_sources(documents: list[Any]) -> list[Source]:
    seen: set[tuple[str, int]] = set()
    sources: list[Source] = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        filename = str(metadata.get("filename") or metadata.get("document") or "").strip()
        if not filename:
            continue
        page = preferred_page_num(metadata)
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(document=filename, page=page))
    return sources


def format_qa_sources(
    query: str,
    documents: list[Any],
    answer: str | None = None,
    max_sources: int = 2,
) -> list[Source]:
    base_candidates = _clustered_qa_source_candidates(query, documents)
    if not base_candidates:
        return format_sources(documents[:max_sources]) if documents and answer is None else []

    candidates = base_candidates
    if answer is not None:
        if _answer_indicates_no_support(answer):
            return []
        candidates = _prune_qa_source_candidates(answer, base_candidates) or base_candidates[:1]

    selected = candidates[:1]
    if max_sources > 1 and len(candidates) > 1 and _should_include_secondary_source(selected[0], candidates[1]):
        selected.append(candidates[1])
    return [Source(document=item["filename"], page=item["page"]) for item in selected]


def format_qa_source_markdown(sources: list[Source]) -> str:
    if not sources:
        return ""
    lines = ["", "## 출처", ""]
    for source in sources:
        lines.append(f"- {source.document} · p.{int(source.page or 0)}")
    return "\n".join(lines).strip()


def _clustered_qa_source_candidates(query: str, documents: list[Any]) -> list[dict[str, Any]]:
    query_profile = _query_source_profile(query)
    page_candidates: dict[tuple[str, int], dict[str, Any]] = {}
    for index, document in enumerate(documents):
        metadata = getattr(document, "metadata", {}) or {}
        filename = str(metadata.get("filename") or metadata.get("document") or "").strip()
        if not filename:
            continue
        page = preferred_page_num(metadata)
        if page <= 0:
            continue
        candidate = _qa_source_candidate(query_profile, document, index)
        key = (filename, page)
        existing = page_candidates.get(key)
        if existing is None or candidate["score"] > existing["score"]:
            page_candidates[key] = candidate

    by_document: dict[str, list[dict[str, Any]]] = {}
    for candidate in page_candidates.values():
        by_document.setdefault(candidate["filename"], []).append(candidate)

    clustered: list[dict[str, Any]] = []
    for rows in by_document.values():
        rows.sort(key=lambda item: (item["page"], -item["score"]))
        current_cluster: list[dict[str, Any]] = []
        previous_page: int | None = None
        for row in rows:
            if previous_page is None or row["page"] <= previous_page + 1:
                current_cluster.append(row)
            else:
                clustered.append(_best_cluster_candidate(current_cluster))
                current_cluster = [row]
            previous_page = row["page"]
        if current_cluster:
            clustered.append(_best_cluster_candidate(current_cluster))

    clustered.sort(
        key=lambda item: (
            item["score"],
            item["has_numeric_match"],
            item["has_date_match"],
            -item["doc_rank"],
        ),
        reverse=True,
    )
    return clustered


def _best_cluster_candidate(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        cluster,
        key=lambda item: (
            item["score"],
            item["has_numeric_match"],
            item["has_date_match"],
            -item["doc_rank"],
        ),
    )


def _qa_source_candidate(query_profile: dict[str, Any], document: Any, doc_rank: int) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    filename = str(metadata.get("filename") or metadata.get("document") or "").strip()
    page = preferred_page_num(metadata)
    text = _document_text(document)
    doc_tokens = set(tokenize_keyword_text(text))
    numeric_match_count = sum(1 for token in query_profile["numeric_tokens"] if token in doc_tokens)
    date_match_count = sum(1 for token in query_profile["date_tokens"] if token in doc_tokens)
    exact_match_count = sum(1 for token in query_profile["meaningful_tokens"] if token in doc_tokens)

    score = float(metadata.get("_qa_rank_score") or 0.0)
    score += min(0.8, exact_match_count * 0.18)
    score += min(1.1, numeric_match_count * 0.45)
    score += min(0.7, date_match_count * 0.35)
    if query_profile["has_numeric_focus"] and numeric_match_count == 0:
        score -= 0.35
    if query_profile["has_date_focus"] and date_match_count == 0:
        score -= 0.2
    if metadata.get("_qa_is_boilerplate"):
        score -= 0.6
    if metadata.get("_keyword_rank"):
        score += max(0.0, 0.18 - ((int(metadata.get("_keyword_rank") or 0) - 1) * 0.02))
    if metadata.get("_vector_rank"):
        score += max(0.0, 0.12 - ((int(metadata.get("_vector_rank") or 0) - 1) * 0.01))

    return {
        "filename": filename,
        "page": page,
        "score": score,
        "doc_rank": doc_rank,
        "has_numeric_match": numeric_match_count > 0,
        "has_date_match": date_match_count > 0,
        "doc_tokens": doc_tokens,
        "text": text,
        "query_exact_match_count": exact_match_count,
        "query_numeric_match_count": numeric_match_count,
        "query_date_match_count": date_match_count,
    }


def _prune_qa_source_candidates(answer: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answer_profile = _query_source_profile(answer)
    grounded_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        support = _answer_support_features(answer_profile, candidate)
        enriched.update(support)
        enriched["score"] = float(enriched["score"]) + float(support["answer_support_score"])
        grounded_candidates.append(enriched)

    grounded_candidates.sort(
        key=lambda item: (
            item["score"],
            item["answer_support_score"],
            item["answer_numeric_match_count"],
            item["answer_date_match_count"],
            item["answer_exact_match_count"],
            -item["doc_rank"],
        ),
        reverse=True,
    )

    if not _answer_has_grounding_signal(answer_profile):
        return grounded_candidates

    numeric_or_date_grounded = [
        item
        for item in grounded_candidates
        if item["answer_numeric_match_count"] > 0 or item["answer_date_match_count"] > 0
    ]
    if numeric_or_date_grounded:
        return numeric_or_date_grounded

    exact_grounded = [
        item
        for item in grounded_candidates
        if item["answer_support_score"] >= 0.24 or item["answer_exact_match_count"] >= 2
    ]
    return exact_grounded or grounded_candidates[:1]


def _answer_support_features(answer_profile: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    doc_tokens = set(candidate.get("doc_tokens") or set())
    exact_match_count = sum(1 for token in answer_profile["meaningful_tokens"] if token in doc_tokens)
    numeric_match_count = sum(1 for token in answer_profile["numeric_tokens"] if token in doc_tokens)
    date_match_count = sum(1 for token in answer_profile["date_tokens"] if token in doc_tokens)

    support_score = 0.0
    support_score += min(1.2, numeric_match_count * 0.52)
    support_score += min(0.8, date_match_count * 0.38)
    support_score += min(0.5, exact_match_count * 0.1)
    if answer_profile["has_numeric_focus"] and numeric_match_count == 0:
        support_score -= 0.28
    if answer_profile["has_date_focus"] and date_match_count == 0:
        support_score -= 0.18

    return {
        "answer_support_score": support_score,
        "answer_exact_match_count": exact_match_count,
        "answer_numeric_match_count": numeric_match_count,
        "answer_date_match_count": date_match_count,
    }


def _should_include_secondary_source(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    if secondary["filename"] == primary["filename"] and abs(int(secondary["page"]) - int(primary["page"])) <= 1:
        return False
    if float(secondary.get("answer_support_score") or 0.0) <= 0.0:
        return False
    if float(secondary["score"]) < float(primary["score"]) - 0.35:
        return False
    return (
        secondary.get("answer_numeric_match_count", 0) > 0
        or secondary.get("answer_date_match_count", 0) > 0
        or float(secondary.get("answer_support_score") or 0.0) >= 0.24
    )


def _query_source_profile(query: str) -> dict[str, Any]:
    tokens = tuple(tokenize_keyword_text(query or ""))
    meaningful_tokens = tuple(
        token
        for token in tokens
        if len(token) >= 2 or classify_keyword_token(token) in {"number", "mixed"}
    )
    numeric_tokens = tuple(token for token in meaningful_tokens if _is_numeric_like(token))
    date_tokens = tuple(token for token in meaningful_tokens if _looks_date_like(token))
    return {
        "meaningful_tokens": meaningful_tokens,
        "numeric_tokens": numeric_tokens,
        "date_tokens": date_tokens,
        "has_numeric_focus": bool(numeric_tokens),
        "has_date_focus": bool(date_tokens) or any(term in str(query or "") for term in ("날짜", "일자", "언제", "기준", "연도", "월", "분기")),
    }


def _document_text(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    parts = [
        str(metadata.get("retrieval_text") or "").strip(),
        str(getattr(document, "page_content", "") or "").strip(),
    ]
    return " ".join(part for part in parts if part).strip()


def _is_numeric_like(token: str) -> bool:
    return classify_keyword_token(token) in {"number", "mixed"} and bool(re.search(r"\d", token))


def _looks_date_like(token: str) -> bool:
    value = str(token or "")
    return bool(
        re.search(r"\b\d{4}[./-]\d{1,2}(?:[./-]\d{1,2})?\b", value)
        or re.search(r"\b\d{4}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?\b", value)
        or re.search(r"\b\d{1,2}q\d{2,4}\b", value, flags=re.IGNORECASE)
    )


def _answer_has_grounding_signal(answer_profile: dict[str, Any]) -> bool:
    return bool(
        answer_profile["numeric_tokens"]
        or answer_profile["date_tokens"]
        or len(answer_profile["meaningful_tokens"]) >= 2
    )


def _answer_indicates_no_support(answer: str) -> bool:
    first_sentence = _first_answer_sentence(answer)
    if not first_sentence:
        return True
    patterns = (
        r"확인할 수 없습니다",
        r"알 수 없습니다",
        r"찾을 수 없습니다",
        r"답변할 수 없습니다",
        r"응답할 수 없습니다",
        r"관련 (?:내용|정보|문장|근거)[^.\n]{0,24}(?:없|부족)",
        r"(?:내용|정보|근거)[^.\n]{0,16}없어[^.\n]{0,20}(?:답변|응답)할 수 없습니다",
        r"문맥(?:상)?[^.\n]{0,30}(?:없|부족)",
        r"제공된 문서[^.\n]{0,60}(?:포함되어 있지 않|확인할 수 없|찾을 수 없|답변할 수 없|응답할 수 없|없어)",
        r"지원[^.\n]{0,20}(?:되지 않|할 수 없)",
    )
    return any(re.search(pattern, first_sentence) for pattern in patterns)


def _first_answer_sentence(answer: str) -> str:
    cleaned = re.sub(r"[#>*`_]", " ", str(answer or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]
    return parts[0] if parts else cleaned
