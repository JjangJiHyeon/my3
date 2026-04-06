"""
Generalized document router.

Public interface:
1. `pre_route_document(doc) -> str`
2. `route_document(doc, pages_data, metadata=None) -> dict`
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

BASE_DOC_TYPES = ("slide_ir", "text_report", "dashboard_brief", "table_heavy")
FINAL_DOC_TYPES = BASE_DOC_TYPES + ("mixed_document",)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _normalize_signature(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[\W_]+", "", text)
    return text[:80]


def _extract_line_candidates(page: dict[str, Any]) -> tuple[str, str]:
    text = str(page.get("text", "") or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", ""
    return " ".join(lines[:2]), " ".join(lines[-2:])


def _extract_header_footer_signature(page: dict[str, Any]) -> tuple[str, str]:
    blocks = page.get("blocks", []) or []
    page_height = float(page.get("page_height", 0) or 0)

    if page_height > 0 and blocks:
        top_texts: list[str] = []
        bottom_texts: list[str] = []
        for block in blocks:
            text = str(block.get("text", "") or "").strip()
            bbox = block.get("bbox") or []
            if not text or len(bbox) != 4:
                continue
            y0 = float(bbox[1])
            y1 = float(bbox[3])
            if y0 <= page_height * 0.15:
                top_texts.append(text)
            if y1 >= page_height * 0.85:
                bottom_texts.append(text)

        top_sig = _normalize_signature(" ".join(top_texts[:2]))
        bottom_sig = _normalize_signature(" ".join(bottom_texts[-2:]))
        if top_sig or bottom_sig:
            return top_sig, bottom_sig

    top, bottom = _extract_line_candidates(page)
    return _normalize_signature(top), _normalize_signature(bottom)


def _estimate_column_count(blocks: list[dict[str, Any]], page_width: float, layout_hint: str) -> int:
    if not blocks:
        return 2 if layout_hint == "two_column_text" else 1

    if layout_hint == "two_column_text":
        return 2

    if layout_hint == "slide_like":
        return 1

    if page_width <= 0:
        return 1

    textish = [
        block for block in blocks
        if block.get("type") in ("text", "title")
        and len(block.get("bbox") or []) == 4
    ]
    if not textish:
        return 1

    left_edges: list[float] = []
    for block in textish:
        x0, _, x1, _ = [float(v) for v in block["bbox"]]
        width = x1 - x0
        if width <= page_width * 0.72:
            left_edges.append(x0)

    if not left_edges:
        return 1

    left_edges.sort()
    clusters = [left_edges[0]]
    threshold = max(page_width * 0.18, 40.0)
    for edge in left_edges[1:]:
        if abs(edge - clusters[-1]) > threshold:
            clusters.append(edge)
        else:
            clusters[-1] = (clusters[-1] + edge) / 2.0

    return max(1, min(3, len(clusters)))


def _page_feature_bundle(
    page: dict[str, Any],
    repeated_signature_pages: set[int],
    page_index: int,
) -> dict[str, Any]:
    text = str(page.get("text", "") or "")
    compact_text = re.sub(r"\s+", "", text)
    blocks = page.get("blocks", []) or []
    debug = page.get("parser_debug", {}) or {}
    block_type_counts = debug.get("block_type_counts", {}) or {}
    candidate_counts = debug.get("candidate_counts", {}) or {}
    page_type = debug.get("page_type", "unknown")
    layout_hint = debug.get("page_layout_hint", "unknown")

    page_width = float(page.get("page_width", 0) or 0)
    page_height = float(page.get("page_height", 0) or 0)
    block_count = int(candidate_counts.get("final_blocks", len(blocks)))
    image_blocks = sum(1 for block in blocks if block.get("type") in ("image", "chart", "drawing"))
    table_blocks = sum(1 for block in blocks if block.get("type") == "table")
    title_blocks = [block for block in blocks if block.get("type") == "title"]
    text_blocks = [block for block in blocks if block.get("type") in ("text", "title", "footer")]

    title_chars = sum(len(str(block.get("text", "") or "").strip()) for block in title_blocks)
    numeric_chars = len(re.findall(r"\d", compact_text)) + text.count("%")
    image_or_drawing_count = max(int(page.get("image_count", 0) or 0), image_blocks)

    table_candidate_count = len(page.get("tables", []) or []) + table_blocks
    for block in blocks:
        meta = block.get("meta", {}) or {}
        if float(meta.get("table_candidate_score", 0.0) or 0.0) >= 0.45:
            table_candidate_count += 1

    if page_type == "appendix_or_table_heavy" and table_candidate_count == 0:
        table_candidate_count = 1

    title_ratio = _safe_div(title_chars, max(1, len(compact_text)))
    numeric_density = _safe_div(numeric_chars, max(1, len(compact_text)))
    estimated_column_count = _estimate_column_count(blocks, page_width, layout_hint)

    return {
        "page_type_hint": page_type,
        "layout_hint": layout_hint,
        "text_density": float(len(compact_text)),
        "numeric_density": numeric_density,
        "table_candidate_count": float(table_candidate_count),
        "image_or_drawing_count": float(image_or_drawing_count),
        "estimated_column_count": float(estimated_column_count),
        "title_ratio": title_ratio,
        "repeated_header_footer_ratio": 1.0 if page_index in repeated_signature_pages else 0.0,
        "block_count": float(block_count),
        "landscape": 1.0 if page_width > 0 and page_height > 0 and page_width > page_height * 1.05 else 0.0,
        "visual_block_count": float(image_blocks + table_blocks),
        "text_block_count": float(len(text_blocks)),
        "block_type_counts": block_type_counts,
    }


def _score_page(features: dict[str, Any]) -> dict[str, float]:
    text_norm = _clamp(features["text_density"] / 1800.0)
    numeric_norm = _clamp(features["numeric_density"] / 0.18)
    table_norm = _clamp(features["table_candidate_count"] / 3.0)
    image_norm = _clamp(features["image_or_drawing_count"] / 3.0)
    column_norm = _clamp((features["estimated_column_count"] - 1.0) / 2.0)
    title_norm = _clamp(features["title_ratio"] / 0.18)
    repeated_norm = _clamp(features["repeated_header_footer_ratio"])
    block_norm = _clamp(features["block_count"] / 18.0)
    landscape_norm = _clamp(features["landscape"])

    slide_hint = 1.0 if features["layout_hint"] == "slide_like" else 0.0
    dashboard_hint = 1.0 if features["page_type_hint"] == "dashboard_kpi_like" else 0.0
    table_hint = 1.0 if features["page_type_hint"] == "appendix_or_table_heavy" else 0.0
    report_fit = 1.0 - _clamp(abs(features["estimated_column_count"] - 1.5) / 1.5)

    return {
        "slide_ir": (
            0.24 * image_norm
            + 0.18 * title_norm
            + 0.16 * (1.0 - text_norm)
            + 0.14 * (1.0 - column_norm)
            + 0.12 * slide_hint
            + 0.08 * landscape_norm
            + 0.08 * (1.0 - repeated_norm)
        ),
        "text_report": (
            0.32 * text_norm
            + 0.18 * (1.0 - table_norm)
            + 0.14 * (1.0 - image_norm)
            + 0.14 * repeated_norm
            + 0.12 * report_fit
            + 0.10 * (1.0 - numeric_norm)
        ),
        "dashboard_brief": (
            0.24 * numeric_norm
            + 0.22 * image_norm
            + 0.18 * block_norm
            + 0.14 * column_norm
            + 0.12 * dashboard_hint
            + 0.10 * (1.0 - repeated_norm)
        ),
        "table_heavy": (
            0.36 * table_norm
            + 0.22 * numeric_norm
            + 0.14 * repeated_norm
            + 0.12 * table_hint
            + 0.08 * (1.0 - image_norm)
            + 0.08 * (1.0 - title_norm)
        ),
    }


def _select_page_type(page_scores: dict[str, float], features: dict[str, Any]) -> str:
    ordered = sorted(page_scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    score_gap = best_score - second_score

    feature_mixture = sum(
        1
        for value in (
            _clamp(features["text_density"] / 1800.0),
            _clamp(features["numeric_density"] / 0.18),
            _clamp(features["table_candidate_count"] / 3.0),
            _clamp(features["image_or_drawing_count"] / 3.0),
        )
        if value >= 0.45
    )

    if score_gap < 0.08 and feature_mixture >= 2:
        return "mixed_document"
    return best_type


def _summarize_page_repetition(pages_data: list[dict[str, Any]]) -> tuple[set[int], float]:
    signatures: list[tuple[int, str]] = []
    for idx, page in enumerate(pages_data):
        top_sig, bottom_sig = _extract_header_footer_signature(page)
        if len(top_sig) >= 8:
            signatures.append((idx, f"top:{top_sig}"))
        if len(bottom_sig) >= 8:
            signatures.append((idx, f"bottom:{bottom_sig}"))

    repeated_values = {sig for sig, count in Counter(sig for _, sig in signatures).items() if count >= 2}
    repeated_pages = {idx for idx, sig in signatures if sig in repeated_values}
    return repeated_pages, _safe_div(len(repeated_pages), len(pages_data))


def _build_routing_reasons(
    doc_type: str,
    aggregate_features: dict[str, float],
    page_type_distribution_ratio: dict[str, float],
    routing_scores: dict[str, float],
) -> list[str]:
    reasons: list[str] = []

    if doc_type == "slide_ir":
        reasons.append(
            "Visual-heavy pages dominate with low narrative density and strong single-column/title signals."
        )
        if aggregate_features["landscape_ratio"] >= 0.4:
            reasons.append(f"Landscape-like page ratio is {aggregate_features['landscape_ratio']:.2f}.")
    elif doc_type == "text_report":
        reasons.append(
            "High text density with lower table/visual pressure indicates a narrative report structure."
        )
        if aggregate_features["repeated_header_footer_ratio"] >= 0.35:
            reasons.append(
                f"Repeated header/footer ratio is {aggregate_features['repeated_header_footer_ratio']:.2f}, which is common in paginated reports."
            )
    elif doc_type == "dashboard_brief":
        reasons.append(
            "Numeric density, visual density, and dense block layout together indicate KPI/dashboard-style pages."
        )
        if aggregate_features["estimated_column_count"] >= 1.8:
            reasons.append(f"Estimated column count is {aggregate_features['estimated_column_count']:.2f}.")
    elif doc_type == "table_heavy":
        reasons.append(
            "Table candidates and numeric concentration consistently outweigh narrative text features."
        )
        if aggregate_features["table_candidate_count"] >= 1.2:
            reasons.append(f"Average table candidate count is {aggregate_features['table_candidate_count']:.2f} per page.")
    else:
        reasons.append(
            "No single layout dominates; text, numeric, table, and visual signals coexist across pages."
        )
        reasons.append(
            f"Top routing scores are close ({max(routing_scores.values()):.2f} max), so a mixed document label is more appropriate."
        )

    dominant_page_type = max(page_type_distribution_ratio.items(), key=lambda item: item[1])[0]
    dominant_ratio = page_type_distribution_ratio[dominant_page_type]
    reasons.append(f"Page-type distribution is led by `{dominant_page_type}` pages at {dominant_ratio:.2f}.")

    return reasons[:4]


def _route_from_pages(
    pages_data: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    allow_mixed: bool = True,
) -> dict[str, Any]:
    page_count = len(pages_data)
    if page_count == 0:
        empty_distribution = {doc_type: 0 for doc_type in FINAL_DOC_TYPES}
        return {
            "document_type": "unknown",
            "doc_type": "unknown",
            "confidence": 0.0,
            "page_type_distribution": empty_distribution,
            "routing_reasons": ["No pages or content found in document."],
            "routing_signals": {
                "page_count": 0,
                "text_density": 0.0,
                "numeric_density": 0.0,
                "table_candidate_count": 0.0,
                "image_or_drawing_count": 0.0,
                "estimated_column_count": 0.0,
                "title_ratio": 0.0,
                "repeated_header_footer_ratio": 0.0,
                "page_type_distribution": empty_distribution,
            },
            "routing_reason": "No pages or content found in document.",
        }

    repeated_pages, repeated_ratio = _summarize_page_repetition(pages_data)
    page_summaries: list[dict[str, Any]] = []
    parser_page_types: list[str] = []
    layout_hints: list[str] = []

    aggregate = {
        "text_density": 0.0,
        "numeric_density": 0.0,
        "table_candidate_count": 0.0,
        "image_or_drawing_count": 0.0,
        "estimated_column_count": 0.0,
        "title_ratio": 0.0,
        "block_count": 0.0,
        "landscape_ratio": 0.0,
    }

    for idx, page in enumerate(pages_data):
        features = _page_feature_bundle(page, repeated_pages, idx)
        page_scores = _score_page(features)
        predicted_type = _select_page_type(page_scores, features)

        page_summaries.append({
            "features": features,
            "scores": page_scores,
            "predicted_type": predicted_type,
        })

        parser_page_types.append(features["page_type_hint"])
        layout_hints.append(features["layout_hint"])

        for key in aggregate:
            if key in features:
                aggregate[key] += float(features[key])

    aggregate_features = {
        "text_density": aggregate["text_density"] / page_count,
        "numeric_density": aggregate["numeric_density"] / page_count,
        "table_candidate_count": aggregate["table_candidate_count"] / page_count,
        "image_or_drawing_count": aggregate["image_or_drawing_count"] / page_count,
        "estimated_column_count": aggregate["estimated_column_count"] / page_count,
        "title_ratio": aggregate["title_ratio"] / page_count,
        "repeated_header_footer_ratio": repeated_ratio,
        "avg_block_count": aggregate["block_count"] / page_count,
        "landscape_ratio": aggregate["landscape_ratio"] / page_count,
    }

    page_type_counts = Counter(summary["predicted_type"] for summary in page_summaries)
    page_type_distribution = {doc_type: page_type_counts.get(doc_type, 0) for doc_type in FINAL_DOC_TYPES}
    page_type_distribution_ratio = {
        doc_type: _safe_div(count, page_count) for doc_type, count in page_type_distribution.items()
    }

    routing_scores = {}
    for doc_type in BASE_DOC_TYPES:
        mean_score = sum(summary["scores"][doc_type] for summary in page_summaries) / page_count
        routing_scores[doc_type] = mean_score + 0.15 * page_type_distribution_ratio.get(doc_type, 0.0)

    ranked_scores = sorted(routing_scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked_scores[0]
    second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
    dominant_ratio = page_type_distribution_ratio.get(best_type, 0.0)
    multi_type_coverage = sum(1 for doc_type in BASE_DOC_TYPES if page_type_distribution_ratio.get(doc_type, 0.0) >= 0.2)

    should_use_mixed = allow_mixed and (
        page_type_distribution_ratio.get("mixed_document", 0.0) >= 0.25
        or (best_score - second_score) < 0.08 and multi_type_coverage >= 2
        or dominant_ratio < 0.5 and multi_type_coverage >= 3
    )

    doc_type = "mixed_document" if should_use_mixed else best_type
    confidence = (
        _clamp(0.52 + page_type_distribution_ratio.get("mixed_document", 0.0) * 0.25 + multi_type_coverage * 0.05)
        if doc_type == "mixed_document"
        else _clamp(0.40 + best_score * 0.35 + dominant_ratio * 0.20 + (best_score - second_score) * 1.20)
    )

    routing_reasons = _build_routing_reasons(
        doc_type=doc_type,
        aggregate_features=aggregate_features,
        page_type_distribution_ratio=page_type_distribution_ratio,
        routing_scores=routing_scores,
    )

    parser_page_type_distribution = {page_type: parser_page_types.count(page_type) for page_type in set(parser_page_types)}
    layout_hint_distribution = {layout_hint: layout_hints.count(layout_hint) for layout_hint in set(layout_hints)}
    narrative = metadata.get("narrative_analysis", {}) if metadata else {}

    signals = {
        "page_count": page_count,
        "text_density": round(aggregate_features["text_density"], 1),
        "numeric_density": round(aggregate_features["numeric_density"], 3),
        "table_candidate_count": round(aggregate_features["table_candidate_count"], 2),
        "image_or_drawing_count": round(aggregate_features["image_or_drawing_count"], 2),
        "estimated_column_count": round(aggregate_features["estimated_column_count"], 2),
        "title_ratio": round(aggregate_features["title_ratio"], 3),
        "repeated_header_footer_ratio": round(aggregate_features["repeated_header_footer_ratio"], 3),
        "avg_block_count": round(aggregate_features["avg_block_count"], 1),
        "landscape_ratio": round(aggregate_features["landscape_ratio"], 2),
        "page_type_distribution": page_type_distribution,
        "page_type_distribution_ratio": {key: round(value, 3) for key, value in page_type_distribution_ratio.items()},
        "parser_page_type_distribution": parser_page_type_distribution,
        "layout_hint_distribution": layout_hint_distribution,
        "routing_scores": {key: round(value, 3) for key, value in routing_scores.items()},
        "confidence": round(confidence, 3),
        "is_truly_dashboard_signal": bool(narrative.get("is_truly_dashboard", False)),
    }

    return {
        "document_type": doc_type,
        "doc_type": doc_type,
        "confidence": round(confidence, 3),
        "page_type_distribution": page_type_distribution,
        "routing_reasons": routing_reasons,
        "routing_signals": signals,
        "routing_reason": " ".join(routing_reasons[:2]),
    }


def pre_route_document(doc: fitz.Document) -> str:
    """
    Fast preliminary routing for pipeline dispatch.
    Keeps the legacy string interface and returns only base pipeline types.
    """
    page_count = doc.page_count
    if page_count == 0:
        return "unknown"

    sample_pages: list[dict[str, Any]] = []
    sample_count = min(page_count, 5)

    for page_idx in range(sample_count):
        try:
            page = doc[page_idx]
            text = page.get_text("text") or ""
            text_dict = page.get_text("dict") or {}
            blocks: list[dict[str, Any]] = []
            image_count = 0

            for block in text_dict.get("blocks", []):
                bbox = block.get("bbox") or []
                if block.get("type") == 1:
                    image_count += 1
                    blocks.append({"type": "image", "bbox": bbox, "text": ""})
                    continue

                block_text_parts: list[str] = []
                max_size = 0.0
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = str(span.get("text", "") or "")
                        if span_text.strip():
                            block_text_parts.append(span_text.strip())
                            max_size = max(max_size, float(span.get("size", 0.0) or 0.0))

                block_text = " ".join(block_text_parts).strip()
                if not block_text:
                    continue

                block_type = "title" if max_size >= 14.0 and len(block_text) <= 80 else "text"
                blocks.append({"type": block_type, "bbox": bbox, "text": block_text})

            sample_pages.append({
                "text": text,
                "blocks": blocks,
                "tables": [],
                "image_count": image_count,
                "page_width": float(page.rect.width),
                "page_height": float(page.rect.height),
                "parser_debug": {},
            })
        except Exception as exc:
            logger.debug("Pre-route sampling failed on page %s: %s", page_idx + 1, exc)

    if not sample_pages:
        return "text_report"

    routed = _route_from_pages(sample_pages, metadata=doc.metadata or {}, allow_mixed=False)
    pre_doc_type = routed.get("document_type", "text_report")
    return pre_doc_type if pre_doc_type in BASE_DOC_TYPES else "text_report"


def route_document(
    doc: fitz.Document | None,
    pages_data: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Analyze page-level features and infer a generalized document route.
    Keeps legacy keys while exposing richer routing metadata.
    """
    return _route_from_pages(pages_data, metadata=metadata, allow_mixed=True)
