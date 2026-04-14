from __future__ import annotations

import re
from typing import Any

from .schema import empty_chunk
from .utils import (
    TABLE_TYPES,
    TEXT_TYPES,
    VISUAL_TYPES,
    associated_title,
    block_id,
    block_meta,
    block_order,
    clean_text,
    compact_join,
    is_excluded_block,
    page_num,
    panel_key,
    split_long_text,
    stable_id,
    summary_priority,
    support_text_from_block,
    type_counts,
)
from .visual_structured import build_visual_structured_records

MAIN_STRATEGY = "text_first_with_visual_support"
BASELINE_STRATEGY = "llm_ready_native"
SENTENCE_END_RE = re.compile(r"[.!?]\s|[.!?]$|[다요음]\s|[다요음]$")
PHONEISH_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
EMAILISH_RE = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
URLISH_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)


def _doc_fields(doc: dict[str, Any], doc_id: str) -> dict[str, Any]:
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return {
        "doc_id": doc_id,
        "filename": str(doc.get("filename") or ""),
        "filepath": doc.get("filepath"),
        "document_type": meta.get("refined_document_type") or meta.get("document_type") or doc.get("document_type"),
        "pipeline_used": meta.get("pipeline_used") or doc.get("pipeline_used"),
        "quality_grade": meta.get("quality_grade") or doc.get("quality_grade"),
        "routing": {
            "routing_confidence": meta.get("routing_confidence"),
            "routing_reason": meta.get("routing_reason"),
            "routing_reasons": meta.get("routing_reasons"),
            "routing_signals": meta.get("routing_signals"),
            "page_type_distribution": meta.get("page_type_distribution"),
        },
    }


def _chunk_id(doc_id: str, strategy: str, page: int | None, index: int, kind: str, source_ids: list[str]) -> str:
    return f"{doc_id}:{strategy}:p{page or 0}:{index:04d}:{kind}:{stable_id(source_ids, length=8)}"


def _make_chunk(
    *,
    strategy: str,
    doc_fields: dict[str, Any],
    page: dict[str, Any],
    chunk_index: int,
    chunk_type: str,
    retrieval_text: str,
    display_text: str | None,
    source_block_ids: list[str],
    block_group: list[dict[str, Any]],
    support_blocks: list[dict[str, Any]] | None = None,
    section_title: str | None = None,
    associated: str | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    support_blocks = support_blocks or []
    blocks = block_group + support_blocks
    counts = type_counts(blocks)
    pnum = page_num(page)
    parser_debug = page.get("parser_debug") if isinstance(page.get("parser_debug"), dict) else {}
    page_title = clean_text(page.get("page_title"))
    md = {
        "parser_debug": parser_debug,
        "preview_image": page.get("preview_image"),
        "page_width": page.get("page_width"),
        "page_height": page.get("page_height"),
        "routing": doc_fields.get("routing"),
        "block_type_counts": parser_debug.get("block_type_counts") or page.get("block_type_counts"),
    }
    if metadata_extra:
        md.update(metadata_extra)
    return empty_chunk(
        chunk_id=_chunk_id(doc_fields["doc_id"], strategy, pnum, chunk_index, chunk_type, source_block_ids),
        strategy_name=strategy,
        doc_id=doc_fields["doc_id"],
        filename=doc_fields["filename"],
        filepath=doc_fields["filepath"],
        document_type=doc_fields["document_type"],
        pipeline_used=doc_fields["pipeline_used"],
        quality_grade=doc_fields["quality_grade"],
        page_num=pnum,
        chunk_index=chunk_index,
        chunk_type=chunk_type,
        section_title=section_title or page_title or associated_title(block_group, page_title),
        page_title=page_title,
        associated_title=associated or associated_title(blocks, page_title),
        source_block_ids=source_block_ids,
        chosen_text_source=parser_debug.get("chosen_text_source") or page.get("rag_text_source"),
        summary_priority=summary_priority(blocks),
        has_table=counts.get("table", 0) > 0,
        has_chart=counts.get("chart", 0) > 0,
        has_image=counts.get("image", 0) > 0,
        table_count=counts.get("table", 0),
        image_count=counts.get("image", 0) + counts.get("chart", 0),
        retrieval_text=retrieval_text,
        display_text=display_text,
        metadata=md,
    )


def _visual_structured_chunk_fields(structured: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(structured, dict):
        return {}
    fields: dict[str, Any] = {}
    text_fields = {
        "table_kind": structured.get("table_kind"),
        "table_title": structured.get("title"),
        "row_label": structured.get("row_label"),
        "column_label": structured.get("column_label"),
        "cell_value": structured.get("value"),
    }
    for key, value in text_fields.items():
        clean = clean_text(value)
        if clean:
            fields[key] = clean

    int_fields = {
        "row_index_in_table": structured.get("row_index"),
        "column_index_in_table": structured.get("column_index"),
    }
    for key, value in int_fields.items():
        try:
            if value is not None and value != "":
                fields[key] = int(value)
        except (TypeError, ValueError):
            continue
    return fields


def _visual_support_for_group(text_blocks: list[dict[str, Any]], support_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = {block_id(block) for block in text_blocks}
    panels = {panel_key(block) for block in text_blocks if panel_key(block)}
    anchor_text = compact_join(block.get("text", "") for block in text_blocks)
    selected: list[dict[str, Any]] = []
    for block in support_blocks:
        meta = block_meta(block)
        context_ids = {str(x) for x in (meta.get("context_block_ids") or [])}
        same_panel = panel_key(block) and panel_key(block) in panels
        references_group = bool(context_ids & ids)
        if not (same_panel or references_group):
            continue
        support_score = _support_block_score(block)
        required_score = 4 if references_group else 6
        if len(anchor_text) < 120:
            required_score += 1
        if support_score >= required_score:
            selected.append(block)
    return selected


def _is_sparse_page(page: dict[str, Any], blocks: list[dict[str, Any]]) -> bool:
    rag = clean_text(page.get("rag_text"))
    text_blocks = [b for b in blocks if b.get("type") in TEXT_TYPES and not is_excluded_block(b)]
    visual_blocks = [b for b in blocks if b.get("type") in VISUAL_TYPES and not is_excluded_block(b, for_support=True)]
    numericish = sum(1 for b in text_blocks if any(ch.isdigit() for ch in clean_text(b.get("text"))))
    return len(rag) < 80 or (len(text_blocks) <= 2 and len(visual_blocks) >= 1) or (numericish >= 1 and len(text_blocks) <= 4 and len(visual_blocks) >= 1)


def _fallback_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    parts = []
    for block in sorted(blocks, key=block_order):
        if is_excluded_block(block):
            continue
        if block.get("type") in TEXT_TYPES | TABLE_TYPES:
            text = clean_text(block.get("text"))
            if text:
                parts.append(text)
        for support in support_text_from_block(block):
            parts.append(support)
    return compact_join(parts)


def _sentence_like_count(text: str) -> int:
    clean = clean_text(text)
    if not clean:
        return 0
    lines = [line for line in clean.splitlines() if clean_text(line)]
    count = len(SENTENCE_END_RE.findall(clean))
    if count:
        return count
    return sum(1 for line in lines if len(line) >= 45)


def _metric_like_count(text: str) -> int:
    return len(re.findall(r"\d[\d,]*(?:\.\d+)?(?:\s*[%]|[A-Za-z]{1,4})?", text or ""))


def _looks_contactish(text: str) -> bool:
    clean = clean_text(text)
    if len(clean) > 220:
        return False
    colon_lines = sum(1 for line in clean.splitlines() if ":" in line or "|" in line)
    contact_signals = len(PHONEISH_RE.findall(clean)) + len(EMAILISH_RE.findall(clean)) + len(URLISH_RE.findall(clean))
    return contact_signals >= 1 and colon_lines >= 1 and _sentence_like_count(clean) == 0 and _metric_like_count(clean) <= 2


def _support_parts_for_block(block: dict[str, Any]) -> list[str]:
    meta = block_meta(block)
    parts: list[str] = []
    candidate_keys = (
        "table_summary",
        "chart_summary",
        "visual_summary",
        "caption_text",
        "table_markdown",
    )
    for key in candidate_keys:
        value = clean_text(meta.get(key) or block.get(key))
        if not value:
            continue
        if key == "table_markdown" and len(value) < 90:
            continue
        parts.append(value)
    rows = meta.get("key_value_rows") or block.get("key_value_rows")
    if isinstance(rows, list) and len(rows) >= 3:
        rendered_rows: list[str] = []
        for row in rows[:8]:
            if isinstance(row, dict):
                item = clean_text(row.get("item"))
                values = clean_text(row.get("values"))
                if item or values:
                    rendered_rows.append(f"{item}: {values}".strip(": "))
            elif isinstance(row, (list, tuple)):
                row_text = " | ".join(clean_text(x) for x in row if clean_text(x))
                if row_text:
                    rendered_rows.append(row_text)
        if rendered_rows:
            parts.append("\n".join(rendered_rows))
    return parts


def _support_block_score(block: dict[str, Any]) -> int:
    meta = block_meta(block)
    text = compact_join(_support_parts_for_block(block))
    if not text:
        return 0
    score = 0
    text_len = len(text)
    sentences = _sentence_like_count(text)
    metrics = _metric_like_count(text)
    rows = meta.get("key_value_rows") or block.get("key_value_rows") or []
    if text_len >= 120:
        score += 2
    elif text_len >= 70:
        score += 1
    if sentences >= 2:
        score += 2
    elif sentences == 1:
        score += 1
    if metrics >= 4:
        score += 2
    elif metrics >= 2:
        score += 1
    if block.get("type") == "table":
        if isinstance(rows, list) and len(rows) >= 4:
            score += 2
        elif isinstance(rows, list) and len(rows) >= 2:
            score += 1
        markdown = clean_text(meta.get("table_markdown"))
        if markdown.count("\n") >= 3 or markdown.count("|") >= 8:
            score += 1
    else:
        if clean_text(meta.get("chart_summary") or meta.get("visual_summary") or meta.get("caption_text")):
            score += 1
    if _looks_contactish(text):
        score -= 4
    if text_len < 45:
        score -= 2
    return score


def _compose_group_text(text: str, support: list[dict[str, Any]]) -> tuple[str, str]:
    anchor = clean_text(text)
    support_candidates = sorted(
        ((block, _support_block_score(block), compact_join(_support_parts_for_block(block))) for block in support),
        key=lambda item: (item[1], len(item[2])),
        reverse=True,
    )
    support_texts: list[str] = []
    for _, score, candidate_text in support_candidates[:2]:
        if not candidate_text:
            continue
        threshold = 5 if len(anchor) < 120 else 4
        if score < threshold:
            continue
        support_texts.append(candidate_text)
    display = anchor
    if support_texts:
        display = compact_join([anchor, *support_texts]) if anchor else compact_join(support_texts)
    retrieval_support = support_texts
    if len(anchor) < 60:
        retrieval_support = support_texts[:1]
    retrieval = compact_join([anchor, *retrieval_support]) if anchor else compact_join(retrieval_support)
    return retrieval, display


def _ordered_text_anchor_groups(text_blocks: list[dict[str, Any]], *, target_chars: int = 950) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    group_by_key: dict[str, list[dict[str, Any]]] = {}
    flow: list[dict[str, Any]] = []

    def flush_flow() -> None:
        nonlocal flow
        if flow:
            groups.append((f"text_flow:{len(groups) + 1}", flow))
            flow = []

    for block in sorted(text_blocks, key=block_order):
        key = panel_key(block)
        text = clean_text(block.get("text"))
        if key:
            flush_flow()
            if key not in group_by_key:
                group_by_key[key] = []
                groups.append((key, group_by_key[key]))
            group_by_key[key].append(block)
            continue

        current_len = len(compact_join(b.get("text", "") for b in flow))
        if flow and current_len + len(text) > target_chars:
            flush_flow()
        flow.append(block)
    flush_flow()
    return [(key, group) for key, group in groups if group]


def build_text_first_chunks(doc: dict[str, Any], llm_ready: dict[str, Any] | None, doc_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    doc_fields = _doc_fields(doc, doc_id)
    chunks: list[dict[str, Any]] = []
    issues: list[str] = []
    llm_pages = {page.get("page_num"): page for page in (llm_ready or {}).get("pages", []) if isinstance(page, dict)}
    chunk_index = 0

    for page in doc.get("pages") or []:
        if not isinstance(page, dict):
            continue
        pnum = page_num(page)
        blocks = [b for b in (page.get("blocks") or []) if isinstance(b, dict)]
        text_blocks = [b for b in blocks if b.get("type") in TEXT_TYPES and not is_excluded_block(b)]
        support_blocks = [
            b for b in blocks
            if b.get("type") in TABLE_TYPES | VISUAL_TYPES and not is_excluded_block(b, for_support=True) and support_text_from_block(b)
        ]
        sparse = _is_sparse_page(page, blocks)

        emitted = False
        for key, group in _ordered_text_anchor_groups(text_blocks):
            text = compact_join([b.get("text", "") for b in group])
            if len(text) < 24 and len(group) == 1 and not _visual_support_for_group(group, support_blocks):
                continue
            support = _visual_support_for_group(group, support_blocks)
            retrieval, display = _compose_group_text(text, support)
            if not retrieval:
                continue
            parts = split_long_text(retrieval)
            for part_no, part in enumerate(parts):
                chunk_index += 1
                chunks.append(_make_chunk(
                    strategy=MAIN_STRATEGY,
                    doc_fields=doc_fields,
                    page=page,
                    chunk_index=chunk_index,
                    chunk_type="block_group" if len(group) > 1 or support else "text",
                    retrieval_text=part,
                    display_text=display if part_no == 0 else part,
                    source_block_ids=[block_id(b) for b in group + support],
                    block_group=group,
                    support_blocks=support,
                    metadata_extra={"sparse_page": sparse, "group_key": key, "split_index": part_no},
                ))
                emitted = True

        if not emitted:
            fallback = clean_text(page.get("rag_text"))
            if not fallback:
                llm_page = llm_pages.get(pnum) or {}
                fallback = clean_text(llm_page.get("rag_text"))
            if not fallback:
                fallback = _fallback_text_from_blocks(blocks)
            if fallback:
                support = [block for block in support_blocks if _support_block_score(block) >= 5] if sparse else []
                retrieval, display = _compose_group_text(fallback, support)
                if not retrieval:
                    retrieval = fallback
                    display = fallback
                for part_no, part in enumerate(split_long_text(retrieval)):
                    chunk_index += 1
                    chunks.append(_make_chunk(
                        strategy=MAIN_STRATEGY,
                        doc_fields=doc_fields,
                        page=page,
                        chunk_index=chunk_index,
                        chunk_type="page" if not sparse else "block_group",
                        retrieval_text=part,
                        display_text=display if part_no == 0 else part,
                        source_block_ids=[block_id(b) for b in blocks],
                        block_group=text_blocks,
                        support_blocks=support,
                        metadata_extra={"sparse_page": sparse, "fallback_source": "rag_text_or_blocks", "split_index": part_no},
                    ))
                    emitted = True

        if not emitted and support_blocks:
            for support in support_blocks:
                retrieval = compact_join(_support_parts_for_block(support))
                support_score = _support_block_score(support)
                if not retrieval or support_score < 7:
                    continue
                if len(retrieval) < 120 and _metric_like_count(retrieval) < 4 and _sentence_like_count(retrieval) < 2:
                    continue
                chunk_index += 1
                chunks.append(_make_chunk(
                    strategy=MAIN_STRATEGY,
                    doc_fields=doc_fields,
                    page=page,
                    chunk_index=chunk_index,
                    chunk_type="table_summary" if support.get("type") == "table" else "chart_summary",
                    retrieval_text=retrieval,
                    display_text=retrieval,
                    source_block_ids=[block_id(support)],
                    block_group=[],
                    support_blocks=[support],
                    metadata_extra={"sparse_page": sparse, "fallback_source": "visual_support_only"},
                ))
                emitted = True

        for visual in build_visual_structured_records(page, blocks):
            retrieval = clean_text(visual.get("retrieval_text"))
            if not retrieval:
                continue
            visual_blocks = [b for b in (visual.get("blocks") or []) if isinstance(b, dict)]
            visual_structured = {}
            visual_metadata = visual.get("metadata") if isinstance(visual.get("metadata"), dict) else {}
            if isinstance(visual_metadata.get("visual_structured"), dict):
                visual_structured = visual_metadata["visual_structured"]
            for part_no, part in enumerate(split_long_text(retrieval, target=1100, hard=1450, overlap=120)):
                chunk_index += 1
                chunk = _make_chunk(
                    strategy=MAIN_STRATEGY,
                    doc_fields=doc_fields,
                    page=page,
                    chunk_index=chunk_index,
                    chunk_type=str(visual.get("chunk_type") or "visual_summary"),
                    retrieval_text=part,
                    display_text=retrieval if part_no == 0 else part,
                    source_block_ids=[str(item) for item in (visual.get("source_block_ids") or [])],
                    block_group=visual_blocks,
                    support_blocks=[],
                    metadata_extra={
                        **(visual.get("metadata") if isinstance(visual.get("metadata"), dict) else {}),
                        "sparse_page": sparse,
                        "fallback_source": "visual_structured_extraction",
                        "split_index": part_no,
                    },
                )
                chunk.update({
                    "visual_type": str(visual.get("visual_type") or ""),
                    "visual_index": int(visual.get("visual_index") or 0),
                    "visual_confidence": float(visual.get("visual_confidence") or 0.0),
                    "source_bbox": str(visual.get("source_bbox") or ""),
                    "extraction_method": str(visual.get("extraction_method") or ""),
                })
                chunk.update(_visual_structured_chunk_fields(visual_structured))
                if chunk["visual_type"] == "table":
                    chunk["has_table"] = True
                    chunk["table_count"] = max(1, int(chunk.get("table_count") or 0))
                elif chunk["visual_type"]:
                    chunk["has_chart"] = True
                chunks.append(chunk)
                emitted = True

        if sparse:
            issues.append(f"sparse_page:{pnum}")
        if not emitted:
            issues.append(f"empty_page_no_chunk:{pnum}")

    return chunks, issues


def _baseline_doc_fields(llm_ready: dict[str, Any], parsed_doc: dict[str, Any], doc_id: str) -> dict[str, Any]:
    parsed_fields = _doc_fields(parsed_doc, doc_id)
    parsed_fields.update({
        "filename": str(llm_ready.get("filename") or parsed_fields["filename"]),
        "document_type": llm_ready.get("refined_document_type") or llm_ready.get("document_type") or parsed_fields["document_type"],
        "pipeline_used": llm_ready.get("pipeline_used") or parsed_fields["pipeline_used"],
        "quality_grade": llm_ready.get("quality_grade") or parsed_fields["quality_grade"],
    })
    return parsed_fields


def build_llm_ready_native_chunks(parsed_doc: dict[str, Any], llm_ready: dict[str, Any] | None, doc_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    if not llm_ready:
        return [], ["missing_llm_ready"]
    doc_fields = _baseline_doc_fields(llm_ready, parsed_doc, doc_id)
    parsed_pages = {page_num(page): page for page in parsed_doc.get("pages", []) if isinstance(page, dict)}
    chunks: list[dict[str, Any]] = []
    issues: list[str] = []
    chunk_index = 0
    for page in llm_ready.get("pages") or []:
        if not isinstance(page, dict):
            continue
        pnum = page.get("page_num")
        source_page = parsed_pages.get(pnum, {"page_num": pnum, "page_title": page.get("page_title")})
        blocks = [b for b in (page.get("blocks") or []) if isinstance(b, dict)]
        page_text = clean_text(page.get("rag_text"))
        if page_text:
            for part_no, part in enumerate(split_long_text(page_text, target=1100, hard=1350, overlap=120)):
                chunk_index += 1
                chunks.append(_make_chunk(
                    strategy=BASELINE_STRATEGY,
                    doc_fields=doc_fields,
                    page=source_page,
                    chunk_index=chunk_index,
                    chunk_type="native_page",
                    retrieval_text=part,
                    display_text=part,
                    source_block_ids=[str(b.get("source_order", i)) for i, b in enumerate(blocks)],
                    block_group=[],
                    metadata_extra={"llm_ready_page": page, "split_index": part_no},
                ))
        else:
            issues.append(f"empty_llm_ready_page:{pnum}")

        for i, block in enumerate(blocks):
            text = compact_join([
                block.get("text", ""),
                block.get("table_summary", ""),
                block.get("table_markdown", ""),
                block.get("chart_summary", ""),
                block.get("visual_summary", ""),
                block.get("caption_text", ""),
            ])
            if not text:
                continue
            for part_no, part in enumerate(split_long_text(text, target=1100, hard=1350, overlap=120)):
                chunk_index += 1
                pseudo_block = {
                    "id": str(block.get("source_order", i)),
                    "type": block.get("type"),
                    "text": text,
                    "meta": {k: v for k, v in block.items() if k not in {"text", "type"}},
                }
                chunks.append(_make_chunk(
                    strategy=BASELINE_STRATEGY,
                    doc_fields=doc_fields,
                    page=source_page,
                    chunk_index=chunk_index,
                    chunk_type=str(block.get("type") or "text"),
                    retrieval_text=part,
                    display_text=part,
                    source_block_ids=[str(block.get("source_order", i))],
                    block_group=[pseudo_block],
                    metadata_extra={"llm_ready_block": block, "split_index": part_no},
                ))
    return chunks, issues
