from __future__ import annotations

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

MAIN_STRATEGY = "text_first_with_visual_support"
BASELINE_STRATEGY = "llm_ready_native"


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


def _visual_support_for_group(text_blocks: list[dict[str, Any]], support_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = {block_id(block) for block in text_blocks}
    panels = {panel_key(block) for block in text_blocks if panel_key(block)}
    selected: list[dict[str, Any]] = []
    for block in support_blocks:
        meta = block_meta(block)
        context_ids = {str(x) for x in (meta.get("context_block_ids") or [])}
        same_panel = panel_key(block) and panel_key(block) in panels
        references_group = bool(context_ids & ids)
        if same_panel or references_group:
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
            support_text = compact_join(part for block in support for part in support_text_from_block(block))
            retrieval = compact_join([text, support_text])
            if not retrieval:
                continue
            display = compact_join([text, support_text])
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
                support = support_blocks if sparse else []
                support_text = compact_join(part for block in support for part in support_text_from_block(block))
                retrieval = compact_join([fallback, support_text])
                for part_no, part in enumerate(split_long_text(retrieval)):
                    chunk_index += 1
                    chunks.append(_make_chunk(
                        strategy=MAIN_STRATEGY,
                        doc_fields=doc_fields,
                        page=page,
                        chunk_index=chunk_index,
                        chunk_type="page" if not sparse else "block_group",
                        retrieval_text=part,
                        display_text=retrieval if part_no == 0 else part,
                        source_block_ids=[block_id(b) for b in blocks],
                        block_group=text_blocks,
                        support_blocks=support,
                        metadata_extra={"sparse_page": sparse, "fallback_source": "rag_text_or_blocks", "split_index": part_no},
                    ))
                    emitted = True

        if not emitted and support_blocks:
            for support in support_blocks:
                retrieval = compact_join(support_text_from_block(support))
                if not retrieval:
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
