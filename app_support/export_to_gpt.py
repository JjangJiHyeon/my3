"""
Export parsed document results as structured files for LLM/RAG consumption.

Usage:
  # Ad-hoc mode: parse file(s) and export
  python -m app_support.export_to_gpt documents/sample.pdf documents/sample.hwp

  # By doc_id: export already-parsed result from parsed_results/
  python -m app_support.export_to_gpt --id 0e989092c3d389b5e7727c1a077e66c5

  # All documents in documents/ folder
  python -m app_support.export_to_gpt --all

  # One export run from existing parsed_results/*.json (no re-parse)
  python -m app_support.export_to_gpt --from-parsed-cache

Outputs per document (under exports/<run_id>/):
  <filename>_raw.json        - Full parse result
  <filename>_llm_ready.json  - Lightweight: rag_text, block summaries, table markdown
  <filename>_llm_report.md   - Human-readable structured markdown
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from parsers import parse_document, SUPPORTED_EXTENSIONS, PARSER_VERSION
from parsers.pdf_parser import (
    _generate_rag_text,
    _rag_clean_text,
    _rag_clean_visual_summary,
    _rag_compact_body_text,
    _rag_compact_title,
    _rag_is_duplicate,
    _rag_mark_seen,
)


def _page_rag_text(page: dict[str, Any]) -> str:
    blocks = page.get("blocks") or []
    if blocks:
        return _generate_rag_text(blocks) or ""
    return page.get("rag_text") or ""


def _block_meta(block: dict[str, Any]) -> dict[str, Any]:
    meta = block.get("meta", {}) or {}
    return meta if isinstance(meta, dict) else {}


def _has_structured_summary(meta: dict[str, Any]) -> bool:
    return bool(
        meta.get("table_summary")
        or meta.get("table_markdown")
        or meta.get("key_value_rows")
        or meta.get("chart_summary")
        or meta.get("visual_summary")
        or meta.get("caption_text")
    )


def _priority_rank(value: Any) -> int:
    priority = str(value or "").lower()
    return {
        "high": 0,
        "medium": 1,
        "normal": 2,
        "low": 4,
    }.get(priority, 2)


def _coerce_order(value: Any, default: int = 999) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _block_roles(meta: dict[str, Any]) -> str:
    values = []
    for key in ("summary_role", "slide_role", "dashboard_role", "role"):
        value = meta.get(key)
        if value:
            values.append(str(value).lower())
    return " ".join(values)


def _is_decorative_or_disclaimer(block: dict[str, Any]) -> bool:
    meta = _block_meta(block)
    reason = str(meta.get("summary_exclude_reason") or "").lower()
    roles = _block_roles(meta)
    markers = ("decorative", "disclaimer", "footer", "page_number", "noise")
    return any(marker in reason or marker in roles for marker in markers)


def _is_title_candidate_excluded(block: dict[str, Any]) -> bool:
    meta = _block_meta(block)
    priority = str(meta.get("summary_priority") or "").lower()
    if block.get("type") == "footer":
        return True
    if priority == "low" or meta.get("summary_exclude"):
        return True
    return _is_decorative_or_disclaimer(block)


def _entry_panel_order(entry: dict[str, Any]) -> int:
    return _coerce_order(entry.get("panel_order"), _coerce_order(entry.get("slide_panel_order")))


def _visual_contexts(blocks: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for block in blocks:
        meta = _block_meta(block)
        if _is_decorative_or_disclaimer(block):
            continue
        btype = block.get("type")
        if btype == "title" and not meta.get("summary_exclude"):
            text = _rag_compact_title(block.get("text", ""))
        elif btype == "text" and not meta.get("summary_exclude"):
            text = _rag_compact_body_text(block.get("text", ""), 700)
        elif btype == "table" or meta.get("table_summary") or meta.get("table_markdown"):
            text = _rag_clean_text(meta.get("table_summary") or meta.get("table_markdown") or block.get("text", ""), 1000)
        else:
            text = ""
        if text:
            contexts.append(text)
    return contexts


def _clean_key_value_rows(rows: Any, limit: int = 20) -> list[Any]:
    if not isinstance(rows, list):
        return []
    cleaned: list[Any] = []
    for row in rows[:limit]:
        if isinstance(row, dict):
            item = _rag_clean_text(row.get("item"), 160)
            values = _rag_clean_text(row.get("values") or row.get("value"), 260)
            extra = {k: _rag_clean_text(v, 260) for k, v in row.items() if k not in {"item", "values", "value"}}
            out = {"item": item, "values": values}
            out.update({k: v for k, v in extra.items() if v})
            cleaned.append(out)
        else:
            value = _rag_clean_text(row, 300)
            if value:
                cleaned.append(value)
    return cleaned


def _key_values_as_text(rows: Any) -> str:
    lines = []
    for row in _clean_key_value_rows(rows, limit=8):
        if isinstance(row, dict):
            item = row.get("item")
            values = row.get("values")
            if item and values:
                lines.append(f"{item}: {values}")
        elif row:
            lines.append(str(row))
    return "\n".join(lines)


def _select_page_title(page: dict[str, Any], blocks: list[dict[str, Any]]) -> str:
    page_meta = page.get("meta") or page.get("metadata") or {}
    if not isinstance(page_meta, dict):
        page_meta = {}
    for key in ("page_title", "title", "heading"):
        title = _rag_compact_title(page.get(key) or page_meta.get(key))
        if title:
            return title

    candidates: list[tuple[int, int, str]] = []
    for idx, block in enumerate(blocks):
        if _is_title_candidate_excluded(block):
            continue
        meta = _block_meta(block)
        btype = block.get("type")
        roles = _block_roles(meta)
        priority = str(meta.get("summary_priority") or "").lower()
        score = 0

        text_candidates: list[Any] = []
        if btype == "title":
            score += 100
            text_candidates.append(block.get("text"))
        for key in ("page_title", "section_title", "title", "associated_title"):
            value = meta.get(key)
            if value:
                score += 40 if key != "associated_title" else 24
                text_candidates.append(value)
        if "title" in roles or "header" in roles:
            score += 28
        if priority == "high":
            score += 18
        elif priority == "medium":
            score += 10
        if btype == "text":
            score += 8
            text_candidates.append(block.get("text"))
        elif btype in {"table", "chart", "image"} and meta.get("associated_title"):
            score += 6

        for raw in text_candidates:
            title = _rag_compact_title(raw)
            if not title or sum(1 for ch in title if ch.isalpha()) < 2:
                continue
            length_penalty = max(0, len(title) - 120) // 8
            candidates.append((score - length_penalty, idx, title))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    return candidates[0][2]


def _page_content_dedupe_keys(blocks: list[dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    for block in blocks:
        meta = _block_meta(block)
        btype = block.get("type")
        priority = str(meta.get("summary_priority") or "").lower()
        if btype in {"image", "chart"}:
            continue
        if priority == "low" or _is_decorative_or_disclaimer(block):
            continue
        if btype == "title":
            text = _rag_compact_title(block.get("text", ""))
        elif btype == "table" or meta.get("table_summary") or meta.get("table_markdown"):
            text = _rag_clean_text(
                meta.get("table_summary")
                or _key_values_as_text(meta.get("key_value_rows"))
                or meta.get("table_markdown"),
                1000,
            )
        elif btype == "text" and not meta.get("summary_exclude"):
            text = _rag_compact_body_text(block.get("text", ""), 900)
        else:
            text = ""
        if text:
            _rag_mark_seen(text, seen)
    return seen


def _entry_section(entry: dict[str, Any]) -> str:
    btype = entry.get("type")
    priority = str(entry.get("summary_priority") or "").lower()
    if btype == "title":
        return "title"
    if entry.get("table_summary") or entry.get("table_markdown") or entry.get("key_value_rows"):
        return "table"
    if entry.get("chart_summary") or entry.get("visual_summary"):
        return "visual"
    if btype in {"chart", "image"}:
        return "visual"
    if btype == "footer" or priority == "low" or entry.get("summary_exclude_reason"):
        return "note"
    if btype == "text":
        return "body"
    return "note"


def _duplicate_against_texts(text: str, *texts: str, threshold: float = 0.90) -> bool:
    seen: set[str] = set()
    for value in texts:
        if value:
            _rag_mark_seen(value, seen)
    return bool(seen) and _rag_is_duplicate(text, seen, threshold=threshold)


def _format_table_entry(entry: dict[str, Any], markdown_limit: int = 1800) -> str:
    parts: list[str] = []
    title = _rag_clean_text(entry.get("associated_title") or entry.get("caption_text"), 180)
    context = _rag_clean_text(entry.get("table_context_text"), 500)
    summary = _rag_clean_text(entry.get("table_summary"), 1200)
    key_values = _key_values_as_text(entry.get("key_value_rows"))
    markdown = _rag_clean_text(entry.get("table_markdown"), markdown_limit)
    text = _rag_clean_text(entry.get("text"), 500)
    if title:
        parts.append(f"Title: {title}")
    if context and not _duplicate_against_texts(context, title, threshold=0.90):
        parts.append(f"Context: {context}")
    if summary:
        parts.append(f"Summary: {summary}")
    if key_values and not _duplicate_against_texts(key_values, summary, threshold=0.90):
        parts.append(f"Key values: {key_values}")
    if markdown and not _duplicate_against_texts(markdown, summary, key_values, threshold=0.90):
        parts.append(f"Markdown: {markdown}")
    if not parts and text:
        parts.append(text)
    return "\n".join(part for part in parts if part)


def _format_visual_entry(entry: dict[str, Any]) -> str:
    summary = _rag_clean_text(entry.get("chart_summary") or entry.get("visual_summary"), 1200)
    caption = _rag_clean_text(entry.get("caption_text"), 260)
    text = _rag_clean_text(entry.get("text"), 360)
    if summary and caption and not _duplicate_against_texts(caption, summary, threshold=0.90):
        return f"{summary}\nCaption: {caption}"
    return summary or caption or text


def _format_entry_for_page_text(entry: dict[str, Any]) -> tuple[str, str]:
    section = entry.get("llm_section") or _entry_section(entry)
    if section == "title":
        return "TITLE", _rag_compact_title(entry.get("text"))
    if section == "table":
        return "TABLE", _format_table_entry(entry)
    if section == "visual":
        label = "CHART" if entry.get("type") == "chart" else "VISUAL"
        return label, _format_visual_entry(entry)
    if section == "note":
        return "NOTE", _rag_clean_text(entry.get("text"), 320)
    return "BODY", _rag_compact_body_text(entry.get("text"), 900)


def _append_labeled_part(parts: list[str], seen: set[str], label: str, text: str, threshold: float = 0.90) -> None:
    clean = _rag_clean_text(text, 2200)
    if not clean or _rag_is_duplicate(clean, seen, threshold=threshold):
        return
    _rag_mark_seen(clean, seen)
    parts.append(f"[{label}] {clean}")


def _compose_llm_page_text(page_title: str, entries: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    if page_title:
        _append_labeled_part(parts, seen, "TITLE", page_title, threshold=0.96)
    for section in ("title", "body", "table", "visual", "note"):
        for entry in entries:
            if entry.get("llm_section") != section:
                continue
            label, text = _format_entry_for_page_text(entry)
            _append_labeled_part(parts, seen, label, text)
    return "\n\n".join(parts)


def _copy_llm_meta(entry: dict[str, Any], meta: dict[str, Any]) -> None:
    for key in (
        "summary_role",
        "slide_role",
        "dashboard_role",
        "role",
        "slide_panel_id",
        "slide_panel_order",
        "panel_id",
        "panel_order",
        "associated_title",
        "summary_priority",
        "summary_exclude_reason",
        "dashboard_region_id",
        "dashboard_region_ids",
        "region_id",
        "region_ids",
        "table_shape",
        "dashboard_table_shape",
        "dashboard_table_summary_kind",
    ):
        value = meta.get(key)
        if value is not None:
            entry[key] = value
    panel = meta.get("panel") or meta.get("panel_id") or meta.get("slide_panel_id")
    if panel is not None:
        entry["panel"] = panel


def _llm_block_entry(
    block: dict[str, Any],
    context_texts: list[str],
    seen_visual_summaries: set[str],
    seen_page_summaries: set[str],
    source_index: int,
) -> dict[str, Any] | None:
    meta = _block_meta(block)
    btype = block.get("type")
    raw_text = _rag_clean_text(block.get("text", ""))
    priority = str(meta.get("summary_priority") or "").lower()
    slide_role = str(meta.get("slide_role") or "")
    has_structured = _has_structured_summary(meta)
    has_table_structured = bool(meta.get("table_summary") or meta.get("table_markdown") or meta.get("key_value_rows"))
    has_visual_structured = bool(meta.get("chart_summary") or meta.get("visual_summary"))

    if btype == "footer" and (meta.get("summary_exclude") or priority == "low" or _is_decorative_or_disclaimer(block)):
        return None
    if meta.get("summary_exclude") and priority == "low" and btype in {"title", "text", "unknown"} and not has_structured:
        return None

    entry: dict[str, Any] = {"type": btype, "source_order": source_index}
    _copy_llm_meta(entry, meta)
    caption = _rag_clean_text(meta.get("caption_text"), 400)
    if caption:
        entry["caption_text"] = caption

    if btype == "title":
        entry_text = _rag_compact_title(raw_text) if not meta.get("summary_exclude") else ""
    elif btype == "table" or has_table_structured:
        table_summary = _rag_clean_text(meta.get("table_summary"), 1200)
        table_md = _rag_clean_text(meta.get("table_markdown"), 4000)
        key_values = _clean_key_value_rows(meta.get("key_value_rows"), limit=20)
        table_context = _rag_clean_text(
            meta.get("dashboard_table_context_text") or meta.get("table_context_text"),
            700,
        )
        if table_summary:
            entry["table_summary"] = table_summary
        if key_values:
            entry["key_value_rows"] = key_values
        if table_md:
            entry["table_markdown"] = table_md
        if table_context:
            entry["table_context_text"] = table_context
        entry_text = table_summary or _key_values_as_text(meta.get("key_value_rows")) or _rag_clean_text(raw_text, 420)
        if raw_text and table_summary and _rag_clean_text(raw_text, 280) != table_summary[:280]:
            entry["raw_text_excerpt"] = _rag_clean_text(raw_text, 280)
    elif btype in {"image", "chart"} or has_visual_structured:
        visual_summary = _rag_clean_visual_summary(block, context_texts)
        duplicate_visual = visual_summary and _rag_is_duplicate(visual_summary, seen_visual_summaries, threshold=0.90)
        duplicate_context = visual_summary and _rag_is_duplicate(visual_summary, seen_page_summaries, threshold=0.88)
        if duplicate_visual or duplicate_context:
            visual_summary = ""
        if visual_summary:
            _rag_mark_seen(visual_summary, seen_visual_summaries)
            entry["chart_summary" if btype == "chart" else "visual_summary"] = visual_summary
        visible_metrics = meta.get("visible_key_metrics")
        if visible_metrics:
            entry["visible_key_metrics"] = visible_metrics
        entry_text = _rag_clean_text(raw_text, 300)
        if not entry_text and not visual_summary and not caption:
            return None
    elif btype == "text":
        limit = 280 if slide_role == "offpage_text_stream" else 650
        entry_text = "" if meta.get("summary_exclude") or priority == "low" else _rag_compact_body_text(raw_text, limit)
    elif btype == "footer":
        entry_text = _rag_clean_text(raw_text, 240)
    else:
        entry_text = _rag_clean_text(raw_text, 420)
        if not entry_text and not has_structured:
            return None

    entry["text"] = entry_text
    entry["llm_section"] = _entry_section(entry)
    return entry


def _block_descriptor(meta: dict[str, Any]) -> str:
    parts = []
    panel = meta.get("panel") or meta.get("panel_id") or meta.get("slide_panel_id")
    if panel:
        parts.append(f"panel={panel}")
    panel_order = meta.get("panel_order")
    if panel_order is None:
        panel_order = meta.get("slide_panel_order")
    if panel_order is not None:
        parts.append(f"order={panel_order}")
    if meta.get("summary_priority"):
        parts.append(f"priority={meta.get('summary_priority')}")
    if meta.get("associated_title"):
        title = _rag_clean_text(meta.get("associated_title"), 80)
        parts.append(f"title={title}")
    return f" ({', '.join(parts)})" if parts else ""


def _llm_entry_order(entry: dict[str, Any], has_table: bool) -> tuple[int, int, int, int]:
    section = entry.get("llm_section") or _entry_section(entry)
    slide_role = entry.get("slide_role")
    if section == "body" and has_table and slide_role in {"offpage_text_stream", "kpi_group"}:
        section = "note"
    bucket = {
        "title": 0,
        "body": 1,
        "table": 2,
        "visual": 3,
        "note": 4,
    }.get(section, 5)
    priority = _priority_rank(entry.get("summary_priority"))
    panel_order = _entry_panel_order(entry)
    source_order = _coerce_order(entry.get("source_order"))
    return bucket, priority, panel_order, source_order


def _prepare_llm_page(page: dict[str, Any]) -> dict[str, Any]:
    blocks = page.get("blocks", []) or []
    context_texts = _visual_contexts(blocks)
    seen_visual_summaries: set[str] = set()
    seen_page_summaries = _page_content_dedupe_keys(blocks)
    block_summary: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        entry = _llm_block_entry(
            block,
            context_texts,
            seen_visual_summaries,
            seen_page_summaries,
            idx,
        )
        if entry is not None:
            block_summary.append(entry)

    has_page_table = any(entry.get("llm_section") == "table" for entry in block_summary)
    if has_page_table:
        for entry in block_summary:
            if entry.get("llm_section") == "body" and entry.get("slide_role") in {"offpage_text_stream", "kpi_group"}:
                entry["llm_section"] = "note"
    block_summary.sort(key=lambda entry: _llm_entry_order(entry, has_page_table))

    page_title = _select_page_title(page, blocks)
    if not page_title:
        for entry in block_summary:
            if entry.get("llm_section") == "title":
                page_title = _rag_compact_title(entry.get("text"))
                if page_title:
                    break

    structured_text = _compose_llm_page_text(page_title, block_summary)
    source_rag_text = _page_rag_text(page)
    page_text = structured_text or source_rag_text
    return {
        "blocks": blocks,
        "block_summary": block_summary,
        "page_title": page_title,
        "rag_text": page_text,
        "source_rag_text": source_rag_text,
        "used_structured_text": bool(structured_text),
    }


def _make_export_dir(run_id: str) -> Path:
    d = REPO_ROOT / "exports" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_llm_ready(result: dict[str, Any]) -> dict[str, Any]:
    meta = result.get("metadata", {})
    pages = result.get("pages", [])
    llm_pages = []
    for page in pages:
        prepared = _prepare_llm_page(page)
        blocks = prepared["blocks"]
        block_summary = prepared["block_summary"]
        rag_text = prepared["rag_text"]
        source_rag_text = prepared["source_rag_text"]

        llm_page = {
            "page_num": page.get("page_num"),
            "page_title": prepared["page_title"],
            "rag_text": rag_text,
            "rag_text_length": len(rag_text),
            "rag_text_source": "structured_export" if prepared["used_structured_text"] else "parser_rag_text",
            "block_count": len(blocks),
            "block_type_counts": page.get("parser_debug", {}).get("block_type_counts", {}),
            "blocks": block_summary,
        }
        if source_rag_text and source_rag_text != rag_text:
            llm_page["source_rag_text_length"] = len(source_rag_text)
        llm_pages.append(llm_page)

    return {
        "filename": result.get("filename"),
        "status": result.get("status"),
        "parser_version": result.get("parser_version", PARSER_VERSION),
        "document_type": meta.get("document_type"),
        "refined_document_type": meta.get("refined_document_type"),
        "pipeline_used": meta.get("pipeline_used"),
        "parser_used": meta.get("parser_used"),
        "page_count": len(pages),
        "total_chars": meta.get("total_chars", 0),
        "quality_score": meta.get("quality_score"),
        "quality_grade": meta.get("quality_grade"),
        "pages": llm_pages,
    }


def _generate_md_report(result: dict[str, Any]) -> str:
    meta = result.get("metadata", {})
    pages = result.get("pages", [])
    lines = [
        f"# {result.get('filename', 'Unknown')}",
        "",
        f"- **Status:** {result.get('status')}",
        f"- **Parser Version:** {result.get('parser_version', 'N/A')}",
        f"- **Document Type:** {meta.get('document_type', 'N/A')}",
        f"- **Refined Type:** {meta.get('refined_document_type', 'N/A')}",
        f"- **Pipeline:** {meta.get('pipeline_used', 'N/A')}",
        f"- **Pages:** {len(pages)}",
        f"- **Total Chars:** {meta.get('total_chars', 0)}",
        f"- **Quality:** {meta.get('quality_grade', 'N/A')} ({meta.get('quality_score', 'N/A')})",
        "",
        "---",
        "",
    ]

    for page in pages:
        pnum = page.get("page_num", "?")
        prepared = _prepare_llm_page(page)
        blocks = prepared["blocks"]
        block_summary = prepared["block_summary"]
        rag = prepared["rag_text"]
        debug = page.get("parser_debug", {})
        type_counts = debug.get("block_type_counts", {})

        lines.append(f"## Page {pnum}")
        lines.append("")
        if prepared["page_title"]:
            lines.append(f"**Page Title:** {prepared['page_title']}")
            lines.append("")
        lines.append(f"**Blocks:** {len(blocks)} | "
                      f"title={type_counts.get('title', 0)}, "
                      f"text={type_counts.get('text', 0)}, "
                      f"table={type_counts.get('table', 0)}, "
                      f"chart={type_counts.get('chart', 0)}, "
                      f"image={type_counts.get('image', 0)}, "
                      f"footer={type_counts.get('footer', 0)}")
        lines.append("")

        if rag:
            lines.append("### LLM-Ready Page Text")
            lines.append("")
            lines.append("```")
            lines.append(rag[:2600])
            if len(rag) > 2600:
                lines.append(f"... ({len(rag)} chars total)")
            lines.append("```")
            lines.append("")
        else:
            lines.append("**LLM-Ready Page Text:** *(empty)*")
            lines.append("")

        table_entries = [entry for entry in block_summary if entry.get("llm_section") == "table"]
        if table_entries:
            lines.append("### Tables")
            lines.append("")
            for i, entry in enumerate(table_entries[:6]):
                md = _rag_clean_text(entry.get("table_markdown", ""), 1200)
                table_summary = _rag_clean_text(entry.get("table_summary", ""), 1000)
                key_values = _key_values_as_text(entry.get("key_value_rows"))
                text = _rag_clean_text(entry.get("text") or "", 300)
                shape = entry.get("table_shape") or entry.get("dashboard_table_shape") or {}
                lines.append(f"**Table {i+1}:** {shape}{_block_descriptor(entry)}")
                if entry.get("table_context_text"):
                    lines.append("")
                    lines.append(f"Context: {_rag_clean_text(entry.get('table_context_text'), 500)}")
                if table_summary:
                    lines.append("")
                    lines.append(f"Summary: {table_summary}")
                if key_values:
                    lines.append("")
                    lines.append("Key values:")
                    lines.append(key_values)
                elif md:
                    lines.append("")
                    lines.append(md)
                elif text:
                    lines.append("")
                    lines.append(f"```\n{text}\n```")
                lines.append("")

        visual_entries = [entry for entry in block_summary if entry.get("llm_section") == "visual"]
        if visual_entries:
            lines.append("### Visuals")
            lines.append("")
            for i, entry in enumerate(visual_entries[:8]):
                visual_summary = _format_visual_entry(entry)
                if not visual_summary:
                    continue
                label = "Chart" if entry.get("type") == "chart" else "Visual"
                lines.append(f"**{label} {i+1}:**{_block_descriptor(entry)}")
                lines.append("")
                lines.append(visual_summary)
                lines.append("")

        note_entries = [entry for entry in block_summary if entry.get("llm_section") == "note" and entry.get("text")]
        if note_entries:
            lines.append("### Notes")
            lines.append("")
            for entry in note_entries[:5]:
                note = _rag_clean_text(entry.get("text"), 260)
                if note:
                    lines.append(f"- {note}{_block_descriptor(entry)}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_one(filepath_or_result: str | dict, export_dir: Path) -> dict[str, str]:
    if isinstance(filepath_or_result, dict):
        result = filepath_or_result
    else:
        print(f"  Parsing: {filepath_or_result}")
        result = parse_document(filepath_or_result)

    filename = result.get("filename", "unknown")
    stem = Path(filename).stem

    raw_path = export_dir / f"{stem}_raw.json"
    llm_path = export_dir / f"{stem}_llm_ready.json"
    md_path = export_dir / f"{stem}_llm_report.md"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    llm_ready = _extract_llm_ready(result)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_ready, f, ensure_ascii=False, indent=2)

    md_report = _generate_md_report(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"  Exported: {stem}")
    print(f"    raw:       {raw_path.name}")
    print(f"    llm_ready: {llm_path.name}")
    print(f"    md_report: {md_path.name}")

    return {"raw": str(raw_path), "llm_ready": str(llm_path), "md_report": str(md_path)}


def write_root_llm_exports(
    result: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Write latest-only sidecars under exports/latest_sidecars/."""
    root = (repo_root or REPO_ROOT).resolve()
    stem = Path(str(result.get("filename") or "export")).stem
    sidecar_dir = root / "exports" / "latest_sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    llm_path = sidecar_dir / f"{stem}_llm_ready.json"
    md_path = sidecar_dir / f"{stem}_llm_report.md"

    llm_ready = _extract_llm_ready(result)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_ready, f, ensure_ascii=False, indent=2)

    md_report = _generate_md_report(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    return {"llm_ready": str(llm_path), "md_report": str(md_path)}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    if "--from-parsed-cache" in args:
        pr = REPO_ROOT / "parsed_results"
        run_id = f"run_{int(time.time())}"
        export_dir = _make_export_dir(run_id)
        print(f"Export run: {run_id}")
        print(f"Output dir: {export_dir}")
        print()
        n_ok = 0
        for jf in sorted(pr.glob("*.json")):
            if not jf.is_file():
                continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except Exception as exc:
                print(f"  SKIP {jf.name}: {exc}")
                continue
            try:
                export_one(result, export_dir)
                n_ok += 1
            except Exception as exc:
                print(f"  ERROR exporting {jf.name}: {exc}")
        print(f"\nDone. {n_ok} document(s) exported to {export_dir}")
        return

    run_id = f"run_{int(time.time())}"
    export_dir = _make_export_dir(run_id)
    print(f"Export run: {run_id}")
    print(f"Output dir: {export_dir}")
    print()

    if "--all" in args:
        doc_dir = REPO_ROOT / "documents"
        if not doc_dir.is_dir():
            print(f"ERROR: documents/ directory not found at {doc_dir}")
            raise SystemExit(1)
        files = sorted(
            str(doc_dir / f)
            for f in os.listdir(doc_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print("No supported documents found.")
            raise SystemExit(1)
        for fp in files:
            try:
                export_one(fp, export_dir)
            except Exception as exc:
                print(f"  ERROR parsing {fp}: {exc}")
        print(f"\nDone. {len(files)} documents exported to {export_dir}")

    elif "--id" in args:
        idx = args.index("--id")
        if idx + 1 >= len(args):
            print("ERROR: --id requires a doc_id argument")
            raise SystemExit(1)
        doc_id = args[idx + 1]
        result_path = REPO_ROOT / "parsed_results" / f"{doc_id}.json"
        if not result_path.exists():
            print(f"ERROR: No cached result for doc_id={doc_id}")
            raise SystemExit(1)
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        export_one(result, export_dir)
        print(f"\nDone. Exported to {export_dir}")

    else:
        for fp in args:
            if not os.path.isfile(fp):
                print(f"  SKIP: {fp} (not found)")
                continue
            try:
                export_one(fp, export_dir)
            except Exception as exc:
                print(f"  ERROR parsing {fp}: {exc}")
        print(f"\nDone. Exported to {export_dir}")


if __name__ == "__main__":
    main()
