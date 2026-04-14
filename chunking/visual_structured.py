from __future__ import annotations

import json
import re
from typing import Any

from .utils import block_id, block_meta, clean_text, compact_join, is_excluded_block, page_num


VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s?(?:%p|%|bp|bps|억원|조원|만원|원|명|개|주|배|x|pt|십억원|백만원|조|억))?",
    re.IGNORECASE,
)
PERIOD_RE = re.compile(r"\b(?:[1-4]Q\d{2,4}|20\d{2}[./-]?\d{0,2})\b", re.IGNORECASE)
CHART_HINT_RE = re.compile(
    r"(chart|graph|trend|series|roe|bps|eps|margin|ratio|추이|그래프|차트|지표|비율|연환산|자기자본|수익률|판매관리비율)",
    re.IGNORECASE,
)
UNIT_RE = re.compile(r"(%p|%|bp|bps|억원|조원|만원|원|명|개|주|배|x|pt|십억원|백만원|조|억)$", re.IGNORECASE)


PHONEISH_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
EMAILISH_RE = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
URLISH_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)

FINANCIAL_TABLE_HINT_RE = re.compile(
    r"(손익계산서|재무상태표|현금흐름표|포괄손익|주요지표|투자지표|실적|매출|매출액|매출총이익|"
    r"영업이익|순이익|세전이익|자산|부채|자본|유동자산|유동부채|현금및현금성자산|"
    r"roe|eps|bps|per|pbr|opm|npm|ebitda|revenue|sales|gross profit|operating profit|"
    r"net income|assets|liabilities|equity|cash flow)",
    re.IGNORECASE,
)

MAX_STRUCTURED_TABLE_ROW_RECORDS = 24
MAX_STRUCTURED_TABLE_CELL_RECORDS = 72


def build_visual_structured_records(page: dict[str, Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(_table_records(page, blocks))
    chart_records = _chart_records(page, blocks)
    records.extend(chart_records)
    if not chart_records:
        page_record = _chart_like_page_record(page, blocks)
        if page_record:
            records.append(page_record)
    return records


def _table_records(page: dict[str, Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, block in enumerate([b for b in blocks if b.get("type") == "table"], start=1):
        if is_excluded_block(block, for_support=True):
            continue
        meta = block_meta(block)
        table = meta.get("normalized_table") if isinstance(meta.get("normalized_table"), dict) else {}
        summary = clean_text(meta.get("table_summary") or block.get("text"))
        markdown = clean_text(meta.get("table_markdown") or table.get("markdown"))
        rows = meta.get("key_value_rows")
        row_preview = _preview_rows(table.get("rows") or rows)
        if not summary and not markdown and not row_preview:
            continue
        if not _should_emit_table_record(block, summary=summary, markdown=markdown, rows=row_preview, table=table):
            continue
        structured = {
            "type": "table",
            "page_num": page_num(page),
            "table_index": index,
            "title": _visual_title(page, block),
            "table_kind": str(table.get("table_kind") or ""),
            "bbox": block.get("bbox") or [],
            "shape": table.get("shape") or meta.get("table_shape"),
            "headers": table.get("headers") or [],
            "rows_preview": row_preview,
            "confidence": _confidence(block, default=0.82),
            "extraction_method": meta.get("table_reconstruction_method") or table.get("source") or block.get("source") or "table_block",
        }
        parts = [
            "[VISUAL STRUCTURED TABLE]",
            f"document_page: {page_num(page)}",
            f"title: {structured['title']}",
            f"table_index: {index}",
            f"confidence: {structured['confidence']}",
            f"table_kind: {structured['table_kind']}",
            f"shape: {_json_text(structured.get('shape'))}",
            "[TABLE SUMMARY]",
            summary,
            "[TABLE MARKDOWN]",
            markdown,
        ]
        records.append(_record(
            chunk_type="table_summary",
            visual_type="table",
            visual_index=index,
            confidence=structured["confidence"],
            source_block_ids=[block_id(block)],
            blocks=[block],
            retrieval_text=compact_join(parts),
            structured=structured,
            source_bbox=block.get("bbox") or [],
            extraction_method=str(structured["extraction_method"]),
        ))
        if _should_emit_structured_table_records(table=table, summary=summary, title=structured["title"]):
            records.extend(
                _structured_table_records(
                    page=page,
                    block=block,
                    table=table,
                    table_index=index,
                    base_confidence=structured["confidence"],
                    title=structured["title"],
                    extraction_method=str(structured["extraction_method"]),
                )
            )
    return records


def _chart_records(page: dict[str, Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    chart_blocks = [b for b in blocks if b.get("type") == "chart"]
    for index, block in enumerate(chart_blocks, start=1):
        if is_excluded_block(block, for_support=True):
            continue
        meta = block_meta(block)
        chart_text = clean_text(meta.get("chart_summary") or meta.get("visual_summary") or meta.get("caption_text"))
        context_text = _context_text_for_block(page, block)
        visual_text = compact_join([chart_text, context_text])
        metrics = _metric_tokens(visual_text)
        periods = _period_tokens(visual_text)
        if not _should_emit_chart_record(block, chart_text=chart_text, context_text=context_text, metrics=metrics, periods=periods):
            continue
        labels = _label_candidates(compact_join([chart_text, context_text]))
        structured = {
            "type": "chart",
            "page_num": page_num(page),
            "chart_index": index,
            "title": _visual_title(page, block),
            "bbox": block.get("bbox") or [],
            "legend_or_series_candidates": labels[:12],
            "data_point_candidates": metrics[:40],
            "period_candidates": periods[:20],
            "trend_summary": _trend_summary(metrics, periods=periods, text=visual_text),
            "confidence": _confidence(block, default=0.72),
            "extraction_method": block.get("source") or "chart_block_context",
            "structured_status": "partial",
        }
        parts = [
            "[VISUAL STRUCTURED CHART]",
            f"document_page: {page_num(page)}",
            f"title: {structured['title']}",
            f"chart_index: {index}",
            f"confidence: {structured['confidence']}",
            f"structured_status: {structured['structured_status']}",
            f"series_or_legend_candidates: {', '.join(labels[:12])}",
            f"data_point_candidates: {', '.join(metrics[:40])}",
            f"period_candidates: {', '.join(structured['period_candidates'])}",
            f"trend_summary: {structured['trend_summary']}",
            "[CHART CONTEXT]",
            chart_text,
            context_text,
        ]
        records.append(_record(
            chunk_type="chart_summary",
            visual_type="chart",
            visual_index=index,
            confidence=structured["confidence"],
            source_block_ids=[block_id(block)],
            blocks=[block],
            retrieval_text=compact_join(parts),
            structured=structured,
            source_bbox=block.get("bbox") or [],
            extraction_method=str(structured["extraction_method"]),
        ))
    return records


def _chart_like_page_record(page: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    text = clean_text(page.get("rag_text") or page.get("text"))
    if not text:
        return None
    metrics = _metric_tokens(text)
    periods = _period_tokens(text)
    has_visual = any(b.get("type") in {"chart", "image"} and not is_excluded_block(b, for_support=True) for b in blocks)
    has_chart_hint = bool(CHART_HINT_RE.search(text))
    labels = _label_candidates(text)
    if not _should_emit_chart_like_page(text=text, metrics=metrics, periods=periods, labels=labels, has_visual=has_visual, has_chart_hint=has_chart_hint):
        return None
    source_ids = [block_id(b) for b in blocks if b.get("type") in {"text", "title"}][:12]
    structured = {
        "type": "chart_like_page",
        "page_num": page_num(page),
        "chart_index": 1,
        "title": _visual_title(page, {}),
        "bbox": [],
        "legend_or_series_candidates": labels[:16],
        "data_point_candidates": metrics[:60],
        "period_candidates": periods[:30],
        "trend_summary": _trend_summary(metrics, periods=periods, text=text),
        "confidence": 0.58 if has_chart_hint else 0.48,
        "extraction_method": "page_text_chart_like_heuristic",
        "structured_status": "partial",
    }
    parts = [
        "[VISUAL STRUCTURED CHART-LIKE PAGE]",
        f"document_page: {page_num(page)}",
        f"title: {structured['title']}",
        f"confidence: {structured['confidence']}",
        f"structured_status: {structured['structured_status']}",
        f"series_or_legend_candidates: {', '.join(labels[:16])}",
        f"data_point_candidates: {', '.join(metrics[:60])}",
        f"period_candidates: {', '.join(periods[:30])}",
        f"trend_summary: {structured['trend_summary']}",
        "[PAGE VISUAL CONTEXT]",
        text[:1600],
    ]
    return _record(
        chunk_type="chart_summary",
        visual_type="chart_like_page",
        visual_index=1,
        confidence=structured["confidence"],
        source_block_ids=source_ids,
        blocks=[b for b in blocks if block_id(b) in set(source_ids)],
        retrieval_text=compact_join(parts),
        structured=structured,
        source_bbox=[],
        extraction_method=str(structured["extraction_method"]),
    )


def _record(
    *,
    chunk_type: str,
    visual_type: str,
    visual_index: int,
    confidence: float,
    source_block_ids: list[str],
    blocks: list[dict[str, Any]],
    retrieval_text: str,
    structured: dict[str, Any],
    source_bbox: list[Any],
    extraction_method: str,
) -> dict[str, Any]:
    return {
        "chunk_type": chunk_type,
        "visual_type": visual_type,
        "visual_index": visual_index,
        "visual_confidence": round(float(confidence), 4),
        "source_bbox": _json_text(source_bbox),
        "extraction_method": extraction_method,
        "source_block_ids": source_block_ids,
        "blocks": blocks,
        "retrieval_text": retrieval_text,
        "display_text": retrieval_text,
        "metadata": {
            "visual_structured": structured,
            "structured_data_preview": _json_text(structured)[:1200],
        },
    }


def _context_text_for_block(page: dict[str, Any], block: dict[str, Any]) -> str:
    meta = block_meta(block)
    context_ids = {str(item) for item in (meta.get("context_block_ids") or [])}
    if not context_ids:
        return clean_text(page.get("rag_text") or page.get("text"))[:900]
    parts = []
    for other in page.get("blocks") or []:
        if isinstance(other, dict) and block_id(other) in context_ids:
            parts.append(clean_text(other.get("text")))
    return compact_join(parts)


def _metric_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in VALUE_RE.finditer(text or ""):
        token = clean_text(match.group(0))
        if not token:
            continue
        if _looks_like_plain_year(token):
            continue
        if not _looks_metric_like(token):
            continue
        compact = re.sub(r"\s+", "", token)
        if compact in seen:
            continue
        seen.add(compact)
        tokens.append(token)
    return tokens


def _text_richness(text: str) -> dict[str, int]:
    clean = clean_text(text)
    lines = [line for line in clean.splitlines() if clean_text(line)]
    return {
        "chars": len(clean),
        "lines": len(lines),
        "metrics": len(_metric_tokens(clean)),
        "periods": len(_period_tokens(clean)),
        "labels": len(_label_candidates(clean)),
    }


def _looks_contactish(text: str) -> bool:
    clean = clean_text(text)
    if len(clean) > 260:
        return False
    lines = [line for line in clean.splitlines() if clean_text(line)]
    structured_lines = sum(1 for line in lines if ":" in line or "|" in line)
    contact_signals = len(PHONEISH_RE.findall(clean)) + len(EMAILISH_RE.findall(clean)) + len(URLISH_RE.findall(clean))
    return contact_signals >= 1 and structured_lines >= 1 and len(_metric_tokens(clean)) <= 2


def _row_signal(rows: list[Any]) -> int:
    signal = 0
    for row in rows[:8]:
        if isinstance(row, dict):
            item = clean_text(row.get("item"))
            values = clean_text(row.get("values"))
            if item and values:
                signal += 2
            elif item or values:
                signal += 1
        elif isinstance(row, (list, tuple)):
            cells = [clean_text(x) for x in row if clean_text(x)]
            if len(cells) >= 3:
                signal += 2
            elif len(cells) == 2:
                signal += 1
    return signal


def _should_emit_structured_table_records(*, table: dict[str, Any], summary: str, title: str) -> bool:
    semantic_rows = _semantic_table_rows(table)
    if len(semantic_rows) < 2:
        return False
    return bool(table.get("row_cell_retrieval_candidate") or table.get("financial_table_candidate"))


def _structured_table_records(
    *,
    page: dict[str, Any],
    block: dict[str, Any],
    table: dict[str, Any],
    table_index: int,
    base_confidence: float,
    title: str,
    extraction_method: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_ids = [block_id(block)]
    semantic_rows = _semantic_table_rows(table)[:MAX_STRUCTURED_TABLE_ROW_RECORDS]
    cell_count = 0
    table_kind = clean_text(table.get("table_kind"))

    for row in semantic_rows:
        row_label = clean_text(row.get("row_label")) or f"row_{int(row.get('row_index') or 0) + 1}"
        pairs = [
            pair for pair in (row.get("pairs") or [])
            if clean_text(pair.get("column_label")) and clean_text(pair.get("value"))
        ]
        if not pairs:
            continue

        pair_preview = [_financial_pair_text(pair) for pair in pairs[:10]]
        row_text = compact_join([
            "[VISUAL STRUCTURED TABLE ROW]",
            f"document_page: {page_num(page)}",
            f"title: {title}",
            f"table_index: {table_index}",
            f"table_kind: {table_kind}",
            f"row_index: {row.get('row_index')}",
            f"row_label: {row_label}",
            "[ROW VALUES]",
            " | ".join(pair_preview),
        ])
        row_structured = {
            "type": "table_row",
            "page_num": page_num(page),
            "table_index": table_index,
            "title": title,
            "table_kind": table_kind,
            "row_index": row.get("row_index"),
            "row_label": row_label,
            "pairs_preview": pairs[:10],
            "confidence": round(max(0.0, base_confidence - 0.03), 4),
        }
        records.append(_record(
            chunk_type="table_row",
            visual_type="table",
            visual_index=table_index,
            confidence=row_structured["confidence"],
            source_block_ids=source_ids,
            blocks=[block],
            retrieval_text=row_text,
            structured=row_structured,
            source_bbox=block.get("bbox") or [],
            extraction_method=extraction_method,
        ))

        for pair in pairs:
            if cell_count >= MAX_STRUCTURED_TABLE_CELL_RECORDS:
                break
            column_label = clean_text(pair.get("column_label"))
            value = clean_text(pair.get("value"))
            if not column_label or not value:
                continue
            if not (_looks_numeric_like(value) or table_kind in {"returns_table", "period_matrix_table"}):
                continue
            cell_text = compact_join([
                "[VISUAL STRUCTURED TABLE CELL]",
                f"document_page: {page_num(page)}",
                f"title: {title}",
                f"table_index: {table_index}",
                f"table_kind: {table_kind}",
                f"row_index: {row.get('row_index')}",
                f"row_label: {row_label}",
                f"column_label: {column_label}",
                f"value: {value}",
                "[CELL CONTEXT]",
                f"{row_label} | {column_label} | {value}",
            ])
            cell_structured = {
                "type": "table_cell",
                "page_num": page_num(page),
                "table_index": table_index,
                "title": title,
                "table_kind": table_kind,
                "row_index": row.get("row_index"),
                "row_label": row_label,
                "column_index": pair.get("column_index"),
                "column_label": column_label,
                "value": value,
                "confidence": round(max(0.0, base_confidence - 0.06), 4),
            }
            records.append(_record(
                chunk_type="table_cell",
                visual_type="table",
                visual_index=table_index,
                confidence=cell_structured["confidence"],
                source_block_ids=source_ids,
                blocks=[block],
                retrieval_text=cell_text,
                structured=cell_structured,
                source_bbox=block.get("bbox") or [],
                extraction_method=extraction_method,
            ))
            cell_count += 1
        if cell_count >= MAX_STRUCTURED_TABLE_CELL_RECORDS:
            break
    return records


def _semantic_table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = table.get("semantic_data_rows")
    if isinstance(rows, list) and rows:
        return [row for row in rows if isinstance(row, dict)]

    raw_rows = table.get("rows") if isinstance(table.get("rows"), list) else []
    if not raw_rows:
        return []
    data_row_start_index = int(table.get("data_row_start_index") or 0)
    row_label_col_index = int(table.get("row_label_col_index") or 0)
    width = max((len(row) for row in raw_rows if isinstance(row, list)), default=0)
    headers = table.get("headers") if isinstance(table.get("headers"), list) else []
    data_col_indices = table.get("data_col_indices")
    if not isinstance(data_col_indices, list) or not data_col_indices:
        data_col_indices = [idx for idx in range(width) if idx != row_label_col_index]

    semantic_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(raw_rows[data_row_start_index:], start=data_row_start_index):
        if not isinstance(row, list):
            continue
        row_label = ""
        if row_label_col_index < len(row):
            row_label = clean_text(row[row_label_col_index])
        if not row_label:
            row_label = next((clean_text(cell) for cell in row if clean_text(cell) and not _looks_numeric_like(clean_text(cell))), "")
        pairs = []
        for col_index in data_col_indices:
            value = clean_text(row[col_index]) if col_index < len(row) else ""
            if not value:
                continue
            column_label = clean_text(headers[col_index]) if col_index < len(headers) else f"col_{col_index + 1}"
            pairs.append({
                "column_index": col_index,
                "column_label": column_label or f"col_{col_index + 1}",
                "value": value,
            })
        if pairs:
            semantic_rows.append({
                "row_index": row_index,
                "row_label": row_label or f"row_{row_index + 1}",
                "pairs": pairs,
            })
    return semantic_rows


def _financial_pair_text(pair: dict[str, Any]) -> str:
    column_label = clean_text(pair.get("column_label")) or "value"
    value = clean_text(pair.get("value"))
    return f"{column_label}={value}" if value else column_label


def _looks_numeric_like(text: str) -> bool:
    clean = clean_text(text)
    if not clean:
        return False
    if re.fullmatch(r"[()\-\s]*\d[\d,./%-]*", clean):
        return True
    if re.search(r"\d", clean) and re.search(r"(억원|백만원|만원|원|%|bp|bps|배|x|주|천주|십억원|조원|usd|krw)", clean, re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Za-z]{1,4}\s*\d+(?:\.\d+)?", clean):
        return True
    return False


def _should_emit_table_record(
    block: dict[str, Any],
    *,
    summary: str,
    markdown: str,
    rows: list[Any],
    table: dict[str, Any],
) -> bool:
    richness = _text_richness(compact_join([summary, markdown]))
    row_signal = _row_signal(rows)
    headers = table.get("headers") if isinstance(table.get("headers"), list) else []
    shape = table.get("shape")
    score = 0
    if richness["chars"] >= 120:
        score += 2
    elif richness["chars"] >= 70:
        score += 1
    if richness["metrics"] >= 4:
        score += 2
    elif richness["metrics"] >= 2:
        score += 1
    if row_signal >= 4:
        score += 2
    elif row_signal >= 2:
        score += 1
    if markdown.count("|") >= 10 or markdown.count("\n") >= 4:
        score += 1
    if len(headers) >= 2:
        score += 1
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            if int(shape[0]) >= 3 and int(shape[1]) >= 2:
                score += 1
        except (TypeError, ValueError):
            pass
    if _looks_contactish(compact_join([summary, markdown])):
        score -= 4
    if richness["chars"] < 45 and row_signal < 2:
        score -= 2
    return score >= 4


def _should_emit_chart_record(
    block: dict[str, Any],
    *,
    chart_text: str,
    context_text: str,
    metrics: list[str],
    periods: list[str],
) -> bool:
    combined = compact_join([chart_text, context_text])
    richness = _text_richness(combined)
    labels = _label_candidates(combined)
    score = 0
    if len(clean_text(chart_text)) >= 80:
        score += 2
    elif clean_text(chart_text):
        score += 1
    if len(metrics) >= 4:
        score += 2
    elif len(metrics) >= 2:
        score += 1
    if len(periods) >= 2:
        score += 2
    elif len(periods) == 1:
        score += 1
    if len(labels) >= 2:
        score += 1
    if richness["chars"] >= 140:
        score += 1
    if _looks_contactish(combined):
        score -= 4
    if len(metrics) <= 1 and len(periods) == 0:
        score -= 3
    if len(labels) >= 4 and len(metrics) == 0 and len(periods) == 0:
        score -= 3
    return score >= 4


def _should_emit_chart_like_page(
    *,
    text: str,
    metrics: list[str],
    periods: list[str],
    labels: list[str],
    has_visual: bool,
    has_chart_hint: bool,
) -> bool:
    richness = _text_richness(text)
    if not (has_visual or has_chart_hint):
        return False
    score = 0
    if len(metrics) >= 6:
        score += 3
    elif len(metrics) >= 4:
        score += 2
    if len(periods) >= 2:
        score += 2
    elif len(periods) == 1:
        score += 1
    if len(labels) >= 3:
        score += 1
    if richness["chars"] >= 220:
        score += 1
    if has_chart_hint:
        score += 1
    if _looks_contactish(text):
        score -= 4
    return score >= 5


def _period_tokens(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in PERIOD_RE.finditer(text or ""):
        token = match.group(0)
        key = token.lower()
        if key not in seen:
            seen.add(key)
            out.append(token)
    return out


def _label_candidates(text: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for line in re.split(r"[\n\r|/]+", text or ""):
        clean = clean_text(line)
        if len(clean) < 3 or len(clean) > 90:
            continue
        if not re.search(r"[A-Za-z가-힣]", clean):
            continue
        if len(_metric_tokens(clean)) >= 4 and len(clean) > 50:
            continue
        key = re.sub(r"\s+", "", clean).lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(clean)
    return labels


def _trend_summary(metrics: list[str], *, periods: list[str] | None = None, text: str = "") -> str:
    numeric = [(metric, parsed[0], parsed[1]) for metric in metrics if (parsed := _parse_metric_value(metric)) is not None]
    if len(numeric) < 2:
        return "numeric labels were extracted, but a trend could not be inferred."
    has_series_signal = bool(periods) or bool(re.search(r"(ROE|BPS|EPS|규제비율|판매관리비율|자기자본비율|레버리지비율)", text or "", re.IGNORECASE))
    if not has_series_signal:
        return "numeric labels were extracted; axis/series pairing is partial, so no trend is inferred."

    unit_groups: dict[str, list[tuple[str, float, str]]] = {}
    for item in numeric:
        unit_groups.setdefault(item[2], []).append(item)
    same_unit = max(unit_groups.values(), key=len)
    if len(same_unit) < 2:
        return "numeric labels were extracted, but units are mixed, so trend inference is partial."
    if not periods and same_unit[0][2] not in {"%", "%p", "bp", "bps"}:
        return "numeric labels were extracted; axis/series pairing is partial, so no trend is inferred."
    first_token, first_value, _ = same_unit[0]
    last_token, last_value, _ = same_unit[-1]
    if last_value > first_value:
        direction = "increased"
    elif last_value < first_value:
        direction = "decreased"
    else:
        direction = "was flat"
    return f"same-unit numeric labels {first_token} -> {last_token} appear to have {direction}; verify axis/series pairing against the source visual."


def _parse_metric_value(metric: str) -> tuple[float, str] | None:
    compact = re.sub(r"\s+", "", metric or "")
    number = re.search(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)", compact)
    if not number:
        return None
    unit_match = UNIT_RE.search(compact)
    unit = unit_match.group(1).lower() if unit_match else ""
    try:
        return float(number.group(0).replace(",", "")), unit
    except ValueError:
        return None


def _looks_metric_like(token: str) -> bool:
    compact = re.sub(r"\s+", "", token)
    if UNIT_RE.search(compact):
        return True
    if "," in compact:
        return True
    if re.fullmatch(r"[+-]?\d+\.\d+", compact):
        return True
    return False


def _looks_like_plain_year(token: str) -> bool:
    return bool(re.fullmatch(r"20\d{2}", re.sub(r"\s+", "", token or "")))


def _preview_rows(rows: Any, limit: int = 12) -> list[Any]:
    if not isinstance(rows, list):
        return []
    return rows[:limit]


def _visual_title(page: dict[str, Any], block: dict[str, Any]) -> str:
    meta = block_meta(block) if block else {}
    for value in (meta.get("associated_title"), meta.get("page_title"), page.get("page_title")):
        text = clean_text(value)
        if text:
            return text[:200]
    text = clean_text(page.get("text") or page.get("rag_text"))
    if text:
        return text.splitlines()[0][:200]
    return f"page {page_num(page)} visual"


def _confidence(block: dict[str, Any], *, default: float) -> float:
    meta = block_meta(block)
    for value in (block.get("confidence"), meta.get("table_quality"), meta.get("overall_table_quality")):
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score > 1.0:
            score = min(0.98, score / 10.0)
        return round(max(0.0, min(1.0, score)), 4)
    return default


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
