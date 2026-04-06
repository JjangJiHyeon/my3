"""
Parser for HWP (Hangul Word Processor) files.

HWP5 is an OLE2 compound file.  Text lives in BodyText/Section{N}
streams, optionally zlib-compressed.  HWPTAG_PARA_TEXT (tag 67)
records hold UTF-16LE text with HWP-specific control characters.
"""

from __future__ import annotations

import logging
import re
import struct
import zlib
from typing import Any

import olefile

# ── parser ─────────────────────────────────────────────────────────

HWPTAG_PARA_TEXT = 67
HWPTAG_CTRL_HEADER = 71
HWPTAG_TABLE = 78
HWPTAG_LIST_HEADER = 73

EXTENDED_CONTROLS = frozenset(
    {1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
)

_TABLE_START_MARKER = "\n<TABLE_START>\n"
_TABLE_END_MARKER = "\n<TABLE_END>\n"
_CELL_BREAK_MARKER = "\n<CELL_BREAK>\n"


# ── text normalisation ───────────────────────────────────────────────


def _estimate_text_quality(pages: list[dict[str, Any]]) -> float:
    if not pages:
        return 0.0
    empty_pages = sum(1 for page in pages if not str(page.get("text", "") or "").strip())
    avg_text_chars = sum(len(str(page.get("text", "") or "")) for page in pages) / len(pages)
    quality = 100.0
    quality -= (empty_pages / max(1, len(pages))) * 45.0
    if avg_text_chars < 300:
        quality -= 10.0
    elif avg_text_chars > 1200:
        quality += 3.0
    return round(max(0.0, min(100.0, quality)), 2)


def _align_hwp_metadata(
    metadata: dict[str, Any],
    pages: list[dict[str, Any]],
    routing_result: dict[str, Any],
    parser_subtype: str,
) -> None:
    doc_type = routing_result.get("document_type", "text_report")
    page_count = len(pages)
    empty_pages = sum(1 for page in pages if not page.get("blocks") or not str(page.get("text", "") or "").strip())
    page_type_distribution = routing_result.get("page_type_distribution") or {}
    routing_reasons = routing_result.get("routing_reasons") or [routing_result.get("routing_reason", "")]
    title_block_count = sum(
        1 for page in pages for block in page.get("blocks", []) if block.get("type") == "title"
    )
    text_block_count = sum(
        1 for page in pages for block in page.get("blocks", []) if block.get("type") == "text"
    )
    table_count = sum(len(page.get("tables", []) or []) for page in pages)

    metadata.update({
        "document_type": doc_type,
        "refined_document_type": doc_type,
        "refined_routing_type": doc_type,
        "doc_type": routing_result.get("doc_type", doc_type),
        "routing_confidence": routing_result.get("confidence", 0.0),
        "page_type_distribution": page_type_distribution,
        "routing_reasons": routing_reasons,
        "routing_signals": routing_result.get("routing_signals"),
        "routing_reason": routing_result.get("routing_reason"),
        "routing_mismatch_flag": False,
        "text_quality": _estimate_text_quality(pages),
        "empty_pages": empty_pages,
        "pipeline_used": f"hwp_parser::{parser_subtype}",
        "parser_subtype": parser_subtype,
        "content_structure": {
            "page_count": page_count,
            "title_block_count": title_block_count,
            "text_block_count": text_block_count,
            "table_count": table_count,
        },
    })

def _analyze_narrative_characteristics(text: str) -> dict[str, Any]:
    import re
    clean = text.replace(" ", "").replace("\n", "")
    num_ratio = len(re.findall(r'\d', clean)) / max(1, len(clean))
    sentences = len(re.findall(r'[가-힣][다음함]\.', text))
    
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    long_paras = sum(1 for p in paragraphs if len(p) > 60 and ('다' in p or '음' in p or '함' in p))
    label_patterns = sum(1 for p in paragraphs if len(p) < 30 and re.search(r'[:\-\u2022\[\]]', p))
    
    return {
        "num_ratio": round(num_ratio, 3),
        "sentence_endings_found": sentences,
        "long_narrative_paragraphs": long_paras,
        "short_label_patterns": label_patterns,
        "is_false_positive_warning_candidate": bool(num_ratio > 0.25 and sentences <= 1 and long_paras > 0),
        "is_truly_dashboard": bool(num_ratio > 0.25 and sentences <= 1 and long_paras == 0)
    }

def _normalise(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" +", " ", text)  # collapse spaces only, keep tabs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── section header splitting ─────────────────────────────────────────

# 줄 전체가 독립 헤더일 때만 매치 (anchored)
_SECTION_HEADER_RE = re.compile(
    r'^\s*'
    r'(?:'
    r'[Ⅰ-Ⅹ]'                           # 로마자 Ⅰ~Ⅹ
    r'|V?I{0,3}'                         # ASCII 로마자 I, II, III, IV, V, VI
    r'|[0-9]{1,2}'                       # 아라비아 숫자 (최대 2자리)
    r'|[가-힣]'                           # 한글 가, 나, 다
    r'|[A-Z]'                            # 영문 대문자 A, B, C
    r')'
    r'[\.\)]\s*'                         # 점 또는 닫는 괄호 + 선택적 공백
    r'(.+)',                             # 뒤에 텍스트가 있어야 함
    re.MULTILINE
)

# 날짜/표성 숫자 라인은 제외
_DATE_LINE_RE = re.compile(
    r'^\s*\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}'   # 2022.01.19 형태
)
_PURE_NUMBER_RE = re.compile(
    r'^\s*[\d,.\-+%()△▲▽※ ]+\s*$'           # 숫자/부호만으로 구성
)
_VALUE_UNIT_PATTERN = r"(?:%|억원|조원|백만원|천원|만원|원|건|배|명|개)"


def _is_doc_section_header(line: str) -> bool:
    """줄 전체가 독립적인 섹션 헤더인지 판단"""
    stripped = line.strip()
    # 너무 긴 라인은 헤더가 아님
    if len(stripped) > 60:
        return False
    # 날짜 라인 제외
    if _DATE_LINE_RE.match(stripped):
        return False
    # 순수 숫자/기호 라인 제외
    if _PURE_NUMBER_RE.match(stripped):
        return False
    # 섹션 패턴 매치
    if _SECTION_HEADER_RE.match(stripped):
        return True
    if re.match(r'^\s*(?:[0-9]{1,2}|[Ⅰ-Ⅹ])\s+.+$', stripped):
        return True
    return False


def _is_data_row(line: str) -> bool:
    """순수 숫자/금액 행인지 판단"""
    stripped = line.strip()
    if not stripped:
        return False
    if _PURE_NUMBER_RE.match(stripped):
        return True
    return False


def _split_tabular_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    colon_match = re.match(r'^\s*(?:[-•○□◦▪]\.?\s*)?(?P<label>[^:]{1,40})\s*:\s*(?P<value>.+)$', stripped)
    if colon_match:
        label = colon_match.group("label").strip()
        value = colon_match.group("value").strip()
        if _looks_value_like(value) and not _is_list_marker(label):
            return [label, value]
    if "|" in stripped:
        cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
        if len(cells) >= 2:
            return cells
    cells = [cell.strip() for cell in re.split(r"\t+|\s{2,}", stripped) if cell.strip()]
    if len(cells) >= 2:
        return cells
    tokens = stripped.split()
    if len(tokens) >= 4:
        paired_cells: list[str] = []
        label_tokens: list[str] = []
        value_pairs = 0
        for token in tokens:
            is_value = bool(re.fullmatch(r"[\d,]+(?:\.\d+)?(?:%|억원|조원|백만원|천원|만원|원|건|배)?", token))
            if is_value and label_tokens:
                paired_cells.append(" ".join(label_tokens).strip())
                paired_cells.append(token)
                label_tokens.clear()
                value_pairs += 1
            else:
                label_tokens.append(token)
        if value_pairs >= 2 and len(paired_cells) >= 4:
            return paired_cells
    return []


def _looks_value_like(text: str) -> bool:
    compact = " ".join(text.split())
    if len(compact) > 42 or not re.search(r"\d", compact):
        return False
    if re.search(r"(감소|증가|발행|조달|대비|으로|이며|합니다|하였다|했다)", compact):
        return False
    unit_hits = len(re.findall(_VALUE_UNIT_PATTERN, compact))
    alpha_groups = len(re.findall(r"[A-Za-z가-힣]+", compact))
    return bool(unit_hits or len(re.findall(r"\d", compact)) >= 3) and alpha_groups <= 4


def _is_summary_pair_row(row: list[str]) -> bool:
    if len(row) != 2:
        return False
    label = row[0].strip()
    value = row[1].strip()
    if not label or not value or len(label) > 40 or _is_list_marker(label):
        return False
    if re.search(r"\d{2,}", label):
        return False
    return _looks_value_like(value)


def _is_list_marker(cell: str) -> bool:
    return bool(re.fullmatch(r"[□◦▪•\-\u25e6\u2022\u25a1\u25cb\u2023\u25a0]|[\d]{1,2}[\.\)]|[Ⅰ-Ⅹ][\.\)]|[가-힣][\.\)]", cell.strip()))


def _is_compact_label_value_row(row: list[str]) -> bool:
    if len(row) < 4 or len(row) % 2 != 0:
        return False
    pair_count = 0
    for i in range(0, len(row), 2):
        label = row[i].strip()
        value = row[i + 1].strip()
        if not label or not value or len(label) > 30 or not re.search(r"\d", value):
            return False
        pair_count += 1
    return pair_count >= 2


def _should_promote_table_rows(table_rows: list[list[str]]) -> tuple[bool, str]:
    if not table_rows:
        return False, "empty_rows"

    row_lengths = [len(row) for row in table_rows if row]
    if not row_lengths:
        return False, "empty_rows"

    shape_cols = max(row_lengths)
    stable_rows = sum(1 for length in row_lengths if abs(length - shape_cols) <= 1)
    stability = stable_rows / len(row_lengths)
    numeric_cells = sum(1 for row in table_rows for cell in row if re.search(r"\d", cell))
    total_cells = sum(len(row) for row in table_rows)
    numeric_ratio = numeric_cells / max(1, total_cells)

    if len(table_rows) == 1:
        if _is_compact_label_value_row(table_rows[0]):
            return True, "compact_label_value_row"
        if _is_summary_pair_row(table_rows[0]):
            return True, "single_summary_pair_row"
        return False, "single_row_not_compact"

    if shape_cols < 2 or stability < 0.65:
        return False, "unstable_columns"

    first_col = [row[0].strip() for row in table_rows if row]
    first_col_list_ratio = sum(1 for cell in first_col if _is_list_marker(cell)) / max(1, len(first_col))
    second_col_avg_len = sum(len(row[1].strip()) for row in table_rows if len(row) > 1) / max(1, sum(1 for row in table_rows if len(row) > 1))
    if shape_cols == 2 and first_col_list_ratio > 0.6 and second_col_avg_len > 40:
        return False, "bullet_narrative_list"

    long_cell_ratio = sum(1 for row in table_rows for cell in row if len(cell) > 70) / max(1, total_cells)
    if long_cell_ratio > 0.35 and numeric_ratio < 0.2:
        return False, "paragraph_like_cells"

    first_row = table_rows[0]
    header_like = sum(1 for cell in first_row if not re.search(r"\d", cell) and len(cell) <= 30) / max(1, len(first_row))
    repeated_label_value = sum(1 for row in table_rows if _is_compact_label_value_row(row)) >= max(1, len(table_rows) // 2)
    summary_pair_rows = sum(1 for row in table_rows if _is_summary_pair_row(row))
    if summary_pair_rows >= max(2, len(table_rows) - 1):
        return True, "summary_pair_rows"
    if numeric_ratio >= 0.15 or header_like >= 0.5 or repeated_label_value:
        return True, "stable_tabular_pattern"
    return False, "weak_tabular_signal"


def _looks_table_like_line(line: str) -> bool:
    cells = _split_tabular_cells(line)
    if len(cells) < 2:
        return False
    numeric_like = sum(1 for cell in cells if re.search(r"\d", cell))
    compact_like = sum(1 for cell in cells if len(cell) <= 30)
    if len(cells) == 2 and _is_list_marker(cells[0]) and len(cells[1]) > 40:
        return False
    return (
        numeric_like >= 1
        or compact_like >= max(2, len(cells) - 1)
        or _is_compact_label_value_row(cells)
        or _is_summary_pair_row(cells)
    )


def _table_text_from_rows(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows if any(cell.strip() for cell in row))


def _flush_table_block(
    blocks: list[dict[str, Any]],
    table_rows: list[list[str]],
    page_num: int,
    block_idx: int,
    source: str,
) -> int:
    should_promote, structure_reason = _should_promote_table_rows(table_rows)
    if not should_promote:
        return block_idx
    shape_cols = max((len(row) for row in table_rows), default=0)
    normalized_rows = [row + [""] * max(0, shape_cols - len(row)) for row in table_rows]
    context_title = None
    if blocks:
        prev_block = blocks[-1]
        prev_text = str(prev_block.get("text", "") or "").strip()
        if (
            prev_block.get("type") == "text"
            and prev_text
            and len(prev_text) <= 40
            and "\n" not in prev_text
            and not re.search(r"\d{3,}", prev_text)
        ):
            context_title = prev_text
    blocks.append({
        "id": f"p{page_num}_b{block_idx}",
        "type": "table",
        "bbox": [0, 0, 0, 0],
        "text": _table_text_from_rows(normalized_rows),
        "page_num": page_num,
        "source": source,
        "score": 1.0,
        "meta": {
            "rows": normalized_rows,
            "block_subtype": "table_like_rows",
            "table_candidate_score": 3.0,
            "table_shape": {"rows": len(normalized_rows), "cols": shape_cols},
            "structure_reason": structure_reason,
            "context_title": context_title,
        }
    })
    return block_idx + 1


def _structure_text_blocks(
    text: str,
    page_num: int,
    source: str,
) -> tuple[list[dict[str, Any]], list[list[list[str]]], dict[str, int]]:
    """텍스트를 섹션 헤더 기준으로 title/text 블록으로 분리.
    빈 줄 기준 body 분리 + 순수 숫자 데이터 행 분리.
    원문 순서 무변경, 텍스트 내용 무변경."""
    if not text or not text.strip():
        return [], [], {
            "title_count": 0,
            "text_count": 0,
            "table_count": 0,
            "table_like_line_count": 0,
            "compact_table_row_count": 0,
            "tab_hint_line_count": 0,
        }

    lines = text.split('\n')
    blocks: list[dict[str, Any]] = []
    tables: list[list[list[str]]] = []
    block_idx = 0
    current_body_lines: list[str] = []
    current_data_lines: list[str] = []
    current_table_rows: list[list[str]] = []
    pending_table_break = False
    stats = {
        "title_count": 0,
        "text_count": 0,
        "table_count": 0,
        "table_like_line_count": 0,
        "compact_table_row_count": 0,
        "tab_hint_line_count": 0,
        "summary_pair_row_count": 0,
    }

    def _flush_body():
        nonlocal block_idx
        body_text = '\n'.join(current_body_lines).strip()
        if body_text:
            blocks.append({
                "id": f"p{page_num}_b{block_idx}",
                "type": "text",
                "bbox": [0, 0, 0, 0],
                "text": body_text,
                "page_num": page_num,
                "source": source,
                "score": 1.0,
                "meta": {}
            })
            block_idx += 1
            stats["text_count"] += 1
        current_body_lines.clear()

    def _flush_data():
        nonlocal block_idx
        data_text = '\n'.join(current_data_lines).strip()
        if data_text:
            blocks.append({
                "id": f"p{page_num}_b{block_idx}",
                "type": "text",
                "bbox": [0, 0, 0, 0],
                "text": data_text,
                "page_num": page_num,
                "source": source,
                "score": 1.0,
                "meta": {"block_subtype": "data_rows"}
            })
            block_idx += 1
            stats["text_count"] += 1
        current_data_lines.clear()

    def _flush_table():
        nonlocal block_idx
        promoted_before = block_idx
        if current_table_rows:
            block_idx = _flush_table_block(blocks, current_table_rows, page_num, block_idx, source)
        if block_idx > promoted_before:
            tables.append([row[:] for row in current_table_rows])
            stats["table_count"] += 1
        elif current_table_rows:
            current_body_lines.extend(" ".join(row) for row in current_table_rows)
        current_table_rows.clear()

    in_record_table = False
    record_table_cells: list[str] = []
    record_table_rows: list[list[str]] = []

    def _flush_record_cell():
        if record_table_cells:
            cell_text = " ".join(record_table_cells).strip()
            if cell_text:
                record_table_rows.append(cell_text)
            record_table_cells.clear()

    def _emit_record_table_with_cols(cells: list[str], col_count: int):
        nonlocal block_idx
        if not cells or col_count < 2:
            _emit_record_table()
            return
        rows: list[list[str]] = []
        for i in range(0, len(cells), col_count):
            row = cells[i:i + col_count]
            if len(row) < col_count:
                row.extend([""] * (col_count - len(row)))
            rows.append(row)
        if rows:
            table_text = _table_text_from_rows(rows)
            blocks.append({
                "id": f"p{page_num}_b{block_idx}",
                "type": "table",
                "bbox": [0, 0, 0, 0],
                "text": table_text,
                "page_num": page_num,
                "source": source,
                "score": 0.9,
                "meta": {
                    "rows": rows,
                    "block_subtype": "hwp_record_table",
                    "table_shape": {"rows": len(rows), "cols": col_count},
                    "structure_reason": "hwp_record_control_with_cols",
                }
            })
            tables.append(rows)
            block_idx += 1
            stats["table_count"] += 1
        record_table_rows.clear()

    def _emit_record_table():
        nonlocal block_idx
        if not record_table_rows:
            return
        guessed_cols = _guess_record_table_cols(record_table_rows)
        if guessed_cols >= 2:
            rows: list[list[str]] = []
            for i in range(0, len(record_table_rows), guessed_cols):
                row = record_table_rows[i:i + guessed_cols]
                if len(row) < guessed_cols:
                    row.extend([""] * (guessed_cols - len(row)))
                rows.append(row)
            if rows:
                table_text = _table_text_from_rows(rows)
                blocks.append({
                    "id": f"p{page_num}_b{block_idx}",
                    "type": "table",
                    "bbox": [0, 0, 0, 0],
                    "text": table_text,
                    "page_num": page_num,
                    "source": source,
                    "score": 0.9,
                    "meta": {
                        "rows": rows,
                        "block_subtype": "hwp_record_table",
                        "table_shape": {"rows": len(rows), "cols": guessed_cols},
                        "structure_reason": "hwp_record_control",
                    }
                })
                tables.append(rows)
                block_idx += 1
                stats["table_count"] += 1
        else:
            body = "\n".join(record_table_rows)
            if body.strip():
                current_body_lines.append(body)
        record_table_rows.clear()

    for line in lines:
        stripped = line.strip()

        if stripped == "<TABLE_START>":
            _flush_table()
            _flush_data()
            _flush_body()
            in_record_table = True
            record_table_cells.clear()
            record_table_rows.clear()
            record_table_col_count = 0
            stats.setdefault("record_table_start_count", 0)
            stats["record_table_start_count"] += 1
            continue
        if stripped == "<TABLE_END>":
            _flush_record_cell()
            if record_table_col_count > 0 and record_table_rows:
                _emit_record_table_with_cols(record_table_rows, record_table_col_count)
            elif record_table_rows:
                _emit_record_table()
            in_record_table = False
            record_table_rows.clear()
            continue
        cols_match = re.match(r"<TABLE_COLS:(\d+)>", stripped)
        if cols_match:
            record_table_col_count = int(cols_match.group(1))
            continue
        if stripped == "<CELL_BREAK>":
            if in_record_table:
                _flush_record_cell()
            continue

        if in_record_table:
            if stripped:
                record_table_cells.append(stripped)
            continue

        is_header = _is_doc_section_header(line)
        is_data = _is_data_row(line)
        is_table_like = _looks_table_like_line(line)
        if "\t" in line:
            stats["tab_hint_line_count"] += 1
        is_list_item = len(stripped) < 200 and re.match(r'^\s*([□◦\-\u25e6\u2022\u25a1\u25cb\u2023\u25a0]|[\d]{1,2}[\.\)]|[Ⅰ-Ⅹ][\.\)]|[가-힣][\.\)])', line)

        if pending_table_break and stripped and not is_table_like:
            _flush_table()
        if pending_table_break and stripped and is_table_like:
            pending_table_break = False
        elif stripped:
            pending_table_break = False

        if is_header:
            _flush_table()
            _flush_data()
            _flush_body()
            blocks.append({
                "id": f"p{page_num}_b{block_idx}",
                "type": "title",
                "bbox": [0, 0, 0, 0],
                "text": line.strip(),
                "page_num": page_num,
                "source": source,
                "score": 1.0,
                "meta": {}
            })
            block_idx += 1
            stats["title_count"] += 1
        elif not stripped:
            # 빈 줄: 블록 단절 생성
            if current_table_rows:
                pending_table_break = True
                continue
            _flush_data()
            _flush_body()
        elif is_table_like:
            _flush_data()
            _flush_body()
            row_cells = _split_tabular_cells(line)
            if _is_compact_label_value_row(row_cells):
                stats["compact_table_row_count"] += 1
            if _is_summary_pair_row(row_cells):
                stats["summary_pair_row_count"] += 1
            current_table_rows.append(row_cells)
            stats["table_like_line_count"] += 1
        elif is_list_item:
            # 목록 아이템 시작: 이전 버퍼 비우고 새 블록 유도
            _flush_table()
            _flush_data()
            _flush_body()
            blocks.append({
                "id": f"p{page_num}_b{block_idx}",
                "type": "text",
                "bbox": [0, 0, 0, 0],
                "text": line.strip(),
                "page_num": page_num,
                "source": source,
                "score": 1.0,
                "meta": {"block_subtype": "list_item"}
            })
            block_idx += 1
            stats["text_count"] += 1
        elif is_data:
            # 데이터 행: 바디 버퍼가 있으면 비우고 데이터 버퍼로
            _flush_table()
            if current_body_lines:
                _flush_body()
            current_data_lines.append(line)
        else:
            # 일반 텍스트: 데이터 버퍼가 있으면 비우고 바디 버퍼로
            _flush_table()
            if current_data_lines:
                _flush_data()
            current_body_lines.append(line)

    _flush_table()
    _flush_data()
    _flush_body()
    stats["title_count"] = sum(1 for block in blocks if block["type"] == "title")
    stats["text_count"] = sum(1 for block in blocks if block["type"] == "text")
    return blocks, tables, stats


def _split_into_section_blocks(text: str, page_num: int) -> list[dict[str, Any]]:
    blocks, _, _ = _structure_text_blocks(text, page_num, "hwp_parser")
    return blocks


def _guess_record_table_cols(flat_cells: list[str]) -> int:
    """Heuristically guess column count from a flat list of cell values."""
    n = len(flat_cells)
    if n < 2:
        return 0

    for cols in range(2, min(n + 1, 15)):
        if n % cols != 0:
            continue
        rows = n // cols
        if rows < 1:
            continue
        first_row = flat_cells[:cols]
        label_count = sum(1 for c in first_row if not re.search(r"\d{2,}", c) and len(c) <= 30)
        if label_count >= cols * 0.5:
            return cols

    for cols in range(2, min(n + 1, 15)):
        if n % cols == 0 and n // cols >= 2:
            return cols

    return 0


def _label_cover_blocks(blocks: list[dict[str, Any]], page_num: int) -> None:
    """Label cover/first-page blocks with cover subtypes for better structuring."""
    if page_num != 1 or len(blocks) < 2:
        return

    for i, block in enumerate(blocks[:8]):
        text = str(block.get("text", "") or "").strip()
        meta = block.setdefault("meta", {})
        if not text:
            continue

        if i == 0 and block["type"] in ("title", "text") and len(text) <= 120:
            meta["block_subtype"] = "cover_title"
            if block["type"] == "text":
                block["type"] = "title"
            continue

        if re.search(r"(20\d{2})\s*[년.]?\s*(\d{1,2})\s*[월.]", text):
            meta["block_subtype"] = "cover_meta"
            continue
        if re.search(r"(사업보고서|실적보고서|분기보고서|반기보고서|연차보고서)", text):
            meta["block_subtype"] = "cover_title"
            if block["type"] == "text":
                block["type"] = "title"
            continue
        if re.search(r"(주식회사|회사명|제출인|발행인|법인명|대표이사)", text) and len(text) <= 80:
            meta["block_subtype"] = "cover_meta"
            continue


def _generate_hwp_rag_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        btype = block.get("type")
        text = str(block.get("text", "") or "").strip()
        meta = block.get("meta", {})
        subtype = meta.get("block_subtype", "")
        if not text:
            continue
        if subtype == "cover_title":
            parts.append(f"\n[COVER: {text}]\n")
        elif subtype == "cover_meta":
            parts.append(f"[META] {text}")
        elif btype == "title":
            parts.append(f"\n[SECTION: {text}]\n")
        elif btype == "table":
            rows = meta.get("rows", [])
            row_count = len(rows)
            col_count = max((len(row) for row in rows), default=0)
            context_title = meta.get("context_title")
            if context_title:
                parts.append(f"[TABLE CONTEXT] {context_title}")
            if row_count <= 20:
                parts.append(f"\n[TABLE: {row_count} rows, {col_count} cols]\n{text}\n")
            else:
                parts.append(f"\n[TABLE: {row_count} rows, {col_count} cols]\n{text[:400]}\n")
        elif subtype == "meta_info":
            parts.append(f"[META] {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()

def parse_hwp(filepath: str) -> dict[str, Any]:
    if not olefile.isOleFile(filepath):
        raise ValueError(f"Not a valid OLE2/HWP file: {filepath}")

    ole = olefile.OleFileIO(filepath)
    try:
        metadata = _read_metadata(ole)
        is_compressed = _check_compressed(ole)
        metadata["is_compressed"] = is_compressed

        sections: list[str] = []
        empty_sections = 0
        section_idx = 0

        while True:
            stream_name = f"BodyText/Section{section_idx}"
            if not ole.exists(stream_name):
                break
            try:
                raw = ole.openstream(stream_name).read()
                data = _decompress(raw, is_compressed)
                text = _normalise(_parse_section_records(data))
                if text:
                    sections.append(text)
                else:
                    empty_sections += 1
            except Exception as exc:
                logger.warning("HWP Section%d failed: %s", section_idx, exc)
                empty_sections += 1
            section_idx += 1
    finally:
        ole.close()

    pages: list[dict[str, Any]] = []
    for i, text in enumerate(sections):
        section_blocks, section_tables, structure_stats = _structure_text_blocks(text, i + 1, "hwp_parser")
        _label_cover_blocks(section_blocks, i + 1)
        pages.append({
            "page_num": i + 1,
            "page_width": 0,
            "page_height": 0,
            "preview_width": 0,
            "preview_height": 0,
            "preview_scale_x": 1.0,
            "preview_scale_y": 1.0,
            "coord_space": "page_points",
            "preview_image": None,
            "text": text,
            "tables": section_tables,
            "blocks": section_blocks,
            "image_count": 0,
            "text_source": "native",
            "ocr_applied": False,
            "ocr_confidence": 0.0,
            "parser_debug": {
                "preview_generated": False,
                "preview_error": None,
                "native_text_chars": len(text),
                "ocr_used": False,
                "ocr_trigger_reason": "ocr_not_needed",
                "candidate_counts": {
                    "raw_text_blocks": 1 if text else 0,
                    "final_blocks": len(section_blocks)
                },
                "block_type_counts": {
                    "title": sum(1 for b in section_blocks if b["type"] == "title"),
                    "text": sum(1 for b in section_blocks if b["type"] == "text"),
                    "table": sum(1 for b in section_blocks if b["type"] == "table")
                },
                "dropped_blocks": [],
                "bbox_warnings": [],
                "narrative_analysis": _analyze_narrative_characteristics(text),
                "structure_extraction": structure_stats,
            }
        })

    metadata.update({
        "parser_used": "custom OLE HWP parser",
        "page_count": len(pages),
        "section_count": section_idx,
        "empty_sections": empty_sections,
    })
    
    if pages:
        metadata["narrative_analysis"] = pages[0]["parser_debug"]["narrative_analysis"]

    if not pages:
        return {
            "pages": [],
            "metadata": metadata,
            "status": "error",
            "error": f"No text extracted from {section_idx} section(s) ({empty_sections} empty)",
        }

    # ── Document Routing ──────────────────────────────
    from .document_router import route_document
    routing_result = route_document(None, pages, metadata=metadata)
    
    doc_type = routing_result.get("document_type", "text_report")
    _align_hwp_metadata(metadata, pages, routing_result, parser_subtype=str(metadata.get("parser_used", "custom OLE HWP parser")))
    
    for p in pages:
        dbg = p.setdefault("parser_debug", {})
        dbg["document_type"] = doc_type
        dbg["refined_document_type"] = doc_type
        dbg["doc_type"] = doc_type
        dbg["pipeline_used"] = metadata.get("pipeline_used")
        p["rag_text"] = _generate_hwp_rag_text(p.get("blocks", []))

    return {"pages": pages, "metadata": metadata, "status": "success"}


# ── internals ────────────────────────────────────────────────────────

def _check_compressed(ole: olefile.OleFileIO) -> bool:
    if not ole.exists("FileHeader"):
        return False
    header = ole.openstream("FileHeader").read()
    return bool(header[36] & 0x01) if len(header) > 36 else False


def _read_metadata(ole: olefile.OleFileIO) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        summary = ole.get_metadata()
        if summary:
            for key in ("title", "author", "subject", "last_saved_by"):
                val = getattr(summary, key, None)
                meta[key] = str(val) if val else ""
    except Exception:
        pass
    return meta


def _decompress(data: bytes, is_compressed: bool) -> bytes:
    if not is_compressed:
        return data
    try:
        return zlib.decompress(data, -15)
    except zlib.error:
        return zlib.decompress(data)


def _parse_section_records(data: bytes) -> str:
    parts: list[str] = []
    offset = 0
    length = len(data)
    in_table = False
    table_para_count = 0
    table_col_count = 0

    while offset < length:
        if offset + 4 > length:
            break

        header = struct.unpack_from("<I", data, offset)[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        offset += 4

        if size == 0xFFF:
            if offset + 4 > length:
                break
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4

        if offset + size > length:
            break

        if tag_id == HWPTAG_CTRL_HEADER and size >= 4:
            ctrl_id_bytes = data[offset: offset + 4]
            if len(ctrl_id_bytes) == 4:
                ctrl_id = ctrl_id_bytes[::-1]
                if ctrl_id == b"tbl ":
                    if in_table:
                        parts.append(_TABLE_END_MARKER)
                    parts.append(_TABLE_START_MARKER)
                    in_table = True
                    table_para_count = 0
                    table_col_count = 0

        if tag_id == HWPTAG_TABLE and in_table and size >= 8:
            try:
                nrows = struct.unpack_from("<H", data, offset + 2)[0]
                ncols = struct.unpack_from("<H", data, offset + 4)[0]
                table_col_count = ncols
                parts.append(f"\n<TABLE_COLS:{ncols}>\n")
            except (struct.error, IndexError):
                pass

        if tag_id == HWPTAG_PARA_TEXT:
            decoded = _decode_para_text(data[offset: offset + size])
            if in_table:
                parts.append(_CELL_BREAK_MARKER)
                table_para_count += 1
            parts.append(decoded)

        offset += size

    if in_table:
        parts.append(_TABLE_END_MARKER)

    return "\n".join(t for t in parts if t.strip())


def _decode_para_text(record: bytes) -> str:
    chars: list[str] = []
    i = 0
    end = len(record) - 1

    while i < end:
        code = struct.unpack_from("<H", record, i)[0]
        i += 2

        if code < 32:
            if code in EXTENDED_CONTROLS:
                i += 14
            elif code == 9:
                chars.append("\t")
            elif code in (10, 13):
                chars.append("\n")
        else:
            chars.append(chr(code))

    return "".join(chars)
