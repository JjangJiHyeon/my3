import logging
import re
import statistics
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# Region seeds: bbox clustering only — no table geometry, no kpi/mini_table/footer (avoids mega-regions).
_REGION_SEED_ROLES = frozenset({"title", "card_text", "issue_box", "ranking"})
# Lines allowed to merge into a single region text block (narrative / card copy only).
_REGION_MERGE_ROLES = frozenset({"card_text", "issue_box", "ranking"})


def process_dashboard_brief(doc, page_idx, plumber_tables):
    """
    Dashboard extraction favors card-level grouping over line-level narrative merging.
    KPI fragments, ranking rows, issue boxes, and mini tables are separated early.
    """
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    raw = page.get_text("dict") or {}

    table_blocks = _build_table_blocks(plumber_tables, page_idx, pw, ph)
    line_candidates, numeric_dropped = _extract_dashboard_lines(
        raw, page_idx, pw, ph, table_blocks
    )
    visual_blocks, visual_dropped = _extract_visual_blocks(raw, page_idx, pw, ph)
    regions, region_seed_count = _segment_dashboard_regions(line_candidates, table_blocks, pw, ph)
    region_assignment_debug = _assign_dashboard_region_ids(line_candidates, regions, pw, ph)
    table_context_debug = _enrich_table_blocks_with_context(table_blocks, line_candidates, pw, ph)
    visual_context_debug = _enrich_visual_blocks_with_context(visual_blocks, line_candidates, table_blocks, pw, ph)
    region_blocks, absorbed_count, region_debug = _build_dashboard_region_blocks(
        line_candidates,
        regions,
        page_idx,
        table_blocks,
        visual_blocks,
        pw,
        ph,
    )
    region_debug["dashboard_region_seed_count"] = region_seed_count
    region_debug.update(getattr(_segment_dashboard_regions, "_last_stats", {}))
    region_debug.update(region_assignment_debug)
    region_debug.update(table_context_debug)
    region_debug.update(visual_context_debug)

    blocks = table_blocks + _sort_dashboard_final_blocks(region_blocks) + visual_blocks
    dashboard_roles = sorted({
        block.get("meta", {}).get("dashboard_role", "unknown")
        for block in region_blocks
    })

    return {
        "page_num": page_idx + 1,
        "page_width": pw,
        "page_height": ph,
        "blocks": blocks,
        "_pipeline_postprocess": _postprocess_dashboard_blocks,
        "parser_debug": {
            "pipeline_used": "dashboard_brief_pipeline",
            "dashboard_signals": ["card_region_grouping", "kpi_fragment_filtering"],
            "segmented_regions": len(regions),
            "dashboard_block_types": dashboard_roles,
            "fragment_absorbed_count": absorbed_count,
            "numeric_fragment_dropped_count": numeric_dropped,
            "decorative_visual_dropped_count": visual_dropped,
            "preferred_layout_hint": "mixed_visual",
            **region_debug,
        },
    }


def _extract_dashboard_lines(
    raw: dict,
    page_idx: int,
    pw: float,
    ph: float,
    table_blocks: list[dict] | None = None,
) -> tuple[list[dict], int]:
    table_blocks = table_blocks or []
    candidates: list[dict] = []
    numeric_dropped = 0

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not text:
                continue

            bbox = list(line.get("bbox", [0, 0, 0, 0]))
            width = bbox[2] - bbox[0]
            font_size = max((float(span.get("size", 0.0) or 0.0) for span in spans), default=0.0)
            if _is_tiny_numeric_fragment(text, width, font_size):
                numeric_dropped += 1
                continue

            role = _classify_dashboard_line(text, bbox, pw, ph, font_size)
            if role == "decorative_noise":
                numeric_dropped += 1
                continue

            block_type = "title" if role == "title" else "footer" if role == "footer" else "text"
            candidates.append({
                "id": f"p{page_idx+1}_d{len(candidates)}",
                "type": block_type,
                "bbox": bbox,
                "text": text,
                "source": "pymupdf_dashboard_native",
                "confidence": 1.0,
                "meta": {
                    "dashboard_role": role,
                    "font_size": round(font_size, 2),
                    "summary_priority": "high" if role in ("title", "kpi", "issue_box", "ranking", "mini_table") else "medium",
                    "summary_exclude": role in ("footer", "decorative_noise"),
                },
            })

    _enrich_dashboard_line_meta(candidates, table_blocks, pw, ph)
    return candidates, numeric_dropped


def _extract_visual_blocks(raw: dict, page_idx: int, pw: float, ph: float) -> tuple[list[dict], int]:
    visuals: list[dict] = []
    dropped = 0
    for block in raw.get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = list(block.get("bbox", [0, 0, 0, 0]))
        x0, y0, x1, y1 = bbox
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        area_ratio = (width * height) / max(1.0, pw * ph)

        if _is_decorative_visual_fragment(width, height, area_ratio, pw, ph):
            dropped += 1
            continue

        visuals.append({
            "id": f"p{page_idx+1}_img{len(visuals)}",
            "type": "image",
            "bbox": bbox,
            "text": "",
            "source": "pymupdf_dashboard_image",
            "confidence": 1.0,
            "meta": {
                "dashboard_role": "visual_tile",
                "summary_priority": "low",
                "visual_area_ratio": round(area_ratio, 5),
                "visual_fragment_count": 1,
            },
        })
    return visuals, dropped


def _build_table_blocks(plumber_tables, page_idx: int, pw: float, ph: float) -> list[dict]:
    tables: list[dict] = []
    for i, tbl in enumerate(plumber_tables):
        cells = _normalize_table_cells(tbl.get("cells") or [])
        flattened = [
            " | ".join(str(cell).strip() for cell in row if cell is not None and str(cell).strip())
            for row in cells
        ]
        text = "\n".join(row for row in flattened if row).strip() or "[Table Data]"
        table_summary, summary_kind, key_value_rows = _summarize_table_cells(cells)
        table_markdown = _table_markdown_from_cells(cells)
        row_count = len(cells)
        col_count = max((len(row) for row in cells), default=0)
        normalized_table = {
            "title": _first_nonempty_table_row(cells),
            "headers": _infer_table_headers(cells),
            "rows": cells,
            "header_rows": [0] if cells else [],
            "shape": {"rows": row_count, "cols": col_count},
            "source": "pdfplumber_dashboard",
            "markdown": table_markdown,
        }
        tables.append({
            "id": f"p{page_idx+1}_t{i}",
            "type": "table",
            "bbox": list(tbl.get("bbox", [0, 0, pw, ph])),
            "text": text,
            "source": "pdfplumber_dashboard",
            "confidence": 1.0,
            "meta": {
                "dashboard_role": "mini_table",
                "normalized_table": normalized_table,
                "table_summary": table_summary,
                "table_markdown": table_markdown,
                "key_value_rows": key_value_rows,
                "dashboard_table_summary_kind": summary_kind,
                "dashboard_table_shape": {"rows": row_count, "cols": col_count},
                "summary_priority": "high",
            },
        })
    return tables


# ── required structural helpers (chunking / RAG oriented) ─────────────

def _clean_table_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _normalize_table_cells(cells: list[Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    max_cols = 0
    for row in cells or []:
        if isinstance(row, (list, tuple)):
            cleaned = [_clean_table_cell(cell) for cell in row]
        else:
            cleaned = [_clean_table_cell(row)]
        while cleaned and not cleaned[-1]:
            cleaned.pop()
        if not cleaned:
            continue
        rows.append(cleaned)
        max_cols = max(max_cols, len(cleaned))
    if max_cols <= 0:
        return rows
    return [row + [""] * (max_cols - len(row)) for row in rows]


def _first_nonempty_table_row(cells: list[list[str]]) -> str:
    for row in cells:
        joined = " | ".join(cell for cell in row if cell).strip()
        if joined:
            return joined
    return ""


def _infer_table_headers(cells: list[list[str]]) -> list[str]:
    for row in cells[:3]:
        nonempty = [cell for cell in row if cell]
        if len(nonempty) >= 2:
            return nonempty
    return [cell for cell in cells[0] if cell] if cells else []


def _table_markdown_from_cells(cells: list[list[str]], max_rows: int = 18, max_cols: int = 10) -> str:
    if not cells:
        return ""
    width = min(max((len(row) for row in cells), default=0), max_cols)
    if width <= 0:
        return ""

    def _row(values: list[str]) -> str:
        clipped = [(values[i] if i < len(values) else "").replace("|", "/") for i in range(width)]
        return "| " + " | ".join(clipped) + " |"

    rows = [_row(cells[0])]
    rows.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in cells[1:max_rows]:
        rows.append(_row(row))
    if len(cells) > max_rows:
        rows.append("| " + " | ".join("..." if i == 0 else "" for i in range(width)) + " |")
    return "\n".join(rows)


def _summarize_table_cells(cells: list[list[str]], max_rows: int = 12) -> tuple[str, str, list[dict[str, str]]]:
    if not cells:
        return "Mini-table: no structured cells extracted.", "empty", []

    row_count = len(cells)
    col_count = max((len(row) for row in cells), default=0)
    title = _first_nonempty_table_row(cells)
    headers = _infer_table_headers(cells)
    key_value_rows: list[dict[str, str]] = []
    body_start = 1 if headers and cells and headers == [cell for cell in cells[0] if cell] else 0

    for row in cells[body_start : body_start + max_rows]:
        nonempty = [cell for cell in row if cell]
        if len(nonempty) < 2:
            continue
        key = nonempty[0]
        value = " | ".join(nonempty[1:])
        if key and value:
            key_value_rows.append({"key": key, "value": value})

    lines = [f"Mini-table ({row_count} rows x {col_count} cols)"]
    if title:
        lines.append(f"Title/first row: {title}")
    if headers:
        lines.append("Columns: " + " | ".join(headers[: min(len(headers), 10)]))
    if key_value_rows:
        for item in key_value_rows[:max_rows]:
            lines.append(f"- {item['key']}: {item['value']}")
        return "\n".join(lines), "key_value", key_value_rows

    for row in cells[:max_rows]:
        joined = " | ".join(cell for cell in row if cell)
        if joined:
            lines.append(f"- {joined}")
    return "\n".join(lines), "row_excerpt", key_value_rows


def _looks_short_metric_value(text: str, width: float = 0.0, font_size: float = 0.0) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 32:
        return False
    if not re.search(r"\d", t):
        return False
    metric_signal = bool(re.search(r"[%+\-]|bp|bps|x\b", t, re.I))
    if not metric_signal:
        return False
    return width >= 12.0 or font_size >= 8.0 or len(t) >= 4


def _looks_bracketed_topic_start(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(re.match(r"^\[[^\]\n]{1,24}\]\s*\S", t))


def _looks_structural_issue_box(text: str) -> bool:
    t = (text or "").strip()
    if _looks_bracketed_topic_start(t):
        return True
    if len(t) <= 80 and re.match(r"^[A-Za-z][A-Za-z0-9 /\-&]{1,40}:$", t):
        return True
    return False


def _looks_structural_ranking(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if re.match(r"^\d{1,2}[\.)]\s+\S", t):
        return True
    if re.match(r"^[\-*]\s+\S", t):
        return True
    return False


def _looks_structural_mini_table(text: str, width: float, page_width: float) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.count("|") >= 1 and width > page_width * 0.12:
        return True
    if len(re.findall(r"\b\d{1,2}[QH]\d{2,4}[A-Z]?\b", t, re.I)) >= 2:
        return True
    if len(re.findall(r"\b(?:1D|1W|1M|3M|6M|12M)\b", t, re.I)) >= 2:
        return True
    return False


def _alpha_ratio(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    return len(re.findall(r"[A-Za-z가-힣]", t)) / max(len(t), 1)


def _numeric_ratio(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    return len(re.findall(r"\d", t)) / max(len(t), 1)


def _symbol_ratio(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    return len(re.findall(r"[\|%\-+,\./\(\)]", t)) / max(len(t), 1)


def _looks_contact_info(text: str) -> bool:
    t = text or ""
    if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", t):
        return True
    if re.search(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}",
        t,
    ):
        return True
    return False


def _looks_axis_tick(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(re.search(r"\d{1,2}/\d{1,2}", t)) and len(t) <= 36


def _looks_unit_label(text: str) -> bool:
    t = (text or "").strip()
    if len(t) <= 22 and re.fullmatch(r"\([^)]{1,18}\)\s*", t):
        return True
    return len(t) <= 12 and _symbol_ratio(t) >= 0.2 and _numeric_ratio(t) < 0.35


def _line_center_x(line: dict) -> float:
    bb = line["bbox"]
    return (bb[0] + bb[2]) / 2.0


def _line_width_ratio(line: dict, page_width: float) -> float:
    bb = line["bbox"]
    return (bb[2] - bb[0]) / max(page_width, 1.0)


def _bbox_intersection_area(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _line_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
    ia = _bbox_intersection_area(a, b)
    aa = _line_area(a)
    ba = _line_area(b)
    return ia / max(min(aa, ba), 1e-6)


def _merge_rects(
    rects: list[tuple[float, float, float, float]],
    x_gap: float,
    y_gap: float,
) -> list[tuple[float, float, float, float]]:
    if not rects:
        return []
    rs = sorted(rects, key=lambda r: (r[1], r[0]))
    cur = [rs[0][0], rs[0][1], rs[0][2], rs[0][3]]
    out: list[tuple[float, float, float, float]] = []
    for r in rs[1:]:
        gaph = max(0.0, max(cur[0] - r[2], r[0] - cur[2]))
        gapv = max(0.0, max(cur[1] - r[3], r[1] - cur[3]))
        if gaph <= x_gap and gapv <= y_gap:
            cur[0] = min(cur[0], r[0])
            cur[1] = min(cur[1], r[1])
            cur[2] = max(cur[2], r[2])
            cur[3] = max(cur[3], r[3])
        else:
            out.append((cur[0], cur[1], cur[2], cur[3]))
            cur = [r[0], r[1], r[2], r[3]]
    out.append((cur[0], cur[1], cur[2], cur[3]))
    return out


def _line_zone_overlap_frac(line_bbox: list[float], zone: list[float]) -> float:
    return _bbox_overlap_ratio(line_bbox, zone)


def _line_max_zone_overlap_frac(line_bbox: list[float], zones: list[list[float]]) -> float:
    if not zones:
        return 0.0
    return max(_line_zone_overlap_frac(line_bbox, z) for z in zones)


def _narrative_merge_excluded_line(text: str) -> bool:
    """Heuristic axis/table debris — pipes, tick density, short numeric/symbol lines."""
    t = text.strip()
    if not t:
        return True
    if t.count("|") >= 2:
        return True
    tick_matches = re.findall(r"\d{1,2}/\d{1,2}", t)
    if len(tick_matches) >= 2:
        return True
    if len(tick_matches) >= 1 and len(t) <= 40:
        return True
    if len(t) <= 20 and re.fullmatch(r"\([^\)]+\)\s*", t):
        return True
    letters = len(re.findall(r"[A-Za-z가-힣]", t))
    digits = len(re.findall(r"\d", t))
    symbols = len(re.findall(r"[\|%\-+,\./]", t))
    ln = max(len(t), 1)
    if len(t) < 52 and (digits + symbols) / ln >= 0.5 and letters < max(6, len(t) // 5):
        return True
    return False


def _axis_tick_signal_count(text: str) -> int:
    return len(re.findall(r"\d{1,2}/\d{1,2}", text)) + (2 if text.count("|") >= 2 else 0)


def _narrative_alpha_ratio(text: str) -> float:
    t = text.strip()
    if not t:
        return 0.0
    letters = len(re.findall(r"[A-Za-z가-힣]", t))
    return letters / max(len(t), 1)


def _band_id(cy: float, ph: float) -> int:
    if ph <= 0:
        return 0
    return max(0, min(24, int(cy / max(ph * 0.125, 1.0))))


def _col_id(cx: float, pw: float) -> int:
    if pw <= 0:
        return 0
    b = int(cx / max(pw / 3.0, 1.0))
    return max(0, min(2, b))


def _digit_symbol_ratio(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    d = len(re.findall(r"\d", t))
    s = len(re.findall(r"[\|%\-+,\./]", t))
    return (d + s) / max(len(t), 1)


def _inflate_bbox(bbox: list[float], mx: float, my: float) -> list[float]:
    return [bbox[0] - mx, bbox[1] - my, bbox[2] + mx, bbox[3] + my]


def _union_bboxes(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _bbox_gap(b1: list[float], b2: list[float]) -> tuple[float, float]:
    h_gap = max(0.0, max(b1[0] - b2[2], b2[0] - b1[2]))
    v_gap = max(0.0, max(b1[1] - b2[3], b2[1] - b1[3]))
    return h_gap, v_gap


def _zone_area_ratio_on_page(zone: list[float], pw: float, ph: float) -> float:
    w = max(0.0, zone[2] - zone[0])
    h = max(0.0, zone[3] - zone[1])
    return (w * h) / max(pw * ph, 1e-6)


def _spatial_line_clusters(
    line_candidates: list[dict], pw: float, ph: float
) -> list[list[dict]]:
    pool = [ln for ln in line_candidates if ln.get("type") in ("text", "title", "footer")]
    sorted_pool = sorted(pool, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    clusters: list[list[dict]] = []
    for ln in sorted_pool:
        bb = ln["bbox"]
        placed = False
        for cl in clusters:
            u = _union_bboxes([x["bbox"] for x in cl])
            hg, vg = _bbox_gap(bb, u)
            if vg < ph * 0.026 and hg < pw * 0.078:
                cl.append(ln)
                placed = True
                break
        if not placed:
            clusters.append([ln])
    return clusters


def _build_dashboard_exclusion_zones(
    line_candidates: list[dict],
    table_blocks: list[dict],
    page_width: float,
    page_height: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Hard = tables + dense right / bottom regions (separate rects, never merged across bands)."""
    pw, ph = page_width, page_height
    mx_t = max(2.0, pw * 0.011)
    my_t = max(2.0, ph * 0.009)
    hard: list[list[float]] = []
    for t in table_blocks:
        bb = list(t.get("bbox", [0, 0, 0, 0]))
        if len(bb) >= 4:
            ib = _inflate_bbox(bb, mx_t, my_t)
            hard.append([float(ib[0]), float(ib[1]), float(ib[2]), float(ib[3])])

    clusters = _spatial_line_clusters(line_candidates, pw, ph)
    for cl in clusters:
        if len(cl) < 3:
            continue
        wrs = [_line_width_ratio(x, pw) for x in cl]
        nrs = [_numeric_ratio((x.get("text") or "")) for x in cl]
        mean_wr = statistics.fmean(wrs)
        mean_nr = statistics.fmean(nrs)
        short_frac = sum(1 for x in cl if len((x.get("text") or "").strip()) <= 22) / len(cl)
        if mean_wr > 0.38:
            continue
        if not (mean_nr >= 0.26 or short_frac >= 0.45):
            continue
        u = _union_bboxes([x["bbox"] for x in cl])
        cx = (u[0] + u[2]) / 2.0
        y_top = u[1]
        y_ctr = (u[1] + u[3]) / 2.0
        in_right = cx >= 0.52 * pw
        in_bottom_strip = y_top >= 0.66 * ph
        if in_right and y_ctr < 0.67 * ph:
            ir = _inflate_bbox(u, pw * 0.008, ph * 0.006)
            hard.append([float(ir[0]), float(ir[1]), float(ir[2]), float(ir[3])])
        elif in_bottom_strip:
            ir = _inflate_bbox(u, pw * 0.006, ph * 0.005)
            hard.append([float(ir[0]), float(ir[1]), float(ir[2]), float(ir[3])])

    soft: list[list[float]] = []
    for cl in clusters:
        if len(cl) < 3:
            continue
        mean_nr = statistics.fmean([_numeric_ratio((x.get("text") or "")) for x in cl])
        mean_wr = statistics.fmean([_line_width_ratio(x, pw) for x in cl])
        if mean_nr < 0.30 or mean_wr > 0.36:
            continue
        u = _union_bboxes([x["bbox"] for x in cl])
        cx = (u[0] + u[2]) / 2.0
        if cx < 0.50 * pw:
            continue
        ir = _inflate_bbox(u, pw * 0.004, ph * 0.003)
        soft.append([float(ir[0]), float(ir[1]), float(ir[2]), float(ir[3])])

    for zl in (hard, soft):
        i = 0
        while i < len(zl):
            if _zone_area_ratio_on_page(zl[i], pw, ph) > 0.40:
                zl.pop(i)
                continue
            i += 1

    return hard, soft


def _build_visual_exclusion_zones(
    line_candidates: list[dict],
    table_blocks: list[dict],
    page_width: float,
    page_height: float,
) -> list[list[float]]:
    h, s = _build_dashboard_exclusion_zones(
        line_candidates, table_blocks, page_width, page_height
    )
    return h + s


def _assign_dashboard_column_band(
    line: dict, page_width: float, page_height: float
) -> tuple[str, str]:
    bb = line["bbox"]
    cx = (bb[0] + bb[2]) / 2.0
    y0 = bb[1]
    if cx < 0.48 * page_width:
        col = "left"
    elif cx > 0.58 * page_width:
        col = "right"
    else:
        col = "center"
    if y0 < 0.22 * page_height:
        band = "top"
    elif y0 > 0.72 * page_height:
        band = "bottom"
    else:
        band = "middle"
    return col, band


def _line_table_overlap_max(line: dict, table_blocks: list[dict]) -> float:
    best = 0.0
    for t in table_blocks:
        bb = list(t.get("bbox", [0, 0, 0, 0]))
        if len(bb) >= 4:
            best = max(best, _bbox_overlap_ratio(line["bbox"], bb))
    return best


def _assign_dashboard_region_ids(
    lines: list[dict],
    regions: list[list[float]],
    pw: float,
    ph: float,
) -> dict[str, Any]:
    assigned = 0
    for ln in lines:
        meta = ln.setdefault("meta", {})
        best_idx = None
        best_score = 0.0
        for idx, region in enumerate(regions):
            inflated = _inflate_bbox(region, pw * 0.012, ph * 0.008)
            overlap = _bbox_overlap_ratio(ln["bbox"], inflated)
            bb = ln["bbox"]
            cx = (bb[0] + bb[2]) / 2.0
            cy = (bb[1] + bb[3]) / 2.0
            inside = inflated[0] <= cx <= inflated[2] and inflated[1] <= cy <= inflated[3]
            score = overlap + (0.25 if inside else 0.0)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score >= 0.20:
            meta["dashboard_region_id"] = f"r{best_idx}"
            meta["dashboard_region_overlap_score"] = round(best_score, 4)
            assigned += 1
    return {
        "dashboard_region_assignment_count": assigned,
        "dashboard_region_unassigned_line_count": max(0, len(lines) - assigned),
    }


def _line_context_text_ok(ln: dict) -> bool:
    meta = ln.get("meta", {})
    rr = meta.get("dashboard_role_refined")
    if rr in ("ignore_noise", "footer", "contact_info_like"):
        return False
    text = (ln.get("text") or "").strip()
    if not text:
        return False
    if len(text) <= 3 and _numeric_ratio(text) >= 0.34:
        return False
    if _numeric_ratio(text) >= 0.62 and _alpha_ratio(text) < 0.18:
        return False
    return True


def _unique_line_texts(lines: list[dict], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ln in sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        text = re.sub(r"\s+", " ", (ln.get("text") or "").strip())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _extract_metric_tokens(texts: list[str], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    metrics: list[str] = []
    for text in texts:
        for token in re.findall(r"[+\-]?\d[\d,]*(?:\.\d+)?%?|[+\-]?\d+(?:\.\d+)?x", text):
            norm = token.strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            metrics.append(norm)
            if len(metrics) >= limit:
                return metrics
    return metrics


def _enrich_table_blocks_with_context(
    table_blocks: list[dict],
    lines: list[dict],
    pw: float,
    ph: float,
) -> dict[str, Any]:
    enriched = 0
    context_line_total = 0
    for table in table_blocks:
        tb = list(table.get("bbox", [0, 0, 0, 0]))
        if len(tb) < 4:
            continue
        expanded = _inflate_bbox(tb, pw * 0.025, ph * 0.018)
        context_lines: list[dict] = []
        for ln in lines:
            if not _line_context_text_ok(ln):
                continue
            bb = ln["bbox"]
            cx = (bb[0] + bb[2]) / 2.0
            cy = (bb[1] + bb[3]) / 2.0
            overlaps = _bbox_overlap_ratio(bb, expanded) >= 0.18
            center_inside = expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]
            near_caption = (
                tb[0] - pw * 0.02 <= bb[0] <= tb[2] + pw * 0.02
                and abs(bb[3] - tb[1]) <= ph * 0.035
            )
            if overlaps or center_inside or near_caption:
                context_lines.append(ln)

        context_texts = _unique_line_texts(context_lines, limit=10)
        if not context_texts:
            continue
        meta = table.setdefault("meta", {})
        context = " / ".join(context_texts)
        existing = str(meta.get("table_summary") or "").strip()
        if context and context not in existing:
            meta["table_summary"] = f"Context: {context}\n{existing}".strip() if existing else f"Context: {context}"
        meta["caption_text"] = context
        meta["dashboard_table_context_text"] = context
        meta["dashboard_table_context_line_ids"] = [ln["id"] for ln in context_lines[:24]]
        meta["dashboard_table_context_line_count"] = len(context_lines)
        meta["summary_priority"] = "high"
        enriched += 1
        context_line_total += len(context_lines)
    return {
        "dashboard_table_context_enriched_count": enriched,
        "dashboard_table_context_line_count": context_line_total,
    }


def _enrich_visual_blocks_with_context(
    visual_blocks: list[dict],
    lines: list[dict],
    table_blocks: list[dict],
    pw: float,
    ph: float,
) -> dict[str, Any]:
    enriched = 0
    context_line_total = 0
    for visual in visual_blocks:
        vb = list(visual.get("bbox", [0, 0, 0, 0]))
        if len(vb) < 4:
            continue
        expanded = _inflate_bbox(vb, pw * 0.055, ph * 0.035)
        context_lines: list[dict] = []
        for ln in lines:
            if not _line_context_text_ok(ln):
                continue
            if _line_table_overlap_max(ln, table_blocks) >= 0.70:
                continue
            bb = ln["bbox"]
            cx = (bb[0] + bb[2]) / 2.0
            cy = (bb[1] + bb[3]) / 2.0
            h_gap, v_gap = _bbox_gap(bb, vb)
            center_inside = expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]
            near_edge = h_gap <= pw * 0.045 and v_gap <= ph * 0.030
            if center_inside or near_edge:
                context_lines.append(ln)

        context_texts = _unique_line_texts(context_lines, limit=12)
        if not context_texts:
            continue
        metrics = _extract_metric_tokens(context_texts)
        parts = ["Context: " + " / ".join(context_texts)]
        if metrics:
            parts.append("Visible metrics: " + ", ".join(metrics))
        summary = "\n".join(parts)
        meta = visual.setdefault("meta", {})
        meta["visual_summary"] = summary
        meta["caption_text"] = summary
        meta["context_block_ids"] = [ln["id"] for ln in context_lines[:24]]
        meta["visible_key_metrics"] = metrics
        meta["summary_priority"] = "medium"
        meta["dashboard_visual_context_line_count"] = len(context_lines)
        enriched += 1
        context_line_total += len(context_lines)
    return {
        "dashboard_visual_context_enriched_count": enriched,
        "dashboard_visual_context_line_count": context_line_total,
    }


def _has_prose_punctuation_hint(text: str) -> bool:
    """Structural sentence/continuation marks — not document-specific phrases."""
    t = text or ""
    return bool(re.search(r"[.。?？!！;；:：]|\.{2,}", t))


def _chart_label_signal_hits(line: dict, ov: float, pw: float) -> int:
    t = (line.get("text") or "").strip()
    n = 0
    if _numeric_ratio(t) >= 0.35:
        n += 1
    if _line_width_ratio(line, pw) <= 0.28:
        n += 1
    if _line_center_x(line) >= 0.55 * pw:
        n += 1
    if ov >= 0.18:
        n += 1
    if _looks_axis_tick(t) or _looks_unit_label(t):
        n += 1
    if len(t) <= 22:
        n += 1
    return n


def _assign_left_column_section_roles(candidates: list[dict], pw: float, ph: float) -> None:
    left_tm = [
        ln
        for ln in candidates
        if ln.get("meta", {}).get("dashboard_column_id") == "left"
        and ln.get("meta", {}).get("dashboard_band_id") in ("top", "middle")
        and ln.get("meta", {}).get("dashboard_role_refined") == "narrative_text"
        and ln.get("meta", {}).get("dashboard_role") != "title"
    ]
    left_tm.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    if len(left_tm) < 2:
        return
    clusters: list[list[dict]] = []
    for ln in left_tm:
        if not clusters:
            clusters.append([ln])
            continue
        prev = clusters[-1][-1]
        vg = ln["bbox"][1] - prev["bbox"][3]
        if vg > ph * 0.036:
            clusters.append([ln])
        else:
            clusters[-1].append(ln)
    for cl in clusters:
        if len(cl) < 2:
            continue
        first = cl[0]
        t = (first.get("text") or "").strip()
        fh = first["bbox"][3] - first["bbox"][1]
        # Stricter: only genuinely short, independent lines qualify as headers
        if len(t) > 35:
            continue
        if fh > ph * 0.022:
            continue
        if _alpha_ratio(t) < 0.22:
            continue
        # Long prose start guard: if first line is very close to next line,
        # it's a paragraph start, not a standalone header
        if len(cl) >= 2:
            gap_to_body = cl[1]["bbox"][1] - first["bbox"][3]
            if gap_to_body < ph * 0.008:
                continue
            # First line should be meaningfully shorter than body lines
            body_lens = [len((x.get("text") or "").strip()) for x in cl[1:]]
            if body_lens:
                avg_body = sum(body_lens) / len(body_lens)
                if avg_body > 0 and len(t) >= 0.6 * avg_body:
                    continue
        first.setdefault("meta", {})["dashboard_role_refined"] = "section_header_like"
        for rest in cl[1:]:
            rest.setdefault("meta", {})["dashboard_role_refined"] = "section_body_like"


def _enrich_dashboard_line_meta(
    candidates: list[dict],
    table_blocks: list[dict],
    pw: float,
    ph: float,
) -> None:
    hard_z, soft_z = _build_dashboard_exclusion_zones(candidates, table_blocks, pw, ph)
    bucket_w = max(pw * 0.035, 1.0)
    xbuckets: dict[int, list[dict]] = {}
    for ln in candidates:
        k = int(ln["bbox"][0] / bucket_w)
        xbuckets.setdefault(k, []).append(ln)
    list_like_keys: set[int] = set()
    for bk, items in xbuckets.items():
        if len(items) < 3:
            continue
        lens = [len((x.get("text") or "").strip()) for x in items]
        if statistics.median(lens) > 22:
            continue
        wrs = [_line_width_ratio(x, pw) for x in items]
        if statistics.median(wrs) > 0.34:
            continue
        if sum(1 for L in lens if L <= 22) < 3:
            continue
        ys = sorted((x["bbox"][1] + x["bbox"][3]) / 2.0 for x in items)
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        if len(gaps) < 2:
            continue
        try:
            gap_sd = statistics.pstdev(gaps)
        except statistics.StatisticsError:
            gap_sd = 0.0
        if gap_sd >= ph * 0.012:
            continue
        prose_heavy = sum(
            1
            for x in items
            if _alpha_ratio((x.get("text") or "")) >= 0.38
            and len((x.get("text") or "").strip()) >= 30
        )
        if prose_heavy / max(len(items), 1) > 0.35:
            continue
        list_like_keys.add(bk)

    short_nbr: dict[int, int] = {}
    for ln in candidates:
        bb = ln["bbox"]
        cy = (bb[1] + bb[3]) / 2.0
        c = 0
        for o in candidates:
            if o is ln:
                continue
            ob = o["bbox"]
            ocy = (ob[1] + ob[3]) / 2.0
            if abs(ocy - cy) > ph * 0.055:
                continue
            x_overlap = min(bb[2], ob[2]) - max(bb[0], ob[0])
            if x_overlap > -3.0 and len((o.get("text") or "").strip()) <= 30:
                c += 1
        short_nbr[id(ln)] = c

    for ln in candidates:
        col, band = _assign_dashboard_column_band(ln, pw, ph)
        meta = ln.setdefault("meta", {})
        meta["dashboard_column_id"] = col
        meta["dashboard_band_id"] = band
        ho = _line_max_zone_overlap_frac(ln["bbox"], hard_z)
        sov = _line_max_zone_overlap_frac(ln["bbox"], soft_z)
        meta["dashboard_hard_zone_overlap_frac"] = round(ho, 4)
        meta["dashboard_soft_zone_overlap_frac"] = round(sov, 4)
        meta["dashboard_zone_overlap_frac"] = round(max(ho, sov), 4)
        tb_ov = _line_table_overlap_max(ln, table_blocks)
        meta["dashboard_table_overlap_frac"] = round(tb_ov, 4)

        t = (ln.get("text") or "").strip()
        y0, y1 = ln["bbox"][1], ln["bbox"][3]
        wr = _line_width_ratio(ln, pw)
        cx = _line_center_x(ln)
        ar = _alpha_ratio(t)
        nr = _numeric_ratio(t)
        sr = _symbol_ratio(t)
        coarse = meta.get("dashboard_role", "card_text")
        font_size = float(meta.get("font_size", 0.0) or 0.0)
        bk_key = int(ln["bbox"][0] / bucket_w)
        cid = id(ln)

        # --- Left prose rescue guard (structural, no string matching) ---
        # Wide prose: typical narrative lines
        # Narrow alpha-heavy: shorter lines that are still meaningful prose (events, etc.)
        _is_left_prose = (
            col == "left"
            and band in ("top", "middle")
            and len(t) >= 15
            and (
                (ar >= 0.28 and wr >= 0.12)
                or (ar >= 0.45 and len(t) >= 18)
            )
        )
        _is_bottom_content_like = (
            y0 < ph * 0.94
            and ar >= 0.45
            and len(t) >= 16
            and wr >= 0.25
        )

        refined = "card_header_like"
        if _looks_contact_info(t):
            refined = "contact_info_like"
        elif (coarse == "footer" or (y1 > ph * 0.9 and len(t) <= 42)) and not _is_bottom_content_like:
            refined = "footer"
        # --- Bottom dense long prose → footer (compliance/disclaimer) ---
        elif y0 > ph * 0.84 and wr >= 0.62 and font_size <= 8.0 and ar >= 0.45 and len(t) >= 40 and nr <= 0.16:
            refined = "footer"
        elif coarse == "title":
            refined = "card_header_like"
        elif _looks_axis_tick(t) and len(t) <= 36 and not _is_left_prose:
            refined = "ignore_noise"
        elif _looks_unit_label(t) and not _is_left_prose:
            refined = "ignore_noise"
        elif len(t) <= 24 and nr + sr >= 0.75 and not _looks_short_metric_value(t, wr * pw, font_size):
            refined = "ignore_noise"
        elif coarse == "issue_box":
            refined = "narrative_text" if col == "left" else "card_header_like"
        elif coarse == "ranking":
            refined = "narrative_text" if col == "left" and band in ("top", "middle") and ar >= 0.25 else "list_item_like"
        elif coarse == "mini_table":
            refined = "table_header_like"
        elif (
            (band == "bottom" or tb_ov >= 0.12)
            and 0.20 <= wr <= 0.55
            and len(t.split()) >= 3
            and sum(1 for w in re.split(r"\s+", t) if len(w) <= 3) >= 3
            and not (ar >= 0.42 and len(t) >= 36 and nr <= 0.32)
            and not (ar >= 0.30 and len(t) >= 32 and t.count("|") == 0 and sr <= 0.24)
            and not (ar >= 0.28 and len(t) >= 32 and t.count("|") == 0 and "/" not in t)
            and not (wr >= 0.42 and cx <= 0.42 * pw and ar >= 0.5 and nr <= 0.35)
            and not (
                col == "left"
                and band in ("top", "middle")
                and wr >= 0.32
                and ar >= 0.35
            )
        ):
            refined = "table_header_like"
        elif _chart_label_signal_hits(ln, sov, pw) >= 2 and not _is_left_prose:
            refined = "chart_label_like"
        elif coarse == "kpi" and not _is_left_prose:
            refined = "chart_label_like"
        elif (
            col == "left"
            and band in ("top", "middle")
            and wr >= 0.26
            and ar >= 0.28
            and nr <= 0.62
            and sr <= 0.35
        ):
            refined = "narrative_text"
        elif (
            bk_key in list_like_keys
            and 6 <= len(t) <= 80
            and (col in ("right", "center") or band == "bottom")
        ):
            list_veto = False
            if len(t) >= 35:
                list_veto = True
            if ar >= 0.38 and wr >= 0.30:
                list_veto = True
            if _has_prose_punctuation_hint(t):
                list_veto = True
            if col == "left" and band in ("top", "middle") and wr >= 0.32:
                list_veto = True
            if len(t) >= 32 and ar >= 0.34:
                list_veto = True
            if not list_veto and 0.12 <= nr <= 0.75 and ar >= 0.18:
                refined = "list_item_like"
        elif (
            0.18 <= wr <= 0.46
            and ar >= 0.35
            and (cx >= 0.50 * pw or ph * 0.12 < y0 < ph * 0.50)
            and short_nbr.get(cid, 0) >= 3
        ):
            refined = "card_header_like"
        elif coarse == "card_text":
            if _looks_structural_issue_box(t):
                refined = "narrative_text" if col == "left" else "card_header_like"
            elif (
                col == "right"
                and not (ar >= 0.35 and len(t) >= 32 and nr <= 0.35)
                and (wr <= 0.32 or _chart_label_signal_hits(ln, sov, pw) >= 1)
            ):
                refined = "chart_label_like"
            elif col == "left" and band in ("top", "middle"):
                refined = "narrative_text"
            else:
                refined = "card_header_like"

        meta["dashboard_role_refined"] = refined
        if refined == "footer":
            ln["type"] = "footer"
        elif ln.get("type") == "footer":
            ln["type"] = "text"
            if meta.get("dashboard_role") == "footer":
                meta["dashboard_role"] = "card_text"

    _assign_left_column_section_roles(candidates, pw, ph)


def _narrative_hard_soft_passes(
    ln: dict,
    hard_zones: list[list[float]],
    soft_zones: list[list[float]],
    page_width: float,
) -> bool:
    ho = _line_max_zone_overlap_frac(ln["bbox"], hard_zones)
    sov = _line_max_zone_overlap_frac(ln["bbox"], soft_zones)
    if ho >= 0.10:
        return False
    bb = ln["bbox"]
    wr = (bb[2] - bb[0]) / max(page_width, 1.0)
    ar = _alpha_ratio((ln.get("text") or ""))
    if ln.get("meta", {}).get("dashboard_column_id") == "left" and wr >= 0.28 and ar >= 0.30:
        return sov < 0.38
    return sov < 0.20


def _reconstruct_narrative_paragraphs(
    lines: list[dict],
    page_width: float,
    page_height: float,
    hard_zones: list[list[float]],
    soft_zones: list[list[float]],
) -> list[dict]:
    def _infer_page_idx() -> int:
        for ln in lines:
            m = re.match(r"p(\d+)_", str(ln.get("id", "")))
            if m:
                return int(m.group(1)) - 1
        return 0

    page_idx = _infer_page_idx()
    narrative_roles = frozenset({"narrative_text", "section_body_like"})

    pre_cand = [
        ln
        for ln in lines
        if ln.get("meta", {}).get("dashboard_role_refined") in narrative_roles
        and ln.get("meta", {}).get("dashboard_column_id") == "left"
        and ln.get("meta", {}).get("dashboard_band_id") in ("top", "middle")
        and not _looks_contact_info((ln.get("text") or ""))
        and ln.get("meta", {}).get("dashboard_role_refined") != "footer"
    ]

    eligible = [ln for ln in pre_cand if _narrative_hard_soft_passes(ln, hard_zones, soft_zones, page_width)]

    if not eligible and pre_cand:
        rescue = sorted(
            [ln for ln in pre_cand if _line_max_zone_overlap_frac(ln["bbox"], hard_zones) < 0.12],
            key=lambda ln: -len((ln.get("text") or "").strip()),
        )
        eligible = rescue[: min(4, len(rescue))]

    eligible.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    cross_blocked = 0
    singleton_count = 0

    if not eligible:
        _reconstruct_narrative_paragraphs._last_stats = {
            "used_line_ids": set(),
            "merge_count": 0,
            "cross_column_blocked": cross_blocked,
            "singleton_count": 0,
            "pre_filter_count": len(pre_cand),
            "post_filter_count": 0,
        }
        return []

    gaps = [
        max(0.0, eligible[i + 1]["bbox"][1] - eligible[i]["bbox"][3])
        for i in range(len(eligible) - 1)
    ]
    med_gap = statistics.median(gaps) if gaps else page_height * 0.012
    v_thresh = max(2.4 * med_gap, 0.030 * page_height)

    def can_merge(a: dict, b: dict) -> bool:
        nonlocal cross_blocked
        ab, bb = a["bbox"], b["bbox"]
        if a.get("meta", {}).get("dashboard_column_id") != "left":
            cross_blocked += 1
            return False
        if b.get("meta", {}).get("dashboard_column_id") != "left":
            cross_blocked += 1
            return False
        if a.get("meta", {}).get("dashboard_band_id") == "bottom":
            return False
        if b.get("meta", {}).get("dashboard_band_id") == "bottom":
            return False
        if _looks_bracketed_topic_start(b.get("text", "")):
            return False
        if _line_max_zone_overlap_frac(ab, hard_zones) >= 0.10:
            return False
        if _line_max_zone_overlap_frac(bb, hard_zones) >= 0.10:
            return False
        soa = _line_max_zone_overlap_frac(ab, soft_zones)
        sob = _line_max_zone_overlap_frac(bb, soft_zones)
        wa = (ab[2] - ab[0]) / max(page_width, 1.0)
        wb = (bb[2] - bb[0]) / max(page_width, 1.0)
        capa = 0.38 if wa >= 0.28 and _alpha_ratio(a.get("text", "")) >= 0.30 else 0.20
        capb = 0.38 if wb >= 0.28 and _alpha_ratio(b.get("text", "")) >= 0.30 else 0.20
        if soa >= capa or sob >= capb:
            return False
        vgap = max(0.0, bb[1] - ab[3])
        if vgap > v_thresh:
            return False
        if abs(ab[0] - bb[0]) > 0.08 * page_width:
            return False
        if abs(ab[2] - bb[2]) > 0.14 * page_width:
            return False
        if abs(_line_width_ratio(a, page_width) - _line_width_ratio(b, page_width)) > 0.22:
            return False
        return True

    groups: list[list[dict]] = []
    cur = [eligible[0]]
    merges = 0
    for i in range(1, len(eligible)):
        if can_merge(cur[-1], eligible[i]):
            cur.append(eligible[i])
            merges += 1
        else:
            groups.append(cur)
            cur = [eligible[i]]
    groups.append(cur)

    def split_group(g: list[dict]) -> list[list[dict]]:
        roles_bad = {"chart_label_like", "table_header_like", "contact_info_like", "list_item_like"}
        if any(x.get("meta", {}).get("dashboard_role_refined") in roles_bad for x in g):
            return [[x] for x in g]
        if any(x.get("meta", {}).get("dashboard_band_id") == "bottom" for x in g):
            return [[x] for x in g]
        if any(x.get("meta", {}).get("dashboard_column_id") in ("right", "center") for x in g):
            return [[x] for x in g]
        return [g]

    split_groups: list[list[dict]] = []
    for g in groups:
        for sg in split_group(g):
            split_groups.append(sg)

    out_bl: list[dict] = []
    used: set[str] = set()
    for gi, g in enumerate(split_groups):
        if not g:
            continue
        if len(g) == 1:
            singleton_count += 1
        text = "\n".join(x.get("text", "") for x in g)
        bbox = [
            min(x["bbox"][0] for x in g),
            min(x["bbox"][1] for x in g),
            max(x["bbox"][2] for x in g),
            max(x["bbox"][3] for x in g),
        ]
        src_roles = [x.get("meta", {}).get("dashboard_role_refined") for x in g]
        region_ids = sorted({
            str(x.get("meta", {}).get("dashboard_region_id"))
            for x in g
            if x.get("meta", {}).get("dashboard_region_id")
        })
        out_bl.append({
            "id": f"p{page_idx + 1}_narr{gi}",
            "type": "text",
            "bbox": [round(v, 2) for v in bbox],
            "text": text,
            "source": "dashboard_narrative_paragraph",
            "confidence": 0.9,
            "meta": {
                "dashboard_role": "card_text",
                "dashboard_role_refined": "narrative_paragraph",
                "dashboard_line_ids": [x["id"] for x in g],
                "dashboard_refined_in_block": src_roles,
                "dashboard_source_roles_refined": src_roles,
                "dashboard_region_ids": region_ids,
                "summary_priority": "high",
                "summary_exclude": False,
            },
        })
        for x in g:
            used.add(x["id"])
    _reconstruct_narrative_paragraphs._last_stats = {
        "used_line_ids": used,
        "merge_count": merges,
        "cross_column_blocked": cross_blocked,
        "singleton_count": singleton_count,
        "pre_filter_count": len(pre_cand),
        "post_filter_count": len(eligible),
    }
    return out_bl


def _dashboard_single_line_block(ln: dict, page_idx: int, sid: str, source: str) -> dict:
    rfn = ln.get("meta", {}).get("dashboard_role_refined", "")
    coarse = ln.get("meta", {}).get("dashboard_role", "card_text")
    dr = coarse
    if rfn == "chart_label_like":
        dr = "kpi"
    elif rfn == "list_item_like":
        dr = "ranking"
    elif rfn == "table_header_like":
        dr = "mini_table"
    line_meta = ln.get("meta", {})
    block_type = "footer" if rfn in ("ignore_noise", "footer") else "text"
    priority = "high" if rfn in ("chart_label_like", "list_item_like", "table_header_like") else "medium"
    return {
        "id": f"p{page_idx + 1}_{sid}",
        "type": block_type,
        "bbox": [round(v, 2) for v in ln["bbox"]],
        "text": ln.get("text", ""),
        "source": source,
        "confidence": 0.85,
        "meta": {
            "dashboard_role": dr,
            "dashboard_role_refined": rfn,
            "dashboard_line_ids": [ln["id"]],
            "dashboard_refined_in_block": [rfn],
            "dashboard_region_id": line_meta.get("dashboard_region_id"),
            "dashboard_region_overlap_score": line_meta.get("dashboard_region_overlap_score"),
            "dashboard_table_overlap_frac": line_meta.get("dashboard_table_overlap_frac", 0.0),
            "summary_priority": "low" if block_type == "footer" else priority,
            "summary_exclude": rfn in ("ignore_noise", "footer"),
        },
    }


def _build_atomic_structured_blocks(
    lines: list[dict],
    page_width: float,
    page_height: float,
    consumed_ids: set[str],
    page_idx: int,
    table_blocks: list[dict] | None = None,
) -> tuple[list[dict], set[str], int, int, dict[str, Any]]:
    table_blocks = table_blocks or []
    pw, ph = page_width, page_height
    structured_roles = {
        "contact_info_like",
        "chart_label_like",
        "card_header_like",
        "list_item_like",
        "table_header_like",
    }
    atomic: list[dict] = []
    used: set[str] = set()
    structured_card_count = 0
    atomic_merge_absorbed = 0
    card_line_counts: list[int] = []
    table_duplicate_card_suppressed = 0
    table_duplicate_line_suppressed = 0

    pool = [
        ln
        for ln in lines
        if ln["id"] not in consumed_ids
        and ln.get("meta", {}).get("dashboard_role_refined") in structured_roles
    ]

    contact_lines = [ln for ln in pool if ln.get("meta", {}).get("dashboard_role_refined") == "contact_info_like"]
    for i, ln in enumerate(contact_lines):
        atomic.append(_dashboard_single_line_block(ln, page_idx, f"atom_c{i}", "dashboard_atomic_contact"))
        used.add(ln["id"])

    def _emit_structured_card(clus: list[dict], sid: str, table_dup: bool) -> None:
        nonlocal structured_card_count, atomic_merge_absorbed
        nonlocal table_duplicate_card_suppressed, table_duplicate_line_suppressed
        refin = [x.get("meta", {}).get("dashboard_role_refined") for x in clus]
        if table_dup:
            for x in clus:
                used.add(x["id"])
            table_duplicate_card_suppressed += 1
            table_duplicate_line_suppressed += len(clus)
            atomic_merge_absorbed += max(0, len(clus) - 1)
            return
        text = "\n".join(x.get("text", "") for x in clus)
        prose_lines = sum(
            1
            for x in clus
            if len((x.get("text") or "").strip()) >= 32
            and _alpha_ratio((x.get("text") or "")) >= 0.32
            and _numeric_ratio((x.get("text") or "")) <= 0.38
        )
        card_role = "card_text" if prose_lines >= max(1, len(clus) // 2) else "kpi"
        bbox = [
            min(x["bbox"][0] for x in clus),
            min(x["bbox"][1] for x in clus),
            max(x["bbox"][2] for x in clus),
            max(x["bbox"][3] for x in clus),
        ]
        atomic.append({
            "id": f"p{page_idx + 1}_{sid}",
            "type": "text",
            "bbox": [round(v, 2) for v in bbox],
            "text": text,
            "source": "dashboard_structured_card",
            "confidence": 0.87,
            "meta": {
                "dashboard_role": card_role,
                "dashboard_role_refined": "structured_card",
                "dashboard_line_ids": [x["id"] for x in clus],
                "dashboard_refined_in_block": refin,
                "dashboard_structured_card_atomic_roles": refin,
                "dashboard_region_ids": sorted({
                    str(x.get("meta", {}).get("dashboard_region_id"))
                    for x in clus
                    if x.get("meta", {}).get("dashboard_region_id")
                }),
                "summary_priority": "high" if card_role == "card_text" else "medium",
                "summary_exclude": False,
                "dashboard_overlaps_table_bbox": table_dup,
            },
        })
        for x in clus:
            used.add(x["id"])
        structured_card_count += 1
        card_line_counts.append(len(clus))
        if len(clus) > 1:
            atomic_merge_absorbed += len(clus) - 1

    right_mix_roles = frozenset({"chart_label_like", "list_item_like", "card_header_like"})
    right_pool = [
        ln
        for ln in pool
        if ln["id"] not in used
        and ln.get("meta", {}).get("dashboard_role_refined") in right_mix_roles
        and ln.get("meta", {}).get("dashboard_column_id") in ("right", "center")
    ]
    right_pool.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    yi = 0
    cg = 0
    while yi < len(right_pool):
        clus = [right_pool[yi]]
        j = yi + 1
        while j < len(right_pool) and len(clus) < 14:
            prev, cur = clus[-1], right_pool[j]
            if cur["bbox"][3] - clus[0]["bbox"][1] > 0.40 * ph:
                break
            vg = max(0.0, cur["bbox"][1] - prev["bbox"][3])
            if vg > 0.028 * ph:
                break
            pw_prev = max(prev["bbox"][2] - prev["bbox"][0], 1.0)
            pw_cur = max(cur["bbox"][2] - cur["bbox"][0], 1.0)
            hx = max(0.0, min(prev["bbox"][2], cur["bbox"][2]) - max(prev["bbox"][0], cur["bbox"][0]))
            if hx < 0.10 * min(pw_prev, pw_cur) and vg > 0.012 * ph:
                break
            clus.append(cur)
            j += 1
        tbl_hits = [
            x for x in clus
            if _line_table_overlap_max(x, table_blocks) >= 0.38
        ]
        tbl_dup = (
            len(tbl_hits) >= max(1, (len(clus) + 1) // 2)
            or any(
                x.get("meta", {}).get("dashboard_role_refined") == "table_header_like"
                for x in tbl_hits
            )
        )
        _emit_structured_card(clus, f"scard{cg}", tbl_dup)
        yi += len(clus)
        cg += 1

    list_pool = [
        ln
        for ln in pool
        if ln.get("meta", {}).get("dashboard_role_refined") == "list_item_like"
        and ln["id"] not in used
        and _line_table_overlap_max(ln, table_blocks) < 0.38
    ]
    list_pool.sort(
        key=lambda ln: (int(ln["bbox"][0] / max(page_width * 0.03, 1.0)), ln["bbox"][1])
    )
    li = 0
    i = 0
    while i < len(list_pool):
        bk = int(list_pool[i]["bbox"][0] / max(page_width * 0.03, 1.0))
        chunk: list[dict] = []
        while i < len(list_pool):
            ln = list_pool[i]
            if ln["id"] in used:
                i += 1
                continue
            if int(ln["bbox"][0] / max(page_width * 0.03, 1.0)) != bk:
                break
            chunk.append(ln)
            i += 1
            if len(chunk) >= 4:
                break
        if not chunk:
            continue
        chunk.sort(key=lambda ln: ln["bbox"][1])
        if len(chunk) == 1:
            ln = chunk[0]
            atomic.append(_dashboard_single_line_block(ln, page_idx, f"atom_l{li}", "dashboard_atomic_list"))
            used.add(ln["id"])
        else:
            text = "\n".join(x.get("text", "") for x in chunk)
            bbox = [
                min(x["bbox"][0] for x in chunk),
                min(x["bbox"][1] for x in chunk),
                max(x["bbox"][2] for x in chunk),
                max(x["bbox"][3] for x in chunk),
            ]
            atomic.append({
                "id": f"p{page_idx + 1}_atom_l{li}",
                "type": "text",
                "bbox": [round(v, 2) for v in bbox],
                "text": text,
                "source": "dashboard_atomic_list",
                "confidence": 0.86,
                "meta": {
                    "dashboard_role": "ranking",
                    "dashboard_role_refined": "list_item_like",
                    "dashboard_line_ids": [x["id"] for x in chunk],
                    "dashboard_refined_in_block": ["list_item_like"] * len(chunk),
                    "dashboard_region_ids": sorted({
                        str(x.get("meta", {}).get("dashboard_region_id"))
                        for x in chunk
                        if x.get("meta", {}).get("dashboard_region_id")
                    }),
                    "summary_priority": "high",
                    "summary_exclude": False,
                },
            })
            for x in chunk:
                used.add(x["id"])
            atomic_merge_absorbed += len(chunk) - 1
        li += 1

    ch_left = 0
    for ln in pool:
        if ln["id"] in used:
            continue
        if (
            ln.get("meta", {}).get("dashboard_role_refined") == "chart_label_like"
            and ln.get("meta", {}).get("dashboard_column_id") not in ("right", "center")
            and _line_table_overlap_max(ln, table_blocks) < 0.38
        ):
            atomic.append(_dashboard_single_line_block(ln, page_idx, f"atom_chl{ch_left}", "dashboard_atomic_chart"))
            used.add(ln["id"])
            ch_left += 1

    hi = 0
    thi = 0
    for ln in pool:
        if ln["id"] in used:
            continue
        rr = ln.get("meta", {}).get("dashboard_role_refined")
        if _line_table_overlap_max(ln, table_blocks) >= 0.38:
            used.add(ln["id"])
            table_duplicate_line_suppressed += 1
            continue
        if rr == "card_header_like":
            atomic.append(
                _dashboard_single_line_block(ln, page_idx, f"atom_h{hi}", "dashboard_atomic_header")
            )
            used.add(ln["id"])
            hi += 1
        elif rr == "table_header_like":
            atomic.append(
                _dashboard_single_line_block(
                    ln, page_idx, f"atom_th{thi}", "dashboard_atomic_table_header"
                )
            )
            used.add(ln["id"])
            thi += 1

    avg_lines = statistics.fmean(card_line_counts) if card_line_counts else 0.0
    max_lines = max(card_line_counts) if card_line_counts else 0
    card_stats: dict[str, Any] = {
        "dashboard_structured_card_avg_lines": round(avg_lines, 3),
        "dashboard_structured_card_max_lines": max_lines,
        "dashboard_table_duplicate_card_suppressed_count": table_duplicate_card_suppressed,
        "dashboard_table_duplicate_line_suppressed_count": table_duplicate_line_suppressed,
    }
    return atomic, used, structured_card_count, atomic_merge_absorbed, card_stats


def _build_left_section_blocks(
    lines: list[dict],
    consumed: set[str],
    page_idx: int,
    pw: float,
    ph: float,
) -> tuple[list[dict], set[str]]:
    sec = [
        ln
        for ln in lines
        if ln["id"] not in consumed
        and ln.get("meta", {}).get("dashboard_column_id") == "left"
        and ln.get("meta", {}).get("dashboard_role_refined")
        in ("section_header_like", "section_body_like")
    ]
    sec.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    out: list[dict] = []
    used: set[str] = set()
    si = 0
    i = 0
    while i < len(sec):
        ln = sec[i]
        if ln.get("meta", {}).get("dashboard_role_refined") != "section_header_like":
            i += 1
            continue
        out.append({
            "id": f"p{page_idx + 1}_sec_h{si}",
            "type": "text",
            "bbox": [round(v, 2) for v in ln["bbox"]],
            "text": ln.get("text", ""),
            "source": "dashboard_left_section_header",
            "confidence": 0.88,
            "meta": {
                "dashboard_role": "issue_box",
                "dashboard_role_refined": "section_header_like",
                "dashboard_line_ids": [ln["id"]],
                "dashboard_refined_in_block": ["section_header_like"],
                "dashboard_region_id": ln.get("meta", {}).get("dashboard_region_id"),
                "summary_priority": "high",
                "summary_exclude": False,
            },
        })
        used.add(ln["id"])
        i += 1
        body: list[dict] = []
        last_bottom = ln["bbox"][3]
        while i < len(sec):
            nxt = sec[i]
            if nxt.get("meta", {}).get("dashboard_role_refined") != "section_body_like":
                break
            if nxt["bbox"][1] - last_bottom > ph * 0.048:
                break
            body.append(nxt)
            last_bottom = nxt["bbox"][3]
            i += 1
        if body:
            text = "\n".join(x.get("text", "") for x in body)
            bbox = [
                min(x["bbox"][0] for x in body),
                min(x["bbox"][1] for x in body),
                max(x["bbox"][2] for x in body),
                max(x["bbox"][3] for x in body),
            ]
            out.append({
                "id": f"p{page_idx + 1}_sec_b{si}",
                "type": "text",
                "bbox": [round(v, 2) for v in bbox],
                "text": text,
                "source": "dashboard_left_section_body",
                "confidence": 0.86,
                "meta": {
                    "dashboard_role": "card_text",
                    "dashboard_role_refined": "section_body_like",
                    "dashboard_line_ids": [x["id"] for x in body],
                    "dashboard_refined_in_block": ["section_body_like"] * len(body),
                    "dashboard_region_ids": sorted({
                        str(x.get("meta", {}).get("dashboard_region_id"))
                        for x in body
                        if x.get("meta", {}).get("dashboard_region_id")
                    }),
                    "summary_priority": "high",
                    "summary_exclude": False,
                },
            })
            for x in body:
                used.add(x["id"])
        si += 1
    return out, used


def _sort_dashboard_final_blocks(blocks: list[dict]) -> list[dict]:
    def _pri(b: dict) -> tuple[int, float]:
        t = b.get("type")
        rr = b.get("meta", {}).get("dashboard_role_refined") or ""
        oy = float(b.get("bbox", [0, 0, 0, 0])[1])
        if t == "title":
            return (0, oy)
        if rr == "narrative_paragraph":
            return (1, oy)
        if rr == "section_header_like":
            return (2, oy)
        if rr == "section_body_like":
            return (3, oy)
        if rr == "structured_card":
            return (4, oy)
        if rr in ("chart_label_like", "list_item_like", "card_header_like"):
            return (5, oy)
        if rr == "contact_info_like":
            return (6, oy)
        if t == "footer" or rr == "footer":
            return (8, oy)
        if rr == "ignore_noise":
            return (9, oy)
        return (7, oy)

    return sorted(blocks, key=_pri)


def _pure_narrative_paragraph_split_exempt(blk: dict, line_by_id: dict[str, dict]) -> bool:
    """Narrow: reconstructed narrative paragraphs must not be torn apart by mixed split."""
    if blk.get("meta", {}).get("dashboard_role_refined") != "narrative_paragraph":
        return False
    lids = blk.get("meta", {}).get("dashboard_line_ids") or []
    if len(lids) < 1:
        return False
    for lid in lids:
        ln = line_by_id.get(lid)
        if ln is None:
            return False
        mm = ln.get("meta", {})
        if mm.get("dashboard_role_refined") != "narrative_text":
            return False
        if mm.get("dashboard_column_id") != "left":
            return False
        if mm.get("dashboard_band_id") not in ("top", "middle"):
            return False
    return True


def _pure_section_block_split_exempt(blk: dict, line_by_id: dict[str, dict]) -> bool:
    rr = blk.get("meta", {}).get("dashboard_role_refined")
    if rr not in ("section_header_like", "section_body_like"):
        return False
    lids = blk.get("meta", {}).get("dashboard_line_ids") or []
    if not lids:
        return False
    for lid in lids:
        ln = line_by_id.get(lid)
        if ln is None:
            return False
        mm = ln.get("meta", {})
        if mm.get("dashboard_column_id") != "left":
            return False
        if mm.get("dashboard_band_id") not in ("top", "middle"):
            return False
        if mm.get("dashboard_role_refined") != rr:
            return False
    return True


def _dashboard_small_mixed_block_split(
    blocks: list[dict],
    line_by_id: dict[str, dict],
    page_idx: int,
    split_debug: dict[str, Any] | None = None,
) -> tuple[list[dict], int]:
    out: list[dict] = []
    split_count = 0
    split_uid = 0
    np_seen = 0
    np_exempt = 0
    for blk in blocks:
        if blk.get("type") not in ("text", "title"):
            out.append(blk)
            continue
        if _pure_section_block_split_exempt(blk, line_by_id):
            out.append(blk)
            continue
        if blk.get("meta", {}).get("dashboard_role_refined") == "narrative_paragraph":
            np_seen += 1
            if _pure_narrative_paragraph_split_exempt(blk, line_by_id):
                np_exempt += 1
                out.append(blk)
                continue
        if blk.get("meta", {}).get("dashboard_role_refined") == "structured_card":
            out.append(blk)
            continue
        lids = blk.get("meta", {}).get("dashboard_line_ids") or []
        if len(lids) < 2:
            out.append(blk)
            continue
        lines_sub = [line_by_id[lid] for lid in lids if lid in line_by_id]
        if len(lines_sub) < 2:
            out.append(blk)
            continue
        refined = [ln.get("meta", {}).get("dashboard_role_refined") for ln in lines_sub]
        cols = {ln.get("meta", {}).get("dashboard_column_id") for ln in lines_sub}
        bands = {ln.get("meta", {}).get("dashboard_band_id") for ln in lines_sub}
        distinct_r = {r for r in refined if r}
        hits = 0
        if len(distinct_r) >= 2:
            hits += 1
        if "contact_info_like" in distinct_r:
            hits += 1
        if "narrative_text" in distinct_r and "table_header_like" in distinct_r:
            hits += 1
        if "narrative_text" in distinct_r and "chart_label_like" in distinct_r:
            hits += 1
        if "narrative_text" in distinct_r and "section_header_like" in distinct_r:
            hits += 1
        if "section_body_like" in distinct_r and "chart_label_like" in distinct_r:
            hits += 1
        if "section_header_like" in distinct_r and "list_item_like" in distinct_r and "section_body_like" not in distinct_r:
            hits += 1
        br = blk.get("meta", {}).get("dashboard_role_refined")
        if br == "narrative_paragraph" and len(distinct_r) >= 2:
            hits += 2
        if len(cols) >= 2:
            hits += 1
        if len(bands) >= 2:
            hits += 1
        short_ticks = sum(
            1
            for ln in lines_sub
            if len((ln.get("text") or "").strip()) <= 14
            and _numeric_ratio((ln.get("text") or "")) >= 0.35
        )
        long_prose = sum(
            1
            for ln in lines_sub
            if len((ln.get("text") or "").strip()) >= 40
            and _alpha_ratio((ln.get("text") or "")) >= 0.38
        )
        if short_ticks >= 1 and long_prose >= 1:
            hits += 1
        lens = [len((ln.get("text") or "").strip()) for ln in lines_sub]
        if lens and max(lens) > 55 and min(lens) < 16:
            if _alpha_ratio(lines_sub[lens.index(max(lens))].get("text", "")) >= 0.48:
                hits += 1
        if hits < 2:
            out.append(blk)
            continue
        split_count += 1
        narr_roles = frozenset({"narrative_text", "section_body_like"})
        narr = [ln for ln in lines_sub if ln.get("meta", {}).get("dashboard_role_refined") in narr_roles]
        other = [ln for ln in lines_sub if ln not in narr]
        if len(narr) >= 1:
            narr.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
            tb = "\n".join(ln.get("text", "") for ln in narr)
            bbox = [
                min(ln["bbox"][0] for ln in narr),
                min(ln["bbox"][1] for ln in narr),
                max(ln["bbox"][2] for ln in narr),
                max(ln["bbox"][3] for ln in narr),
            ]
            out.append({
                "id": f"p{page_idx + 1}_split_narr{split_uid}",
                "type": "text",
                "bbox": [round(v, 2) for v in bbox],
                "text": tb,
                "source": "dashboard_mixed_split",
                "confidence": 0.82,
                "meta": {
                    "dashboard_role": "card_text",
                    "dashboard_role_refined": "narrative_text",
                    "dashboard_line_ids": [ln["id"] for ln in narr],
                    "dashboard_refined_in_block": ["narrative_text"] * len(narr),
                    "summary_priority": "high",
                    "summary_exclude": False,
                },
            })
            split_uid += 1
        for j, ln in enumerate(other):
            out.append(
                _dashboard_single_line_block(
                    ln,
                    page_idx,
                    f"split_o{split_uid}_{j}",
                    "dashboard_mixed_split_atomic",
                )
            )
            split_uid += 1
    if split_debug is not None:
        split_debug["dashboard_small_split_narrative_paragraph_blocks_seen"] = np_seen
        split_debug["dashboard_small_split_narrative_paragraph_blocks_exempt"] = np_exempt
    return out, split_count


def _segment_dashboard_regions(
    lines: list[dict],
    _table_blocks: list[dict],
    pw: float,
    ph: float,
) -> tuple[list[list[float]], int]:
    region_roles = {
        "narrative_text",
        "section_body_like",
        "section_header_like",
        "card_header_like",
        "list_item_like",
        "table_header_like",
        "chart_label_like",
    }
    seed_blocks = [
        ln
        for ln in lines
        if ln.get("meta", {}).get("dashboard_role_refined") in region_roles
        and ln.get("meta", {}).get("dashboard_band_id") != "bottom"
        and _line_table_overlap_max(ln, _table_blocks) < 0.55
        and _line_context_text_ok(ln)
    ]
    stats: dict[str, Any] = {
        "dashboard_region_cluster_count": 0,
        "dashboard_region_seed_role_distribution": dict(Counter(
            str(ln.get("meta", {}).get("dashboard_role_refined", "")) for ln in seed_blocks
        )),
        "dashboard_region_cross_column_blocked_count": 0,
        "dashboard_region_cross_band_blocked_count": 0,
        "dashboard_region_area_ratio_stats": {"min": None, "max": None, "mean": None, "count": 0},
        "dashboard_region_line_count_stats": {"min": None, "max": None, "mean": None, "count": 0},
    }
    if len(seed_blocks) < 2:
        _segment_dashboard_regions._last_stats = stats
        return [], len(seed_blocks)

    clusters: list[list[dict]] = []
    for block in sorted(seed_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        assigned = False
        bx0, by0, bx1, by1 = block["bbox"]
        b_col = block.get("meta", {}).get("dashboard_column_id")
        b_band = block.get("meta", {}).get("dashboard_band_id")
        for cluster in clusters:
            cx0 = min(item["bbox"][0] for item in cluster)
            cy0 = min(item["bbox"][1] for item in cluster)
            cx1 = max(item["bbox"][2] for item in cluster)
            cy1 = max(item["bbox"][3] for item in cluster)
            cluster_cols = {item.get("meta", {}).get("dashboard_column_id") for item in cluster}
            cluster_bands = {item.get("meta", {}).get("dashboard_band_id") for item in cluster}
            if b_col not in cluster_cols and not (b_col == "center" or "center" in cluster_cols):
                stats["dashboard_region_cross_column_blocked_count"] += 1
                continue
            if b_band not in cluster_bands and by0 - cy1 > ph * 0.025:
                stats["dashboard_region_cross_band_blocked_count"] += 1
                continue
            h_gap = max(0.0, max(cx0 - bx1, bx0 - cx1))
            v_gap = max(0.0, max(cy0 - by1, by0 - cy1))
            same_band = abs(((cy0 + cy1) / 2.0) - ((by0 + by1) / 2.0)) < ph * 0.08
            candidate_bbox = [min(cx0, bx0), min(cy0, by0), max(cx1, bx1), max(cy1, by1)]
            candidate_area = _zone_area_ratio_on_page(candidate_bbox, pw, ph)
            if candidate_area > 0.22:
                continue
            if (h_gap < pw * 0.045 and v_gap < ph * 0.032) or (same_band and h_gap < pw * 0.060 and v_gap < ph * 0.018):
                cluster.append(block)
                assigned = True
                break
        if not assigned:
            clusters.append([block])

    regions: list[list[float]] = []
    for cluster in clusters:
        x0 = min(item["bbox"][0] for item in cluster)
        y0 = min(item["bbox"][1] for item in cluster)
        x1 = max(item["bbox"][2] for item in cluster)
        y1 = max(item["bbox"][3] for item in cluster)
        regions.append([x0, y0, x1, y1])
    area_ratios = [_zone_area_ratio_on_page(region, pw, ph) for region in regions]
    line_counts = [len(cluster) for cluster in clusters]
    if area_ratios:
        stats["dashboard_region_area_ratio_stats"] = {
            "min": round(min(area_ratios), 5),
            "max": round(max(area_ratios), 5),
            "mean": round(statistics.fmean(area_ratios), 5),
            "count": len(area_ratios),
        }
    if line_counts:
        stats["dashboard_region_line_count_stats"] = {
            "min": min(line_counts),
            "max": max(line_counts),
            "mean": round(statistics.fmean(line_counts), 3),
            "count": len(line_counts),
        }
    stats["dashboard_region_cluster_count"] = len(regions)
    _segment_dashboard_regions._last_stats = stats
    return regions, len(seed_blocks)


def _build_dashboard_region_blocks(
    lines: list[dict],
    regions: list[list[float]],
    page_idx: int,
    table_blocks: list[dict],
    visual_blocks: list[dict],
    pw: float,
    ph: float,
) -> tuple[list[dict], int, dict[str, Any]]:
    hard_z, soft_z = _build_dashboard_exclusion_zones(lines, table_blocks, pw, ph)
    filtered_excl = sum(
        1
        for ln in lines
        if _line_max_zone_overlap_frac(ln["bbox"], hard_z) >= 0.15
    )
    def _max_zone_ratio(zl: list[list[float]]) -> float:
        if not zl:
            return 0.0
        return max(_zone_area_ratio_on_page(z, pw, ph) for z in zl)
    col_ids = {ln.get("meta", {}).get("dashboard_column_id") for ln in lines}
    band_ids = {ln.get("meta", {}).get("dashboard_band_id") for ln in lines}
    role_dist = Counter(
        str(ln.get("meta", {}).get("dashboard_role_refined", "")) for ln in lines
    )

    debug: dict[str, Any] = {
        "dashboard_region_merge_input_count": 0,
        "dashboard_region_merge_kept_count": 0,
        "dashboard_region_merge_excluded_count": 0,
        "dashboard_narrative_filter_drop_count": 0,
        "dashboard_large_region_guard_triggered": 0,
        "dashboard_exclusion_zone_count": len(hard_z) + len(soft_z),
        "dashboard_hard_exclusion_zone_count": len(hard_z),
        "dashboard_soft_exclusion_zone_count": len(soft_z),
        "dashboard_hard_exclusion_max_area_ratio": round(_max_zone_ratio(hard_z), 4),
        "dashboard_soft_exclusion_max_area_ratio": round(_max_zone_ratio(soft_z), 4),
        "dashboard_exclusion_zone_filtered_line_count": filtered_excl,
        "dashboard_band_count": len(band_ids),
        "dashboard_column_count": len(col_ids),
        "dashboard_cross_band_merge_blocked_count": 0,
        "dashboard_cross_column_merge_blocked_count": 0,
        "dashboard_region_purity_score_stats": {"min": None, "max": None, "mean": None, "count": 0},
        "dashboard_region_purity_reject_count": 0,
        "dashboard_large_region_abort_count": 0,
        "dashboard_narrative_paragraph_count": 0,
        "dashboard_paragraph_reconstruction_merge_count": 0,
        "dashboard_narrative_singleton_count": 0,
        "dashboard_narrative_candidate_count_pre_filter": 0,
        "dashboard_narrative_candidate_count_post_filter": 0,
        "dashboard_left_section_count": 0,
        "dashboard_left_structured_body_count": 0,
        "dashboard_structured_card_count": 0,
        "dashboard_structured_card_avg_lines": 0.0,
        "dashboard_structured_card_max_lines": 0,
        "dashboard_table_duplicate_card_suppressed_count": 0,
        "dashboard_table_duplicate_line_suppressed_count": 0,
        "dashboard_small_mixed_block_split_count": 0,
        "dashboard_role_distribution_refined": dict(role_dist),
        "dashboard_narrative_returned_block_count": 0,
        "dashboard_narrative_returned_line_count": 0,
        "dashboard_narrative_returned_block_ids_head": [],
        "dashboard_narrative_consumed_line_count": 0,
        "dashboard_blocks_after_title_count": 0,
        "dashboard_blocks_after_narrative_extend_count": 0,
        "dashboard_blocks_after_atomic_extend_count": 0,
        "dashboard_blocks_before_small_split_count": 0,
        "dashboard_blocks_after_small_split_count": 0,
        "dashboard_narrative_blocks_survived_after_small_split_count": 0,
        "dashboard_narrative_line_ids_reemitted_as_fallback_count": 0,
        "dashboard_narrative_line_ids_missing_after_finalize_count": 0,
        "dashboard_ignore_noise_suppressed_count": 0,
    }
    if not lines:
        return [], 0, debug

    line_by_id = {ln["id"]: ln for ln in lines}

    narr_blocks = _reconstruct_narrative_paragraphs(lines, pw, ph, hard_z, soft_z)
    narr_stats = getattr(_reconstruct_narrative_paragraphs, "_last_stats", {})
    narr_used = set(narr_stats.get("used_line_ids") or [])
    para_merges = narr_stats.get("merge_count", 0)
    cross_blk = narr_stats.get("cross_column_blocked", 0)
    debug["dashboard_cross_column_merge_blocked_count"] = cross_blk
    debug["dashboard_narrative_paragraph_count"] = len(narr_blocks)
    debug["dashboard_narrative_singleton_count"] = narr_stats.get("singleton_count", 0)
    debug["dashboard_narrative_candidate_count_pre_filter"] = narr_stats.get("pre_filter_count", 0)
    debug["dashboard_narrative_candidate_count_post_filter"] = narr_stats.get("post_filter_count", 0)

    narr_line_ids_from_blocks: set[str] = set()
    for _nb in narr_blocks:
        narr_line_ids_from_blocks.update(_nb.get("meta", {}).get("dashboard_line_ids") or [])
    debug["dashboard_narrative_returned_block_count"] = len(narr_blocks)
    debug["dashboard_narrative_returned_line_count"] = sum(
        len(_b.get("meta", {}).get("dashboard_line_ids") or []) for _b in narr_blocks
    )
    debug["dashboard_narrative_returned_block_ids_head"] = [
        _b.get("id", "") for _b in narr_blocks[:5]
    ]
    narrative_line_ids = set(narr_used) | narr_line_ids_from_blocks
    debug["dashboard_narrative_consumed_line_count"] = len(narrative_line_ids)
    debug["dashboard_paragraph_reconstruction_merge_count"] = para_merges

    section_blocks, section_used = _build_left_section_blocks(
        lines, set(narrative_line_ids), page_idx, pw, ph
    )
    debug["dashboard_left_section_count"] = sum(
        1
        for b in section_blocks
        if b.get("meta", {}).get("dashboard_role_refined") == "section_header_like"
    )
    debug["dashboard_left_structured_body_count"] = sum(
        1
        for b in section_blocks
        if b.get("meta", {}).get("dashboard_role_refined") == "section_body_like"
    )

    consumed_pre_atomic = set(narrative_line_ids) | set(section_used)
    atomic_blocks, atomic_used, sc_count, atomic_merges, card_stats = _build_atomic_structured_blocks(
        lines, pw, ph, consumed_pre_atomic, page_idx, table_blocks
    )
    debug["dashboard_structured_card_count"] = sc_count
    debug["dashboard_structured_card_avg_lines"] = card_stats.get("dashboard_structured_card_avg_lines", 0.0)
    debug["dashboard_structured_card_max_lines"] = card_stats.get("dashboard_structured_card_max_lines", 0)
    debug["dashboard_table_duplicate_card_suppressed_count"] = card_stats.get("dashboard_table_duplicate_card_suppressed_count", 0)
    debug["dashboard_table_duplicate_line_suppressed_count"] = card_stats.get("dashboard_table_duplicate_line_suppressed_count", 0)

    consumed: set[str] = set(atomic_used) | consumed_pre_atomic

    titles = [ln for ln in lines if ln.get("meta", {}).get("dashboard_role") == "title"]
    footers = [ln for ln in lines if ln["type"] == "footer"]

    merged: list[dict] = []
    for ti, tln in enumerate(titles):
        merged.append({
            "id": f"p{page_idx + 1}_title{ti}",
            "type": "title",
            "bbox": [round(v, 2) for v in tln["bbox"]],
            "text": tln.get("text", ""),
            "source": "dashboard_title",
            "confidence": 1.0,
            "meta": {
                "dashboard_role": "title",
                "dashboard_role_refined": tln.get("meta", {}).get("dashboard_role_refined", "card_header_like"),
                "dashboard_line_ids": [tln["id"]],
                "dashboard_refined_in_block": [tln.get("meta", {}).get("dashboard_role_refined", "card_header_like")],
                "dashboard_region_id": tln.get("meta", {}).get("dashboard_region_id"),
                "summary_priority": "high",
                "summary_exclude": False,
            },
        })
        consumed.add(tln["id"])

    debug["dashboard_blocks_after_title_count"] = len(merged)
    merged.extend(narr_blocks)
    debug["dashboard_blocks_after_narrative_extend_count"] = len(merged)
    merged.extend(section_blocks)
    merged.extend(atomic_blocks)
    debug["dashboard_blocks_after_atomic_extend_count"] = len(merged)

    reemit_fb = 0
    for ln in lines:
        if ln["id"] in consumed:
            continue
        if ln["id"] in narrative_line_ids:
            reemit_fb += 1
            continue
        rr = ln.get("meta", {}).get("dashboard_role_refined", "")
        if rr == "ignore_noise":
            consumed.add(ln["id"])
            debug["dashboard_ignore_noise_suppressed_count"] += 1
            continue
        if ln in footers:
            continue
        coarse = ln.get("meta", {}).get("dashboard_role", "card_text")
        # Lines classified as footer by refined role → emit as proper footer
        is_refined_footer = (rr == "footer")
        merged.append(
            {
                "id": f"p{page_idx + 1}_rest_{ln['id']}",
                "type": "footer" if is_refined_footer else "text",
                "bbox": [round(v, 2) for v in ln["bbox"]],
                "text": ln.get("text", ""),
                "source": "dashboard_line_fallback",
                "confidence": 0.78,
                "meta": {
                    "dashboard_role": "footer" if is_refined_footer else coarse,
                    "dashboard_role_refined": rr or "card_header_like",
                    "dashboard_line_ids": [ln["id"]],
                    "dashboard_refined_in_block": [rr or "card_header_like"],
                    "dashboard_region_id": ln.get("meta", {}).get("dashboard_region_id"),
                    "dashboard_region_overlap_score": ln.get("meta", {}).get("dashboard_region_overlap_score"),
                    "dashboard_table_overlap_frac": ln.get("meta", {}).get("dashboard_table_overlap_frac", 0.0),
                    "summary_priority": "low" if is_refined_footer else "medium",
                    "summary_exclude": is_refined_footer,
                },
            }
        )
        consumed.add(ln["id"])

    for fln in footers:
        merged.append(
            {
                "id": f"p{page_idx + 1}_foot{fln['id']}",
                "type": "footer",
                "bbox": [round(v, 2) for v in fln["bbox"]],
                "text": fln.get("text", ""),
                "source": "dashboard_footer",
                "confidence": 0.9,
                "meta": {
                    "dashboard_role": "footer",
                    "dashboard_role_refined": "footer",
                    "dashboard_line_ids": [fln["id"]],
                    "dashboard_refined_in_block": ["footer"],
                    "summary_priority": "low",
                    "summary_exclude": True,
                },
            }
        )

    def _all_block_line_ids(blocks: list[dict]) -> set[str]:
        acc: set[str] = set()
        for _b in blocks:
            for _lid in _b.get("meta", {}).get("dashboard_line_ids") or []:
                acc.add(_lid)
        return acc

    debug["dashboard_narrative_line_ids_reemitted_as_fallback_count"] = reemit_fb

    debug["dashboard_blocks_before_small_split_count"] = len(merged)

    split_aux: dict[str, Any] = {}
    merged, spl = _dashboard_small_mixed_block_split(merged, line_by_id, page_idx, split_aux)
    debug["dashboard_small_mixed_block_split_count"] = spl
    debug.update(split_aux)

    debug["dashboard_blocks_after_small_split_count"] = len(merged)
    debug["dashboard_narrative_blocks_survived_after_small_split_count"] = sum(
        1
        for _b in merged
        if _b.get("meta", {}).get("dashboard_role_refined") == "narrative_paragraph"
    )

    final_line_ids = _all_block_line_ids(merged)
    debug["dashboard_narrative_line_ids_missing_after_finalize_count"] = len(
        narrative_line_ids - final_line_ids
    )

    absorbed_count = para_merges + atomic_merges

    return merged, absorbed_count, debug


def _classify_dashboard_line(text: str, bbox: list[float], pw: float, ph: float, font_size: float) -> str:
    """
    Coarse dashboard line role for legacy merge hints. Refined roles for chunking/RAG
    (`meta["dashboard_role_refined"]`) are assigned later in `_enrich_dashboard_line_meta`.
    """
    clean = text.strip()
    y0, y1 = bbox[1], bbox[3]
    width = bbox[2] - bbox[0]
    alpha_count = len(re.findall(r"[A-Za-z가-힣]", clean))
    digit_count = len(re.findall(r"\d", clean))

    if y0 < ph * 0.14 and font_size >= 13 and alpha_count >= 2 and len(clean) <= 90:
        return "title"
    if y1 > ph * 0.9 and len(clean) <= 40:
        return "footer"
    if _looks_structural_issue_box(clean):
        return "issue_box"
    if _looks_structural_ranking(clean):
        return "ranking"
    if _looks_structural_mini_table(clean, width, pw):
        return "mini_table"
    if digit_count >= max(3, alpha_count) and len(clean) <= 28:
        return "kpi"
    if re.search(r"(주요|핵심|issue|comment|요약|summary|체크포인트)", clean, re.I):
        return "issue_box"
    if re.search(r"^\s*(\d+[\.\)]|[①-⑳]|[가-힣]\.)", clean):
        return "ranking"
    if (clean.count("|") >= 2 or clean.count(":") >= 2 or clean.count("/") >= 3) and width > pw * 0.18:
        return "mini_table"
    if alpha_count == 0 and digit_count <= 2 and width < pw * 0.03:
        return "decorative_noise"
    return "card_text"


def _classify_dashboard_region(group: list[dict]) -> str:
    joined = " ".join(item["text"] for item in group)
    roles = [item.get("meta", {}).get("dashboard_role", "card_text") for item in group]
    if any(role == "issue_box" for role in roles):
        return "issue_box"
    if any(role == "mini_table" for role in roles):
        return "mini_table"
    if roles.count("ranking") >= 2:
        return "ranking"
    if roles.count("kpi") >= max(1, len(group) // 2):
        return "kpi"
    if re.search(r"(전망|요약|핵심|point|comment)", joined, re.I):
        return "issue_box"
    return "card_text"


def _is_tiny_numeric_fragment(text: str, width: float, font_size: float) -> bool:
    clean = text.strip()
    if not clean:
        return True
    if _looks_short_metric_value(clean, width, font_size):
        return False
    if re.fullmatch(r"[\d\.\,%\+\-]{1,5}", clean) and width < 18:
        return True
    if re.fullmatch(r"[\d\.\,%\+\-]{1,3}", clean) and font_size < 8.5:
        return True
    return False


def _is_decorative_visual_fragment(width: float, height: float, area_ratio: float, pw: float, ph: float) -> bool:
    if width <= 0 or height <= 0:
        return True
    if area_ratio < 0.0005:
        return True
    if width < pw * 0.03 and height < ph * 0.03:
        return True
    if width > pw * 0.5 and height < ph * 0.008:
        return True
    if height > ph * 0.5 and width < pw * 0.008:
        return True
    return False


def _bbox_inner_coverage_ratio(inner: list[float], outer: list[float]) -> float:
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / inner_area


def _cluster_visual_panels(visuals: list[dict], pw: float, ph: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for visual in sorted(visuals, key=lambda item: ((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])), reverse=True):
        assigned = False
        for cluster in clusters:
            cluster_bbox = [
                min(item["bbox"][0] for item in cluster),
                min(item["bbox"][1] for item in cluster),
                max(item["bbox"][2] for item in cluster),
                max(item["bbox"][3] for item in cluster),
            ]
            h_gap, v_gap = _bbox_gap(visual["bbox"], cluster_bbox)
            overlap = _bbox_inner_coverage_ratio(visual["bbox"], cluster_bbox)
            same_band = abs(((visual["bbox"][1] + visual["bbox"][3]) / 2.0) - ((cluster_bbox[1] + cluster_bbox[3]) / 2.0)) < ph * 0.08
            if overlap > 0.2 or (h_gap < pw * 0.035 and v_gap < ph * 0.035) or (same_band and h_gap < pw * 0.05):
                cluster.append(visual)
                assigned = True
                break
        if not assigned:
            clusters.append([visual])
    return clusters


def _find_visual_host(cluster_bbox: list[float], content_blocks: list[dict], pw: float, ph: float) -> dict | None:
    best = None
    best_score = 0.0
    for block in content_blocks:
        if block["type"] not in ("text", "title", "table"):
            continue
        role = block.get("meta", {}).get("dashboard_role")
        if role == "footer":
            continue
        overlap = _bbox_inner_coverage_ratio(cluster_bbox, block["bbox"])
        h_gap, v_gap = _bbox_gap(cluster_bbox, block["bbox"])
        proximity = 1.0 if (h_gap < pw * 0.04 and v_gap < ph * 0.04) else 0.0
        score = overlap + proximity
        if role in ("kpi", "ranking", "mini_table", "issue_box", "card_text", "title"):
            score += 0.25
        if score > best_score:
            best = block
            best_score = score
    return best if best_score >= 0.35 else None


def _is_meaningful_visual_cluster(
    cluster_bbox: list[float],
    fragment_count: int,
    area_ratio: float,
    pw: float,
    ph: float,
) -> tuple[bool, str]:
    width = cluster_bbox[2] - cluster_bbox[0]
    height = cluster_bbox[3] - cluster_bbox[1]
    aspect_ratio = width / max(1.0, height)

    if area_ratio >= 0.006:
        return True, "panel_scale"
    if fragment_count >= 3 and area_ratio >= 0.002:
        return True, "fragment_cluster"
    if 0.4 <= aspect_ratio <= 3.2 and area_ratio >= 0.002:
        return True, "chart_like_shape"
    if width >= pw * 0.15 and height >= ph * 0.06:
        return True, "visual_panel_dimensions"
    return False, ""


def _collapse_dashboard_visuals(blocks: list[dict], pw: float, ph: float, merge_events: list) -> tuple[list[dict], int, int]:
    visuals = [block for block in blocks if block["type"] in ("image", "chart")]
    content = [block for block in blocks if block["type"] not in ("image", "chart")]
    if not visuals:
        return blocks, 0, 0

    merged_visuals: list[dict] = []
    absorbed_count = 0
    dropped_count = 0

    for cluster in _cluster_visual_panels(visuals, pw, ph):
        bbox = [
            min(item["bbox"][0] for item in cluster),
            min(item["bbox"][1] for item in cluster),
            max(item["bbox"][2] for item in cluster),
            max(item["bbox"][3] for item in cluster),
        ]
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        area_ratio = area / max(1.0, pw * ph)
        fragment_count = len(cluster)
        is_meaningful, preserve_reason = _is_meaningful_visual_cluster(bbox, fragment_count, area_ratio, pw, ph)
        host = _find_visual_host(bbox, content, pw, ph)

        if host and not is_meaningful and area_ratio < 0.004 and fragment_count <= 2:
            host_meta = host.setdefault("meta", {})
            host_meta["visual_cluster_count"] = host_meta.get("visual_cluster_count", 0) + fragment_count
            host_meta["visual_support_area_ratio"] = round(host_meta.get("visual_support_area_ratio", 0.0) + area_ratio, 5)
            summaries = [
                str(fragment.get("meta", {}).get("visual_summary") or fragment.get("meta", {}).get("caption_text") or "").strip()
                for fragment in cluster
                if str(fragment.get("meta", {}).get("visual_summary") or fragment.get("meta", {}).get("caption_text") or "").strip()
            ]
            if summaries and not host_meta.get("visual_summary"):
                host_meta["visual_summary"] = "\n".join(dict.fromkeys(summaries))
            for fragment in cluster:
                merge_events.append({"kept": host["id"], "dropped": fragment["id"], "reason": "dashboard_visual_absorbed"})
            absorbed_count += fragment_count
            continue

        if area_ratio < 0.0003:
            dropped_count += fragment_count
            continue

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / max(1.0, height)
        visual_type = "chart" if (0.7 <= aspect_ratio <= 2.4 and area_ratio >= 0.003) else "image"
        visual_summaries = [
            str(fragment.get("meta", {}).get("visual_summary") or fragment.get("meta", {}).get("caption_text") or "").strip()
            for fragment in cluster
            if str(fragment.get("meta", {}).get("visual_summary") or fragment.get("meta", {}).get("caption_text") or "").strip()
        ]
        context_ids: list[str] = []
        metrics: list[str] = []
        for fragment in cluster:
            fmeta = fragment.get("meta", {})
            for cid in fmeta.get("context_block_ids") or []:
                if cid not in context_ids:
                    context_ids.append(cid)
            for metric in fmeta.get("visible_key_metrics") or []:
                if metric not in metrics:
                    metrics.append(metric)
        visual_summary = "\n".join(dict.fromkeys(visual_summaries))

        merged_visuals.append({
            "id": cluster[0]["id"],
            "type": visual_type,
            "bbox": [round(v, 2) for v in bbox],
            "text": "",
            "source": "dashboard_visual_cluster",
            "confidence": max(fragment.get("score", 0.0) for fragment in cluster),
            "meta": {
                "dashboard_role": "visual_panel",
                "summary_priority": "low",
                "visual_fragment_count": fragment_count,
                "visual_area_ratio": round(area_ratio, 5),
                "visual_preserve_reason": preserve_reason or "non_decorative_cluster",
                "visual_summary": visual_summary,
                "caption_text": visual_summary,
                "context_block_ids": context_ids,
                "visible_key_metrics": metrics[:16],
            },
        })
        absorbed_count += max(0, fragment_count - 1)

    return content + merged_visuals, absorbed_count, dropped_count


def _restore_dashboard_context_block(blocks: list[dict], ph: float) -> tuple[list[dict], bool]:
    text_like = [block for block in blocks if block["type"] in ("text", "title")]
    if len(text_like) >= 1:
        return blocks, False

    candidates = []
    for block in blocks:
        if block["type"] == "footer":
            continue
        if block["type"] == "table":
            snippet = block.get("meta", {}).get("caption_text") or str(block.get("text", "")).splitlines()[0]
        else:
            snippet = str(block.get("text", "")).strip()
        snippet = snippet.strip()
        if snippet:
            candidates.append((block, snippet[:140]))

    if len(candidates) < 2:
        return blocks, False

    selected = candidates[:2]
    bbox = [
        min(item[0]["bbox"][0] for item in selected),
        max(0.0, min(item[0]["bbox"][1] for item in selected) - 8.0),
        max(item[0]["bbox"][2] for item in selected),
        min(ph, max(item[0]["bbox"][3] for item in selected) + 8.0),
    ]
    blocks.append({
        "id": f"{selected[0][0]['id']}_ctx",
        "type": "text",
        "bbox": [round(v, 2) for v in bbox],
        "text": "\n".join(snippet for _, snippet in selected),
        "source": "dashboard_context_bridge",
        "score": 0.75,
        "meta": {
            "dashboard_role": "context_bridge",
            "summary_priority": "medium",
        },
    })
    return blocks, True


def _postprocess_dashboard_blocks(blocks, pw, ph, layout_hint, merge_events, quality_notes):
    out: list[dict] = []
    for block in blocks:
        meta = block.setdefault("meta", {})
        role = meta.get("dashboard_role")
        text = str(block.get("text", "") or "").strip()
        if role in ("kpi", "issue_box", "mini_table", "ranking") and not meta.get("summary_exclude") and not meta.get("dashboard_overlaps_table_bbox"):
            meta["preserve_atomic"] = True
            meta["summary_priority"] = "high"
        elif meta.get("dashboard_overlaps_table_bbox"):
            meta["summary_priority"] = "low"
            meta["summary_exclude"] = True
            meta["summary_exclude_reason"] = "covered_by_table_block"
        if role == "mini_table" and block["type"] != "table":
            block["type"] = "table"
            if not meta.get("table_summary"):
                meta["table_summary"] = str(block.get("text", "") or "").strip()
        if role == "footer" or (block["bbox"][3] > ph * 0.92 and len(text) < 36):
            block["type"] = "footer"
            meta["summary_exclude"] = True
            meta["summary_priority"] = "low"
        out.append(block)

    out, collapsed_count, dropped_visual_count = _collapse_dashboard_visuals(out, pw, ph, merge_events)
    if collapsed_count:
        quality_notes.append("dashboard_visual_fragments_collapsed")
    if dropped_visual_count:
        quality_notes.append("dashboard_decorative_visuals_dropped")

    out, context_added = _restore_dashboard_context_block(out, ph)
    if context_added:
        quality_notes.append("dashboard_context_bridge_added")

    out.sort(key=lambda item: (
        0 if item["type"] == "title" else 2 if item["type"] == "footer" else 1,
        item["bbox"][1],
        item["bbox"][0],
    ))
    quality_notes.append("dashboard_pipeline_postprocess_applied")
    return out, "mixed_visual"
