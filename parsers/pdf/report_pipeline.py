import logging
import re

logger = logging.getLogger(__name__)


def process_text_report(doc, page_idx, plumber_tables):
    """
    Report extraction keeps paragraph-scale native blocks, preserves column hints,
    and attaches caption/table context before common normalization.
    """
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    raw = page.get_text("dict") or {}

    text_blocks = _extract_report_text_blocks(raw, page_idx, pw, ph)
    table_blocks = _build_report_table_blocks(plumber_tables, page_idx, pw, ph)
    visual_blocks = _extract_report_visual_blocks(raw, page_idx, pw, ph)
    _attach_caption_context(text_blocks, table_blocks)
    _attach_visual_caption_context(text_blocks, visual_blocks)

    preferred_layout_hint = "multi_column" if _looks_multi_column(text_blocks, pw) else "single_column"
    blocks = text_blocks + table_blocks + visual_blocks

    return {
        "page_num": page_idx + 1,
        "page_width": pw,
        "page_height": ph,
        "blocks": blocks,
        "_pipeline_postprocess": _postprocess_report_blocks,
        "parser_debug": {
            "pipeline_used": "text_report_pipeline",
            "block_count": len(blocks),
            "preferred_layout_hint": preferred_layout_hint,
            "report_signals": ["paragraph_native_blocks", "caption_table_linking"],
        },
    }


def _extract_report_text_blocks(raw: dict, page_idx: int, pw: float, ph: float) -> list[dict]:
    blocks: list[dict] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_lines = []
        max_font = 0.0
        min_font = 999.0
        for line in block.get("lines", []):
            line_text = "".join(str(span.get("text", "")) for span in line.get("spans", [])).strip()
            if not line_text:
                continue
            block_lines.append(line_text)
            for span in line.get("spans", []):
                size = float(span.get("size", 0.0) or 0.0)
                max_font = max(max_font, size)
                min_font = min(min_font, size)

        text = "\n".join(block_lines).strip()
        if not text:
            continue

        bbox = list(block.get("bbox", [0, 0, 0, 0]))
        role, block_type = _classify_report_block(text, bbox, pw, ph, max_font, min_font if min_font < 999 else max_font)
        blocks.append({
            "id": f"p{page_idx+1}_r{len(blocks)}",
            "type": block_type,
            "bbox": bbox,
            "text": text,
            "source": "pymupdf_report_native",
            "confidence": 1.0,
            "meta": {
                "report_role": role,
                "font_size_max": round(max_font, 2),
                "summary_priority": "high" if role in ("section_title", "caption") else "medium",
                "summary_exclude": role == "footer_note",
                "column_hint": 0 if ((bbox[0] + bbox[2]) / 2.0) < (pw / 2.0) else 1,
            },
        })
    return blocks


def _build_report_table_blocks(plumber_tables, page_idx: int, pw: float, ph: float) -> list[dict]:
    tables: list[dict] = []
    for i, tbl in enumerate(plumber_tables):
        cells = tbl.get("cells") or []
        flattened = [
            " | ".join(str(cell).strip() for cell in row if cell is not None and str(cell).strip())
            for row in cells
        ]
        text = "\n".join(row for row in flattened if row).strip() or "[Table Data]"
        bbox = list(tbl.get("bbox", [0, 0, pw, ph]))
        tables.append({
            "id": f"p{page_idx+1}_t{i}",
            "type": "table",
            "bbox": bbox,
            "text": text,
            "source": "pdfplumber_report",
            "confidence": 1.0,
            "meta": {
                "report_role": "table",
                "normalized_table": cells,
                "summary_priority": "medium",
                "column_hint": 0 if ((bbox[0] + bbox[2]) / 2.0) < (pw / 2.0) else 1,
            },
        })
    return tables


def _extract_report_visual_blocks(raw: dict, page_idx: int, pw: float, ph: float) -> list[dict]:
    visuals: list[dict] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = list(block.get("bbox", [0, 0, 0, 0]))
        x0, y0, x1, y1 = bbox
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        area_ratio = (width * height) / max(1.0, pw * ph)
        if area_ratio < 0.003 and width < pw * 0.05 and height < ph * 0.05:
            continue
        aspect = width / max(1.0, height)
        visual_type = "chart" if (0.5 <= aspect <= 3.0 and area_ratio >= 0.01) else "image"
        visuals.append({
            "id": f"p{page_idx+1}_img{len(visuals)}",
            "type": visual_type,
            "bbox": bbox,
            "text": "",
            "source": "pymupdf_report_image",
            "confidence": 1.0,
            "meta": {
                "report_role": "visual",
                "summary_priority": "low",
                "visual_area_ratio": round(area_ratio, 5),
            },
        })
    return visuals


def _classify_report_block(text: str, bbox: list[float], pw: float, ph: float, max_font: float, min_font: float) -> tuple[str, str]:
    clean = text.strip()
    y0, y1 = bbox[1], bbox[3]
    line_count = len([line for line in clean.splitlines() if line.strip()])

    if y0 < ph * 0.12 and max_font >= 13.5 and len(clean) <= 120:
        return "section_title", "title"
    if y1 > ph * 0.92 and len(clean) <= 60:
        return "footer_note", "footer"
    if re.match(r"^\s*(표|table|figure|그림)\s*[\dA-Za-z\-\.]+", clean, re.I):
        return "caption", "text"
    if line_count >= 2 and len(clean) >= 120:
        return "paragraph", "text"
    if max_font - min_font >= 2.5 and len(clean) <= 90:
        return "section_title", "title"
    return "body_text", "text"


def _attach_caption_context(text_blocks: list[dict], table_blocks: list[dict]) -> None:
    captions = [block for block in text_blocks if block.get("meta", {}).get("report_role") == "caption"]
    for table in table_blocks:
        tx0, ty0, tx1, ty1 = table["bbox"]
        best_caption = None
        best_dist = float("inf")
        for caption in captions:
            cx0, cy0, cx1, cy1 = caption["bbox"]
            same_column = abs(((cx0 + cx1) / 2.0) - ((tx0 + tx1) / 2.0)) < max((tx1 - tx0) * 0.8, 80)
            vertical_gap = min(abs(ty0 - cy1), abs(cy0 - ty1))
            if same_column and vertical_gap < 60 and vertical_gap < best_dist:
                best_caption = caption
                best_dist = vertical_gap
        if best_caption:
            table["meta"]["caption_text"] = best_caption["text"]
            table["meta"]["summary_priority"] = "high"
            best_caption["meta"]["attached_to_table"] = True
            best_caption["meta"]["summary_priority"] = "high"


def _attach_visual_caption_context(text_blocks: list[dict], visual_blocks: list[dict]) -> None:
    """Find nearby caption/text blocks for visuals so they can be included in rag_text."""
    captions = [b for b in text_blocks if b.get("meta", {}).get("report_role") == "caption"]
    body_texts = [b for b in text_blocks if b.get("meta", {}).get("report_role") in ("paragraph", "body_text")]

    for visual in visual_blocks:
        vx0, vy0, vx1, vy1 = visual["bbox"]
        best = None
        best_dist = float("inf")

        for caption in captions:
            cx0, cy0, cx1, cy1 = caption["bbox"]
            same_column = abs(((cx0 + cx1) / 2.0) - ((vx0 + vx1) / 2.0)) < max((vx1 - vx0) * 0.8, 80)
            vertical_gap = min(abs(vy0 - cy1), abs(cy0 - vy1))
            if same_column and vertical_gap < 80 and vertical_gap < best_dist:
                best = caption
                best_dist = vertical_gap

        if not best:
            for body in body_texts:
                bx0, by0, bx1, by1 = body["bbox"]
                vertical_gap = min(abs(vy0 - by1), abs(by0 - vy1))
                same_column = abs(((bx0 + bx1) / 2.0) - ((vx0 + vx1) / 2.0)) < max((vx1 - vx0) * 0.8, 80)
                if same_column and vertical_gap < 40 and vertical_gap < best_dist:
                    best = body
                    best_dist = vertical_gap

        if best:
            visual["meta"]["caption_text"] = (best.get("text") or "")[:200]


def _looks_multi_column(text_blocks: list[dict], pw: float) -> bool:
    candidates = [
        block for block in text_blocks
        if block["type"] == "text" and (block["bbox"][2] - block["bbox"][0]) < pw * 0.48
    ]
    if len(candidates) < 4:
        return False
    left = sum(1 for block in candidates if block["bbox"][0] < pw * 0.32)
    right = sum(1 for block in candidates if block["bbox"][0] > pw * 0.52)
    return left >= 2 and right >= 2


def _postprocess_report_blocks(blocks, pw, ph, layout_hint, merge_events, quality_notes):
    text_blocks = [block for block in blocks if block["type"] in ("text", "title", "footer")]
    other_blocks = [block for block in blocks if block["type"] not in ("text", "title", "footer")]

    merged: list[dict] = []
    used = set()
    text_blocks.sort(key=lambda item: (
        item.get("meta", {}).get("column_hint", 0),
        item["bbox"][1],
        item["bbox"][0],
    ))

    for i, block in enumerate(text_blocks):
        if i in used:
            continue
        current = block.copy()
        current["meta"] = dict(block.get("meta", {}))

        for j in range(i + 1, len(text_blocks)):
            if j in used:
                continue
            nxt = text_blocks[j]
            cur_meta = current.get("meta", {})
            nxt_meta = nxt.get("meta", {})
            if cur_meta.get("column_hint") != nxt_meta.get("column_hint"):
                continue
            if current["type"] != "text" or nxt["type"] != "text":
                continue
            if cur_meta.get("report_role") in ("caption", "footer_note") or nxt_meta.get("report_role") in ("caption", "footer_note"):
                continue

            y_gap = nxt["bbox"][1] - current["bbox"][3]
            x_diff = abs(nxt["bbox"][0] - current["bbox"][0])
            width_ratio = abs((nxt["bbox"][2] - nxt["bbox"][0]) - (current["bbox"][2] - current["bbox"][0])) / max(1.0, (current["bbox"][2] - current["bbox"][0]))

            if -3 <= y_gap <= 18 and x_diff < 12 and width_ratio < 0.25:
                current["text"] = current["text"].rstrip() + "\n" + nxt["text"].lstrip()
                current["bbox"] = [
                    min(current["bbox"][0], nxt["bbox"][0]),
                    min(current["bbox"][1], nxt["bbox"][1]),
                    max(current["bbox"][2], nxt["bbox"][2]),
                    max(current["bbox"][3], nxt["bbox"][3]),
                ]
                merge_events.append({"kept": current["id"], "dropped": nxt["id"], "reason": "report_paragraph_merge"})
                used.add(j)
            elif y_gap > 28:
                break

        merged.append(current)

    for block in merged:
        meta = block.setdefault("meta", {})
        if meta.get("report_role") == "footer_note":
            block["type"] = "footer"
            meta["summary_exclude"] = True
            meta["summary_priority"] = "low"

    quality_notes.append("report_pipeline_postprocess_applied")
    return merged + other_blocks, ("multi_column" if _looks_multi_column(merged, pw) else layout_hint)
