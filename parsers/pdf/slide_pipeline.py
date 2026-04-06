import logging
import re

import fitz

from .utils import get_text_similarity, is_visual_noise_text

logger = logging.getLogger(__name__)


def process_slide_ir(doc, page_idx, plumber_tables):
    """Slide extraction separates title, KPI, bullet, footer, and decorative zones early."""
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    raw = page.get_text("dict") or {}

    text_blocks = _extract_slide_text_blocks(raw, page_idx, pw, ph)
    visual_candidates = _extract_visual_candidates(raw, page_idx, pw, ph)
    visual_blocks, visual_dropped = _filter_and_cluster_visuals(visual_candidates, pw, ph, page_idx)
    table_blocks = _build_slide_table_blocks(plumber_tables, page_idx, pw, ph)

    blocks = text_blocks + visual_blocks + table_blocks

    role = _estimate_page_role(blocks, page_idx, doc.page_count)
    salvage_applied = False
    salvage_source = "native"
    near_empty_reason = None

    text_content = "".join(block.get("text", "") for block in blocks if block["type"] in ("text", "title")).strip()
    title_text_count = sum(1 for b in blocks if b["type"] in ("text", "title"))
    image_count = sum(1 for b in blocks if b["type"] in ("image", "chart"))
    avg_image_area = 0.0
    if image_count > 0:
        avg_image_area = sum(
            max(0.0, (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))
            for b in blocks if b["type"] in ("image", "chart")
        ) / image_count / max(1.0, pw * ph)

    needs_salvage = (
        not blocks
        or len(text_content) < 50
        or (title_text_count == 0 and image_count > 0)
        or image_count >= 15
        or (image_count > 0 and avg_image_area < 0.02 and title_text_count == 0)
    )

    visual_cap_applied = False
    visual_before_cap = len(visual_blocks)
    cap_note_global: str | None = None

    if needs_salvage:
        near_empty_reason = (
            "no_blocks_found" if not blocks
            else "excessive_image_fragments" if image_count >= 15
            else "no_title_text_blocks" if title_text_count == 0
            else "sparse_text_content"
        )
        salvaged_blocks, salvage_source = _salvage_page(page, pw, ph)
        if salvaged_blocks:
            salvaged_text = "".join(b.get("text", "") for b in salvaged_blocks).strip()
            should_use_salvage = (
                len(salvaged_text) >= len(text_content)
                or (image_count >= 15 and len(salvaged_text) >= 30)
                or (len(text_content) < 50 and len(salvaged_text) >= 20)
                or (title_text_count == 0 and image_count > 0 and len(salvaged_text) >= 15)
            )
            if should_use_salvage:
                capped_visuals, cap_note_global = _cap_slide_visual_blocks(
                    visual_blocks, pw, ph, max_kept=10, image_count=len(visual_blocks)
                )
                visual_cap_applied = cap_note_global is not None
                blocks = salvaged_blocks + table_blocks + capped_visuals
                salvage_applied = True
                if cap_note_global:
                    near_empty_reason = (near_empty_reason or "") + ";" + cap_note_global

    slide_signals: list[str] = []
    for block in blocks:
        _classify_slide_block(block, slide_signals, ph)

    return {
        "page_num": page_idx + 1,
        "page_width": pw,
        "page_height": ph,
        "blocks": blocks,
        "_pipeline_postprocess": _postprocess_slide_blocks,
        "parser_debug": {
            "pipeline_used": "slide_ir_pipeline",
            "page_role": role,
            "block_count": len(blocks),
            "slide_like_signals": sorted(set(slide_signals)),
            "visual_preservation_reason": "title_kpi_bullet_footer_split",
            "empty_page_flag": not bool(blocks),
            "near_empty_reason": near_empty_reason,
            "salvage_applied": salvage_applied,
            "salvage_source": salvage_source,
            "preferred_layout_hint": "slide_like",
            "visual_candidate_count": len(visual_candidates),
            "visual_dropped_count": visual_dropped,
            "visual_kept_count": len([b for b in blocks if b["type"] in ("image", "chart")]),
            "visual_pre_cap_count": visual_before_cap,
            "visual_cap_applied": visual_cap_applied,
        },
    }


def _extract_slide_text_blocks(raw: dict, page_idx: int, pw: float, ph: float) -> list[dict]:
    """One output block per PyMuPDF text block (type 0), lines joined — reduces over-splitting on slides."""
    blocks: list[dict] = []
    bid = 0
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue

        line_texts: list[str] = []
        line_bboxes: list[list[float]] = []
        max_font = 0.0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not text:
                continue
            if is_visual_noise_text(text) and len(text) < 12:
                continue
            line_bboxes.append(list(line.get("bbox", [0, 0, 0, 0])))
            line_texts.append(text)
            for span in spans:
                max_font = max(max_font, float(span.get("size", 0.0) or 0.0))

        if not line_texts:
            continue

        combined = "\n".join(line_texts).strip()
        if not combined:
            continue
        xs0 = [b[0] for b in line_bboxes]
        ys0 = [b[1] for b in line_bboxes]
        xs1 = [b[2] for b in line_bboxes]
        ys1 = [b[3] for b in line_bboxes]
        bbox = [min(xs0), min(ys0), max(xs1), max(ys1)]
        role_hint = line_texts[0]
        role = _infer_slide_role(role_hint, bbox, pw, ph, max_font)
        if role == "decorative_noise":
            continue

        block_type = "title" if role == "title" else "footer" if role == "footer" else "text"
        blocks.append({
            "id": f"p{page_idx+1}_s{bid}",
            "type": block_type,
            "bbox": bbox,
            "text": combined,
            "source": "pymupdf_slide_native",
            "confidence": 1.0,
            "meta": {
                "slide_role": role,
                "font_size": round(max_font, 2),
                "summary_priority": "high" if role in ("title", "kpi_metric", "bullet_item") else "medium",
                "summary_exclude": role in ("footer", "decorative_noise"),
            },
        })
        bid += 1
    return blocks


def _extract_visual_candidates(raw: dict, page_idx: int, pw: float, ph: float) -> list[dict]:
    candidates: list[dict] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = list(block.get("bbox", [0, 0, 0, 0]))
        x0, y0, x1, y1 = bbox
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        area = width * height
        area_ratio = area / max(1.0, pw * ph)
        candidates.append({
            "bbox": bbox,
            "width": width,
            "height": height,
            "area_ratio": area_ratio,
        })
    return candidates


def _cap_slide_visual_blocks(
    visual_blocks: list[dict],
    pw: float,
    ph: float,
    max_kept: int = 10,
    image_count: int = 0,
) -> tuple[list[dict], str | None]:
    """When salvage recovered text but many visuals remain, keep largest panels only."""
    if not visual_blocks:
        return [], None
    need_cap = image_count > max_kept or image_count >= 12
    if not need_cap:
        return visual_blocks, None

    def _area(b: dict) -> float:
        bb = b.get("bbox") or [0, 0, 0, 0]
        return max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])

    ranked = sorted(visual_blocks, key=_area, reverse=True)
    kept = ranked[:max_kept]
    kept.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    note = f"slide_visual_capped_{len(visual_blocks)}_to_{len(kept)}"
    return kept, note


def _filter_and_cluster_visuals(
    candidates: list[dict], pw: float, ph: float, page_idx: int
) -> tuple[list[dict], int]:
    if not candidates:
        return [], 0

    filtered = []
    dropped = 0
    for c in candidates:
        if c["width"] <= 0 or c["height"] <= 0:
            dropped += 1
            continue
        if c["area_ratio"] < 0.005:
            dropped += 1
            continue
        if c["width"] < pw * 0.03 and c["height"] < ph * 0.03:
            dropped += 1
            continue
        if c["width"] > pw * 0.5 and c["height"] < ph * 0.008:
            dropped += 1
            continue
        if c["height"] > ph * 0.5 and c["width"] < pw * 0.008:
            dropped += 1
            continue
        filtered.append(c)

    if not filtered:
        return [], dropped

    clusters = _cluster_nearby_visuals(filtered, pw, ph)

    result_blocks: list[dict] = []
    for cluster in clusters:
        bbox = [
            min(c["bbox"][0] for c in cluster),
            min(c["bbox"][1] for c in cluster),
            max(c["bbox"][2] for c in cluster),
            max(c["bbox"][3] for c in cluster),
        ]
        cw = bbox[2] - bbox[0]
        ch = bbox[3] - bbox[1]
        area_ratio = (cw * ch) / max(1.0, pw * ph)
        aspect = cw / max(1.0, ch)
        fragment_count = len(cluster)

        if area_ratio < 0.003 and fragment_count <= 2:
            dropped += fragment_count
            continue

        visual_type = "chart" if (0.5 <= aspect <= 3.0 and area_ratio >= 0.005) else "image"

        result_blocks.append({
            "id": f"p{page_idx+1}_img{len(result_blocks)}",
            "type": visual_type,
            "bbox": [round(v, 2) for v in bbox],
            "text": "",
            "source": "pymupdf_slide_image",
            "confidence": 1.0,
            "meta": {
                "slide_role": "visual_panel",
                "summary_priority": "low",
                "visual_fragment_count": fragment_count,
                "visual_area_ratio": round(area_ratio, 5),
            },
        })

    return result_blocks, dropped


def _cluster_nearby_visuals(visuals: list[dict], pw: float, ph: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for v in sorted(visuals, key=lambda c: c["area_ratio"], reverse=True):
        assigned = False
        for cluster in clusters:
            cbbox = [
                min(c["bbox"][0] for c in cluster),
                min(c["bbox"][1] for c in cluster),
                max(c["bbox"][2] for c in cluster),
                max(c["bbox"][3] for c in cluster),
            ]
            h_gap = max(0.0, max(cbbox[0] - v["bbox"][2], v["bbox"][0] - cbbox[2]))
            v_gap = max(0.0, max(cbbox[1] - v["bbox"][3], v["bbox"][1] - cbbox[3]))
            vcy = (v["bbox"][1] + v["bbox"][3]) / 2.0
            ccy = (cbbox[1] + cbbox[3]) / 2.0
            same_band = abs(vcy - ccy) < ph * 0.1

            if h_gap < pw * 0.04 and v_gap < ph * 0.04:
                cluster.append(v)
                assigned = True
                break
            if same_band and h_gap < pw * 0.06:
                cluster.append(v)
                assigned = True
                break
        if not assigned:
            clusters.append([v])
    return clusters


def _build_slide_table_blocks(plumber_tables, page_idx: int, pw: float, ph: float) -> list[dict]:
    tables: list[dict] = []
    for i, tbl in enumerate(plumber_tables):
        cells = tbl.get("cells") or []
        flattened = [
            " | ".join(str(cell).strip() for cell in row if cell is not None and str(cell).strip())
            for row in cells
        ]
        tables.append({
            "id": f"p{page_idx+1}_t{i}",
            "type": "table",
            "bbox": list(tbl.get("bbox", [0, 0, pw, ph])),
            "text": "\n".join(row for row in flattened if row).strip() or "[Table Data]",
            "source": "pdfplumber_slide",
            "confidence": 1.0,
            "meta": {
                "slide_role": "table_panel",
                "normalized_table": cells,
                "summary_priority": "high",
            },
        })
    return tables


def _infer_slide_role(text: str, bbox: list[float], pw: float, ph: float, font_size: float) -> str:
    clean = text.strip()
    y0, y1 = bbox[1], bbox[3]
    width = bbox[2] - bbox[0]

    if y0 < ph * 0.16 and (font_size >= 14 or (len(clean) <= 80 and width > pw * 0.18)):
        return "title"
    if y1 > ph * 0.92 and len(clean) <= 40:
        return "footer"
    if re.search(r"[\d,]+(?:\.\d+)?\s*(%|원|억원|조원|USD|배|x|pt|bps)", clean, re.I) and len(clean) <= 48:
        return "kpi_metric"
    if clean.startswith(("▪", "•", "-", "□", "◦", "*", "+")):
        return "bullet_item"
    if len(clean) <= 8 and not re.search(r"[A-Za-z가-힣]{2,}", clean):
        return "decorative_noise"
    if width < pw * 0.12 and len(clean) < 16 and y0 < ph * 0.25:
        return "section_chip"
    return "body_note"


def _classify_slide_block(block, signals, ph: float):
    text = str(block.get("text", "") or "").strip()
    meta = block.setdefault("meta", {})
    role = meta.get("slide_role", "")

    if role == "title":
        block["type"] = "title"
        meta["summary_role"] = "title"
        meta["summary_priority"] = "high"
        signals.append("title_detected")
    if role == "kpi_metric":
        meta["summary_priority"] = "high"
        meta["preserve_atomic"] = True
        signals.append("kpi_detected")
    if role == "bullet_item":
        signals.append("bullet_list_detected")
    if role == "footer" or block["bbox"][3] > ph * 0.94:
        block["type"] = "footer"
        meta["summary_exclude"] = True
        meta["summary_priority"] = "low"
        signals.append("footer_detected")


def _postprocess_slide_blocks(blocks, pw, ph, layout_hint, merge_events, quality_notes):
    ordered = sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    out: list[dict] = []
    skip = set()

    for i, block in enumerate(ordered):
        if i in skip:
            continue
        current = block.copy()
        current["meta"] = dict(block.get("meta", {}))

        for j in range(i + 1, len(ordered)):
            if j in skip:
                continue
            nxt = ordered[j]
            cur_role = current.get("meta", {}).get("slide_role")
            nxt_role = nxt.get("meta", {}).get("slide_role")
            if current["type"] == "footer" or nxt["type"] == "footer":
                continue

            y_gap = nxt["bbox"][1] - current["bbox"][3]
            x_diff = abs(nxt["bbox"][0] - current["bbox"][0])

            mergeable = False
            if cur_role == "title" and nxt_role in ("bullet_item", "body_note") and -5 <= y_gap <= 24 and x_diff < 60:
                mergeable = True
            elif cur_role == "bullet_item" and nxt_role == "bullet_item" and -5 <= y_gap <= 20 and x_diff < 36:
                mergeable = True
            elif "kpi_metric" in (cur_role, nxt_role) and -6 <= y_gap <= 26 and x_diff < 70:
                mergeable = True

            if mergeable:
                current["text"] = current["text"].rstrip() + "\n" + nxt["text"].lstrip()
                current["bbox"] = [
                    min(current["bbox"][0], nxt["bbox"][0]),
                    min(current["bbox"][1], nxt["bbox"][1]),
                    max(current["bbox"][2], nxt["bbox"][2]),
                    max(current["bbox"][3], nxt["bbox"][3]),
                ]
                if cur_role != nxt_role and "kpi_metric" in (cur_role, nxt_role):
                    current["meta"]["slide_role"] = "kpi_group"
                    current["meta"]["summary_priority"] = "high"
                merge_events.append({"kept": current["id"], "dropped": nxt["id"], "reason": "slide_role_merge"})
                skip.add(j)
            elif y_gap > 80:
                break

        out.append(current)

    quality_notes.append("slide_pipeline_postprocess_applied")
    return out, "slide_like"


def _salvage_page(page, pw, ph) -> tuple[list[dict], str]:
    """Attempts multiple recovery strategies when standard extraction fails."""
    salvaged = []

    # Strategy 1: get_text("blocks") for structured text blocks
    try:
        raw_blocks = page.get_text("blocks")
        for b in raw_blocks:
            if b[6] != 0:  # type != text
                continue
            text = b[4].strip()
            if not text:
                continue
            salvaged.append({
                "id": f"salvage_b{len(salvaged)}",
                "type": "text",
                "bbox": list(b[:4]),
                "text": text,
                "source": "pymupdf_salvage_blocks",
                "confidence": 0.8,
                "meta": {"slide_role": "body_note"},
            })
    except Exception:
        pass

    # Strategy 2: get_text("words") word-level reassembly
    if not salvaged:
        try:
            words = page.get_text("words")
            if words:
                lines: dict[int, list] = {}
                for w in words:
                    y_key = int(w[1] / 8) * 8
                    lines.setdefault(y_key, []).append(w)
                for y_key in sorted(lines.keys()):
                    group = sorted(lines[y_key], key=lambda x: x[0])
                    text = " ".join(w[4] for w in group).strip()
                    if not text:
                        continue
                    bbox = [
                        min(w[0] for w in group),
                        min(w[1] for w in group),
                        max(w[2] for w in group),
                        max(w[3] for w in group),
                    ]
                    salvaged.append({
                        "id": f"salvage_w{len(salvaged)}",
                        "type": "text",
                        "bbox": bbox,
                        "text": text,
                        "source": "pymupdf_salvage_words",
                        "confidence": 0.7,
                        "meta": {"slide_role": "body_note"},
                    })
        except Exception:
            pass

    # Strategy 3: rawdict line/span reassembly
    if not salvaged:
        try:
            full_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_IMAGES)
            for block in full_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text", "")).strip()
                        if not text:
                            continue
                        if any(get_text_similarity(text, existing["text"]) > 0.9 for existing in salvaged):
                            continue
                        salvaged.append({
                            "id": f"salvage_s{len(salvaged)}",
                            "type": "text",
                            "bbox": list(span.get("bbox", [0, 0, 0, 0])),
                            "text": text,
                            "source": "pymupdf_salvage_spans",
                            "confidence": 0.6,
                            "meta": {"slide_role": "body_note"},
                        })
        except Exception:
            pass

    salvaged.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    def _combined_len(items: list[dict]) -> int:
        return len("".join(b.get("text", "") for b in items).strip())

    rect = page.rect
    page_bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
    try:
        plain = (page.get_text("text") or "").strip()
    except Exception:
        plain = ""

    if plain:
        clen = _combined_len(salvaged)
        # Prefer plain when salvage failed, or full-page text is materially longer (keep block bboxes).
        if not salvaged or len(plain) > max(clen, 1) * 5 // 4:
            salvaged = [{
                "id": "salvage_plain",
                "type": "text",
                "bbox": list(page_bbox),
                "text": plain,
                "source": "pymupdf_plain_text_fallback",
                "confidence": 0.65,
                "meta": {"slide_role": "body_note"},
            }]
            source = "pymupdf_plain_text_fallback"
        elif salvaged:
            src_types = [b["source"] for b in salvaged]
            if any("blocks" in s for s in src_types):
                source = "native_blocks_fallback"
            elif any("words" in s for s in src_types):
                source = "native_words_fallback"
            else:
                source = "native_spans_fallback"
        else:
            source = "none"
    elif salvaged:
        src_types = [b["source"] for b in salvaged]
        if any("blocks" in s for s in src_types):
            source = "native_blocks_fallback"
        elif any("words" in s for s in src_types):
            source = "native_words_fallback"
        else:
            source = "native_spans_fallback"
    else:
        source = "none"
    return salvaged, source


def _estimate_page_role(blocks, page_num, total_pages):
    all_text = " ".join(str(block.get("text", "")) for block in blocks).lower().replace(" ", "")
    if page_num == 0:
        return "cover"
    if page_num >= total_pages - 1:
        return "disclaimer"
    if any(keyword in all_text for keyword in ["목차", "contents", "index"]):
        return "toc"
    if len(blocks) < 5:
        return "section_divider"
    return "body"
