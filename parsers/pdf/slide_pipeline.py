import logging
import re

import fitz
import numpy as np

from .utils import get_text_similarity, is_visual_noise_text
from .ppt_export_native import recover_ppt_export_blocks

logger = logging.getLogger(__name__)


# ── page-level sub-mode classifier ──────────────────────────────────

def _classify_slide_page_mode(
    native_text_chars: int,
    native_word_count: int,
    text_block_count: int,
    image_block_count: int,
    image_candidate_count: int,
    avg_image_area: float,
) -> str:
    """
    Classify a slide page into a sub-mode based purely on structural signals.

    Returns
    -------
    "native_text_slide"  – native text is rich enough; no OCR needed
    "ocr_slide"          – text-sparse, image-fragment-heavy; OCR is primary
    "hybrid_slide"       – in-between; try native first, OCR if quality poor
    """
    # ── native_text_slide: text is abundant ──
    if native_text_chars >= 120 or native_word_count >= 40:
        return "native_text_slide"
    if text_block_count >= 4 and native_text_chars >= 60:
        return "native_text_slide"

    # ── ocr_slide: extremely sparse text + heavy image fragments ──
    if native_text_chars < 40 and native_word_count < 15:
        if image_candidate_count >= 50 or image_block_count >= 15:
            return "ocr_slide"
        if image_block_count > 0 and text_block_count == 0:
            return "ocr_slide"
        if native_text_chars < 10:
            return "ocr_slide"

    # ── hybrid_slide: ambiguous ──
    return "hybrid_slide"


# ── OCR salvage ─────────────────────────────────────────────────────

def _run_ocr_salvage(
    page: fitz.Page,
    pw: float,
    ph: float,
    page_idx: int,
    zoom: float = 2.0,
) -> dict:
    """
    Render the page to a pixmap, run OCR, and reconstruct slide blocks.

    Returns dict with keys: blocks, engine, box_count, block_count, success, error,
    plus OCR preprocessing debug fields.
    """
    result = {
        "blocks": [],
        "engine": None,
        "box_count": 0,
        "block_count": 0,
        "success": False,
        "error": None,
        # OCR preprocessing debug
        "ocr_preprocess_variant": None,
        "ocr_variant_candidates": [],
        "ocr_variant_scores": {},
        "ocr_selected_reason": None,
        "ocr_confidence_avg": 0.0,
        "ocr_box_count_after_cleanup": 0,
    }
    try:
        from ..ocr_utils import run_ocr_on_image

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Convert RGBA → RGB if needed
        if pix.n == 4:
            img_arr = img_arr[:, :, :3]

        ocr_result = run_ocr_on_image(
            img_arr,
            page_width=pw,
            page_height=ph,
            zoom=zoom,
        )

        # Capture preprocessing debug fields
        result["ocr_preprocess_variant"] = ocr_result.get("preprocess_variant")
        result["ocr_variant_candidates"] = ocr_result.get("variant_candidates", [])
        result["ocr_variant_scores"] = ocr_result.get("variant_scores", {})
        result["ocr_selected_reason"] = ocr_result.get("selected_reason")
        result["ocr_confidence_avg"] = ocr_result.get("confidence", 0.0)
        result["ocr_box_count_after_cleanup"] = ocr_result.get("box_count_after_cleanup", 0)

        if not ocr_result.get("success") or not ocr_result.get("boxes"):
            result["error"] = ocr_result.get("error", "no boxes returned")
            return result

        boxes = ocr_result["boxes"]
        result["engine"] = ocr_result.get("engine")
        result["box_count"] = len(boxes)

        # Reconstruct slide blocks from OCR boxes
        slide_blocks = _ocr_boxes_to_slide_blocks(boxes, page_idx, pw, ph)
        result["blocks"] = slide_blocks
        result["block_count"] = len(slide_blocks)
        result["success"] = len(slide_blocks) > 0

    except Exception as exc:
        logger.warning("OCR salvage failed on page %d: %s", page_idx + 1, exc)
        result["error"] = str(exc)

    return result


# ── OCR box → slide block reconstruction ────────────────────────────

def _ocr_boxes_to_slide_blocks(
    boxes: list[dict],
    page_idx: int,
    pw: float,
    ph: float,
) -> list[dict]:
    """
    Convert raw OCR boxes into structured slide blocks.

    Steps:
    1. Filter micro-fragments
    2. Cluster boxes into lines (similar y-center)
    3. Merge lines into blocks (similar x-start, small y-gap)
    4. Assign slide roles (title / footer / body_note)
    """
    if not boxes:
        return []

    # Step 1: Filter micro-fragments
    filtered = []
    for box in boxes:
        text = box.get("text", "").strip()
        conf = box.get("confidence", 0.0)
        if not text:
            continue
        # Skip very short low-confidence fragments
        if len(text) <= 2 and conf < 0.6:
            continue
        if len(text) <= 1 and conf < 0.8:
            continue
        filtered.append(box)

    if not filtered:
        return []

    # Step 2: Cluster into lines
    lines = _cluster_ocr_into_lines(filtered, ph)

    # Step 3: Merge lines into blocks
    blocks_raw = _merge_lines_into_blocks(lines, pw, ph)

    # Step 4: Build slide blocks with role assignment
    slide_blocks = []
    bid = 0
    for block_data in blocks_raw:
        text = block_data["text"].strip()
        if not text:
            continue
        # Skip tiny fragments after merge
        if len(text) < 3 and not re.search(r'[가-힣a-zA-Z]{2,}', text):
            continue

        bbox = block_data["bbox"]
        role = _infer_ocr_slide_role(text, bbox, pw, ph, block_data.get("avg_height", 12.0))

        if role == "decorative_noise":
            continue

        block_type = "title" if role == "title" else "footer" if role == "footer" else "text"
        slide_blocks.append({
            "id": f"p{page_idx + 1}_ocr{bid}",
            "type": block_type,
            "bbox": [round(v, 2) for v in bbox],
            "text": text,
            "source": "ocr_slide_salvage",
            "confidence": round(block_data.get("avg_confidence", 0.7), 4),
            "meta": {
                "slide_role": role,
                "ocr_line_count": block_data.get("line_count", 1),
                "summary_priority": "high" if role in ("title", "kpi_metric", "bullet_item") else "medium",
                "summary_exclude": role in ("footer", "decorative_noise"),
            },
        })
        bid += 1

    return slide_blocks


def _cluster_ocr_into_lines(boxes: list[dict], ph: float) -> list[list[dict]]:
    """Cluster OCR boxes into horizontal lines by y-center proximity."""
    if not boxes:
        return []

    # Sort by y-center then x
    sorted_boxes = sorted(boxes, key=lambda b: ((b["bbox"][1] + b["bbox"][3]) / 2.0, b["bbox"][0]))

    y_tolerance = max(ph * 0.012, 6.0)  # ~1.2% of page height or 6pt
    lines: list[list[dict]] = []
    current_line: list[dict] = [sorted_boxes[0]]
    current_cy = (sorted_boxes[0]["bbox"][1] + sorted_boxes[0]["bbox"][3]) / 2.0

    for box in sorted_boxes[1:]:
        box_cy = (box["bbox"][1] + box["bbox"][3]) / 2.0
        if abs(box_cy - current_cy) <= y_tolerance:
            current_line.append(box)
            # Update center as running average
            current_cy = sum((b["bbox"][1] + b["bbox"][3]) / 2.0 for b in current_line) / len(current_line)
        else:
            lines.append(sorted(current_line, key=lambda b: b["bbox"][0]))
            current_line = [box]
            current_cy = box_cy

    if current_line:
        lines.append(sorted(current_line, key=lambda b: b["bbox"][0]))

    return lines


def _merge_lines_into_blocks(lines: list[list[dict]], pw: float, ph: float) -> list[dict]:
    """
    Merge adjacent lines into blocks based on spatial proximity.

    Lines with similar x-start and small vertical gap are merged.
    """
    if not lines:
        return []

    # Build line records with aggregate info
    line_records = []
    for line_boxes in lines:
        x0 = min(b["bbox"][0] for b in line_boxes)
        y0 = min(b["bbox"][1] for b in line_boxes)
        x1 = max(b["bbox"][2] for b in line_boxes)
        y1 = max(b["bbox"][3] for b in line_boxes)
        text = " ".join(b["text"] for b in line_boxes)
        avg_conf = sum(b.get("confidence", 0.7) for b in line_boxes) / len(line_boxes)
        avg_h = (y1 - y0) if (y1 - y0) > 0 else 10.0
        line_records.append({
            "bbox": [x0, y0, x1, y1],
            "text": text,
            "avg_confidence": avg_conf,
            "avg_height": avg_h,
            "box_count": len(line_boxes),
        })

    # Sort by y then x
    line_records.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    # Merge adjacent lines into blocks
    max_y_gap = max(ph * 0.025, 14.0)  # ~2.5% of page height
    max_x_drift = max(pw * 0.15, 60.0)  # x-start tolerance

    blocks: list[dict] = []
    current = {
        "bbox": list(line_records[0]["bbox"]),
        "texts": [line_records[0]["text"]],
        "confidences": [line_records[0]["avg_confidence"]],
        "heights": [line_records[0]["avg_height"]],
        "line_count": 1,
    }

    for rec in line_records[1:]:
        y_gap = rec["bbox"][1] - current["bbox"][3]
        x_drift = abs(rec["bbox"][0] - current["bbox"][0])

        can_merge = (
            y_gap <= max_y_gap
            and y_gap >= -5  # allow small overlap
            and x_drift <= max_x_drift
        )

        if can_merge:
            current["bbox"][0] = min(current["bbox"][0], rec["bbox"][0])
            current["bbox"][1] = min(current["bbox"][1], rec["bbox"][1])
            current["bbox"][2] = max(current["bbox"][2], rec["bbox"][2])
            current["bbox"][3] = max(current["bbox"][3], rec["bbox"][3])
            current["texts"].append(rec["text"])
            current["confidences"].append(rec["avg_confidence"])
            current["heights"].append(rec["avg_height"])
            current["line_count"] += 1
        else:
            blocks.append(_finalize_block(current))
            current = {
                "bbox": list(rec["bbox"]),
                "texts": [rec["text"]],
                "confidences": [rec["avg_confidence"]],
                "heights": [rec["avg_height"]],
                "line_count": 1,
            }

    blocks.append(_finalize_block(current))
    return blocks


def _finalize_block(cur: dict) -> dict:
    return {
        "bbox": cur["bbox"],
        "text": "\n".join(cur["texts"]),
        "avg_confidence": sum(cur["confidences"]) / len(cur["confidences"]),
        "avg_height": sum(cur["heights"]) / len(cur["heights"]),
        "line_count": cur["line_count"],
    }


def _infer_ocr_slide_role(
    text: str,
    bbox: list[float],
    pw: float,
    ph: float,
    avg_height: float,
) -> str:
    """Assign slide role to an OCR-reconstructed block, based on position and structure."""
    clean = text.strip()
    y0, y1 = bbox[1], bbox[3]
    width = bbox[2] - bbox[0]
    line_count = clean.count("\n") + 1

    # Title: near top, relatively large text, short
    if y0 < ph * 0.18 and line_count <= 3 and len(clean) <= 100:
        if avg_height >= 14.0 or width > pw * 0.20:
            return "title"

    # Footer: near bottom, short
    if y1 > ph * 0.90 and len(clean) <= 60:
        return "footer"

    # KPI metric
    if re.search(r"[\d,]+(?:\.\d+)?\s*(%|원|억원|조원|USD|배|x|pt|bps|bp)", clean, re.I) and len(clean) <= 80:
        return "kpi_metric"

    # Bullet
    if clean[0:1] in "▪•-□◦*+":
        return "bullet_item"

    # Decorative noise: very short, no meaningful characters
    if len(clean) <= 4 and not re.search(r'[A-Za-z가-힣]{2,}', clean):
        return "decorative_noise"

    return "body_note"


# ── main entry point ────────────────────────────────────────────────

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

    # ── Structural signal collection ──
    text_content = "".join(block.get("text", "") for block in blocks if block["type"] in ("text", "title")).strip()
    native_text_chars = len(re.sub(r'\s', '', text_content))
    native_word_count = len(text_content.split()) if text_content else 0
    title_text_count = sum(1 for b in blocks if b["type"] in ("text", "title"))
    image_count = sum(1 for b in blocks if b["type"] in ("image", "chart"))
    avg_image_area = 0.0
    if image_count > 0:
        avg_image_area = sum(
            max(0.0, (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))
            for b in blocks if b["type"] in ("image", "chart")
        ) / image_count / max(1.0, pw * ph)

    # ── Page mode classification ──
    page_mode = _classify_slide_page_mode(
        native_text_chars=native_text_chars,
        native_word_count=native_word_count,
        text_block_count=title_text_count,
        image_block_count=image_count,
        image_candidate_count=len(visual_candidates),
        avg_image_area=avg_image_area,
    )

    role = _estimate_page_role(blocks, page_idx, doc.page_count)
    salvage_applied = False
    salvage_source = "native"
    near_empty_reason = None
    chosen_text_source = "native"
    chosen_reason = "native_text_sufficient"

    # ── OCR tracking ──
    ocr_attempted = False
    ocr_engine = None
    ocr_box_count = 0
    ocr_block_count = 0
    ocr_preprocess_variant = None
    ocr_variant_candidates: list[str] = []
    ocr_variant_scores: dict = {}
    ocr_selected_reason = None
    ocr_confidence_avg = 0.0
    ocr_box_count_after_cleanup = 0

    # ── Salvage logic ──
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
        native_salvage_applied = False
        native_salvage_source = "none"
        near_empty_reason = (
            "no_blocks_found" if not blocks
            else "excessive_image_fragments" if image_count >= 15
            else "no_title_text_blocks" if title_text_count == 0
            else "sparse_text_content"
        )

        # ── Step 1: Default Native Salvage ──
        salvaged_blocks, native_salvage_source = _salvage_page(page, pw, ph)
        native_salvaged_text = "".join(b.get("text", "") for b in salvaged_blocks).strip() if salvaged_blocks else ""

        # ── Step 2: NEW PPT-Export Native Recovery ──
        ppt_recovery = recover_ppt_export_blocks(page, page_idx, pw, ph)
        ppt_blocks = ppt_recovery.get("blocks", [])
        ppt_text = "".join(b.get("text", "") for b in ppt_blocks).strip()
        ppt_recovered_len = ppt_recovery.get("recovered_text_len", 0)

        # ── Step 3: OCR salvage (Conditional / Final Fallback) ──
        ocr_blocks: list[dict] = []
        ocr_salvaged_text = ""

        # SKIP OCR if PPT recovery was exceptionally good (length > 150 and structurally stable)
        # Or if we are in hybrid mode and PPT recovery already gave us enough
        ocr_skipped_due_to_native_recovery = False
        ocr_skip_reason = None

        if (
            ppt_recovered_len > 250
            or native_text_chars > 80
            or (page_mode == "hybrid_slide" and ppt_recovered_len >= 45)
            or (page_mode == "ocr_slide" and ppt_recovered_len >= 50)
        ):
            ocr_skipped_due_to_native_recovery = True
            ocr_skip_reason = f"native_or_ppt_recovery_sufficient_{max(native_text_chars, ppt_recovered_len)}"

        if page_mode in ("ocr_slide", "hybrid_slide") and not ocr_skipped_due_to_native_recovery:
            ocr_attempted = True
            ocr_result = _run_ocr_salvage(page, pw, ph, page_idx, zoom=2.0)
            ocr_engine = ocr_result.get("engine")
            ocr_box_count = ocr_result.get("box_count", 0)
            ocr_block_count = ocr_result.get("block_count", 0)

            # Capture preprocessing debug
            ocr_preprocess_variant = ocr_result.get("ocr_preprocess_variant")
            ocr_variant_candidates = ocr_result.get("ocr_variant_candidates", [])
            ocr_variant_scores = ocr_result.get("ocr_variant_scores", {})
            ocr_selected_reason = ocr_result.get("ocr_selected_reason")
            ocr_confidence_avg = ocr_result.get("ocr_confidence_avg", 0.0)
            ocr_box_count_after_cleanup = ocr_result.get("ocr_box_count_after_cleanup", 0)

            if ocr_result.get("success"):
                ocr_blocks = ocr_result["blocks"]
                ocr_salvaged_text = "".join(b.get("text", "") for b in ocr_blocks).strip()

        # ── Choose best source ──
        native_len = len(text_content)
        native_salvage_len = len(native_salvaged_text)
        ppt_salvage_len = len(re.sub(r'\s', '', ppt_text))
        ocr_salvage_len = len(re.sub(r'\s', '', ocr_salvaged_text))

        best_source = "native"
        best_blocks = blocks
        best_len = native_len
        best_reason = "native_text_best"

        # Evaluate native salvage (Legacy)
        if native_salvage_len > best_len:
            best_source = "native_salvage"
            best_len = native_salvage_len
            best_reason = "native_salvage_recovered_more_text"

        # Evaluate PPT recovery (New)
        if ppt_salvage_len > best_len * 1.1 or (best_len < 50 and ppt_salvage_len > 30):
            best_source = "ppt_export_native"
            best_len = ppt_salvage_len
            best_reason = "ppt_export_native_recovered_structured_text"

        # Evaluate OCR salvage (Final)
        # Only take OCR if it's significantly better than native recovery
        # (OCR is noisy, so we give native recovery a 1.2x advantage)
        if ocr_salvage_len > (best_len * 1.2) and ocr_salvage_len >= 50:
            best_source = "ocr_salvage"
            best_len = ocr_salvage_len
            best_reason = "ocr_salvage_recovered_significantly_more_text"

        # Apply chosen source
        if best_source == "ocr_salvage" and ocr_blocks:
            capped_visuals, cap_note_global = _cap_slide_visual_blocks(
                visual_blocks, pw, ph, max_kept=8, image_count=len(visual_blocks)
            )
            visual_cap_applied = cap_note_global is not None
            blocks = ocr_blocks + table_blocks + capped_visuals
            salvage_applied = True
            salvage_source = "ocr_salvage"
            chosen_text_source = "ocr_salvage"
            chosen_reason = best_reason
            if cap_note_global:
                near_empty_reason = (near_empty_reason or "") + ";" + cap_note_global

        elif best_source == "ppt_export_native" and ppt_blocks:
            capped_visuals, cap_note_global = _cap_slide_visual_blocks(
                visual_blocks, pw, ph, max_kept=10, image_count=len(visual_blocks)
            )
            visual_cap_applied = cap_note_global is not None
            blocks = ppt_blocks + table_blocks + capped_visuals
            salvage_applied = True
            salvage_source = "ppt_export_native"
            chosen_text_source = "ppt_export_native"
            chosen_reason = best_reason
            if cap_note_global:
                near_empty_reason = (near_empty_reason or "") + ";" + cap_note_global

        elif best_source == "native_salvage" and salvaged_blocks:
            should_use = (
                native_salvage_len >= native_len
                or (image_count >= 15 and native_salvage_len >= 30)
                or (native_len < 50 and native_salvage_len >= 20)
                or (title_text_count == 0 and image_count > 0 and native_salvage_len >= 15)
            )
            if should_use:
                capped_visuals, cap_note_global = _cap_slide_visual_blocks(
                    visual_blocks, pw, ph, max_kept=10, image_count=len(visual_blocks)
                )
                visual_cap_applied = cap_note_global is not None
                blocks = salvaged_blocks + table_blocks + capped_visuals
                salvage_applied = True
                salvage_source = native_salvage_source
                chosen_text_source = "native_salvage"
                chosen_reason = best_reason
                if cap_note_global:
                    near_empty_reason = (near_empty_reason or "") + ";" + cap_note_global

    elif page_mode == "native_text_slide":
        chosen_text_source = "native"
        chosen_reason = "native_text_sufficient"

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
            # ── Enhanced debug fields ──
            "slide_page_mode": page_mode,
            "native_text_char_count": native_text_chars,
            "native_word_count": native_word_count,
            "native_text_block_count": title_text_count,
            "native_image_block_count": image_count,
            "ocr_attempted": ocr_attempted,
            "ocr_engine": ocr_engine,
            "ocr_box_count": ocr_box_count,
            "ocr_block_count": ocr_block_count,
            "chosen_text_source": chosen_text_source,
            "chosen_reason": chosen_reason,
            "ppt_export_native_attempted": True if needs_salvage else False,
            "ppt_export_native_success": ppt_recovery.get("success", False) if needs_salvage else False,
            "ppt_export_native_source": ppt_recovery.get("source") if needs_salvage else None,
            "ppt_export_native_recovered_text_len": ppt_recovered_len if needs_salvage else 0,
            "ppt_export_native_block_count": len(ppt_blocks) if needs_salvage else 0,
            "ocr_skipped_due_to_native_recovery": ocr_skipped_due_to_native_recovery if needs_salvage else False,
            "ocr_skip_reason": ocr_skip_reason if needs_salvage else None,
            # ── OCR preprocessing debug ──
            "ocr_preprocess_variant": ocr_preprocess_variant if ocr_attempted else None,
            "ocr_variant_candidates": ocr_variant_candidates if ocr_attempted else [],
            "ocr_variant_scores": ocr_variant_scores if ocr_attempted else {},
            "ocr_selected_reason": ocr_selected_reason if ocr_attempted else None,
            "ocr_confidence_avg": ocr_confidence_avg if ocr_attempted else None,
            "ocr_box_count_after_cleanup": ocr_box_count_after_cleanup if ocr_attempted else 0,
        },
    }


# ── existing helpers (unchanged) ────────────────────────────────────

def _extract_slide_text_blocks(raw: dict, page_idx: int, pw: float, ph: float) -> list[dict]:
    """
    Reassemble slide-native text from low-level line fragments.

    PPT-export PDFs often expose one visual word per PyMuPDF block, and some
    invisible text streams have off-page bboxes. This keeps visible positioned
    text for panel reconstruction and preserves off-page text only as an
    explicit low-confidence stream when it adds otherwise missing context.
    """
    visible_items: list[dict] = []
    offpage_items: list[dict] = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not text:
                continue

            bbox = [float(v) for v in line.get("bbox", [0, 0, 0, 0])]
            if len(bbox) != 4:
                continue
            is_offpage = _is_offpage_text_item(bbox, ph)
            if not is_offpage and is_visual_noise_text(text) and len(text) < 12:
                continue
            max_font = max((float(span.get("size", 0.0) or 0.0) for span in spans), default=0.0)
            item = {
                "text": text,
                "bbox": bbox,
                "font_size": max_font,
                "source": "pymupdf_slide_native",
            }

            if is_offpage:
                offpage_items.append(item)
            else:
                visible_items.append(item)

    blocks = _reassemble_visible_slide_items(visible_items, page_idx, pw, ph)
    offpage_blocks = _build_offpage_text_blocks(offpage_items, blocks, page_idx, pw, ph, len(blocks))
    blocks.extend(offpage_blocks)
    blocks.sort(key=lambda item: (
        item.get("meta", {}).get("slide_reading_order", 9999),
        item["bbox"][1],
        item["bbox"][0],
    ))
    return blocks


def _is_offpage_text_item(bbox: list[float], ph: float) -> bool:
    return bbox[3] < ph * 0.025 or bbox[1] < -ph * 0.02


def _compact_text_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _text_digit_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    return len(re.findall(r"\d", compact)) / max(1, len(compact))


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_intersection(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _inner_overlap_ratio(a: list[float], b: list[float]) -> float:
    return _bbox_intersection(a, b) / max(1.0, min(_bbox_area(a), _bbox_area(b)))


def _horizontal_overlap_ratio(a: list[float], b: list[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / max(1.0, min(a[2] - a[0], b[2] - b[0]))


def _median(values: list[float], default: float = 0.0) -> float:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return default
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _cluster_items_into_line_groups(items: list[dict], ph: float) -> list[list[dict]]:
    if not items:
        return []
    heights = [max(1.0, item["bbox"][3] - item["bbox"][1]) for item in items]
    y_tol = max(3.0, min(9.0, _median(heights, 8.0) * 0.55))
    groups: list[list[dict]] = []
    current = [items[0]]
    current_cy = (items[0]["bbox"][1] + items[0]["bbox"][3]) / 2.0

    for item in items[1:]:
        cy = (item["bbox"][1] + item["bbox"][3]) / 2.0
        if abs(cy - current_cy) <= y_tol:
            current.append(item)
            current_cy = sum((x["bbox"][1] + x["bbox"][3]) / 2.0 for x in current) / len(current)
        else:
            groups.append(current)
            current = [item]
            current_cy = cy
    groups.append(current)
    return groups


def _dedupe_line_items(items: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for item in sorted(items, key=lambda x: (_bbox_area(x["bbox"]), x["bbox"][0])):
        text = item["text"].strip()
        if not text:
            continue
        duplicate = False
        overlapping = []
        for existing in kept:
            overlap = _inner_overlap_ratio(item["bbox"], existing["bbox"])
            if overlap >= 0.86:
                a = re.sub(r"\s+", "", text)
                b = re.sub(r"\s+", "", existing["text"])
                if a in b or b in a:
                    duplicate = True
                    break
            if _horizontal_overlap_ratio(item["bbox"], existing["bbox"]) >= 0.35:
                overlapping.append(existing)
        if duplicate:
            continue
        if len(overlapping) >= 2:
            overlap_text_len = sum(_compact_text_len(x["text"]) for x in overlapping)
            if _compact_text_len(text) <= overlap_text_len * 1.2:
                continue
        kept.append(item)
    return sorted(kept, key=lambda x: x["bbox"][0])


def _line_group_to_records(group: list[dict], pw: float) -> list[dict]:
    deduped = _dedupe_line_items(group)
    if not deduped:
        return []
    font = _median([item.get("font_size", 0.0) for item in deduped], 9.0)
    split_gap = max(pw * 0.075, font * 8.0, 44.0)
    sublines: list[list[dict]] = []
    current = [deduped[0]]
    for item in deduped[1:]:
        gap = item["bbox"][0] - current[-1]["bbox"][2]
        if gap > split_gap:
            sublines.append(current)
            current = [item]
        else:
            current.append(item)
    sublines.append(current)

    records: list[dict] = []
    for subline in sublines:
        parts: list[str] = []
        prev = None
        for item in subline:
            if prev is not None:
                gap = item["bbox"][0] - prev["bbox"][2]
                if gap > max(1.2, min(item.get("font_size", 9.0), prev.get("font_size", 9.0)) * 0.18):
                    parts.append(" ")
            parts.append(item["text"].strip())
            prev = item
        text = "".join(parts).strip()
        if not text:
            continue
        bbox = [
            min(item["bbox"][0] for item in subline),
            min(item["bbox"][1] for item in subline),
            max(item["bbox"][2] for item in subline),
            max(item["bbox"][3] for item in subline),
        ]
        max_font = max(item.get("font_size", 0.0) for item in subline)
        records.append({"text": text, "bbox": bbox, "font_size": max_font, "item_count": len(subline)})
    return records


def _reassemble_visible_slide_items(items: list[dict], page_idx: int, pw: float, ph: float) -> list[dict]:
    if not items:
        return []
    items = sorted(items, key=lambda item: (((item["bbox"][1] + item["bbox"][3]) / 2.0), item["bbox"][0]))
    line_records: list[dict] = []
    for group in _cluster_items_into_line_groups(items, ph):
        line_records.extend(_line_group_to_records(group, pw))
    if not line_records:
        return []

    line_records.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    blocks: list[dict] = []
    body_clusters: list[list[dict]] = []

    for rec in line_records:
        role = _infer_slide_role(rec["text"], rec["bbox"], pw, ph, rec["font_size"])
        if role == "decorative_noise":
            continue
        rec["slide_role"] = role
        if role in ("title", "footer"):
            blocks.append(_finalize_slide_text_cluster([rec], page_idx, len(blocks)))
            continue

        placed = False
        for cluster in reversed(body_clusters):
            last = cluster[-1]
            y_gap = rec["bbox"][1] - last["bbox"][3]
            x_overlap = _horizontal_overlap_ratio(rec["bbox"], last["bbox"])
            x_drift = abs(rec["bbox"][0] - last["bbox"][0])
            center_drift = abs(((rec["bbox"][0] + rec["bbox"][2]) / 2.0) - ((last["bbox"][0] + last["bbox"][2]) / 2.0))
            same_region = x_overlap >= 0.12 or x_drift <= pw * 0.08 or center_drift <= pw * 0.20
            max_gap = max(18.0, max(rec["font_size"], last["font_size"], 8.0) * 2.4)
            if -6.0 <= y_gap <= max_gap and same_region:
                cluster.append(rec)
                placed = True
                break
            if y_gap > ph * 0.14:
                break
        if not placed:
            body_clusters.append([rec])

    for cluster in body_clusters:
        blocks.append(_finalize_slide_text_cluster(cluster, page_idx, len(blocks)))

    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for order, block in enumerate(blocks):
        block.setdefault("meta", {})["slide_reading_order"] = order
    return blocks


def _finalize_slide_text_cluster(cluster: list[dict], page_idx: int, bid: int) -> dict:
    text = "\n".join(item["text"] for item in cluster).strip()
    bbox = [
        min(item["bbox"][0] for item in cluster),
        min(item["bbox"][1] for item in cluster),
        max(item["bbox"][2] for item in cluster),
        max(item["bbox"][3] for item in cluster),
    ]
    max_font = max(item.get("font_size", 0.0) for item in cluster)
    roles = [item.get("slide_role", "body_note") for item in cluster]
    role = roles[0] if len(set(roles)) == 1 else "body_note"
    if "kpi_metric" in roles:
        role = "kpi_group" if len(cluster) > 1 else "kpi_metric"
    block_type = "title" if role == "title" else "footer" if role == "footer" else "text"
    return {
        "id": f"p{page_idx+1}_s{bid}",
        "type": block_type,
        "bbox": [round(v, 2) for v in bbox],
        "text": text,
        "source": "pymupdf_slide_native_reflow",
        "confidence": 1.0,
        "meta": {
            "slide_role": role,
            "font_size": round(max_font, 2),
            "slide_line_count": len(cluster),
            "slide_reflow_item_count": sum(item.get("item_count", 1) for item in cluster),
            "summary_priority": "high" if role in ("title", "kpi_metric", "kpi_group", "bullet_item") else "medium",
            "summary_exclude": role in ("footer", "decorative_noise"),
        },
    }


def _build_offpage_text_blocks(
    items: list[dict],
    visible_blocks: list[dict],
    page_idx: int,
    pw: float,
    ph: float,
    start_id: int,
) -> list[dict]:
    if not items:
        return []
    items = sorted(items, key=lambda item: (((item["bbox"][1] + item["bbox"][3]) / 2.0), item["bbox"][0]))
    line_texts: list[str] = []
    for group in _cluster_items_into_line_groups(items, ph):
        for rec in _line_group_to_records(group, pw):
            line_texts.append(rec["text"])
    text = "\n".join(line_texts).strip()
    if _compact_text_len(text) < 12:
        return []

    visible_text_len = sum(_compact_text_len(block.get("text", "")) for block in visible_blocks)
    has_visible_title = any(block.get("type") == "title" for block in visible_blocks)
    should_keep = visible_text_len < 80 or _compact_text_len(text) > visible_text_len * 1.25 or not has_visible_title
    if not should_keep:
        return []

    digit_ratio = _text_digit_ratio(text)
    sentence_like = bool(re.search(r"[.!?。]", text)) and _compact_text_len(text) > 55
    role = "offpage_text_stream"
    block_type = "text"
    if not has_visible_title and _compact_text_len(text) <= 140 and digit_ratio < 0.22 and not sentence_like:
        role = "title"
        block_type = "title"

    return [{
        "id": f"p{page_idx+1}_s{start_id}",
        "type": block_type,
        "bbox": [0.0, 0.0, round(pw, 2), round(min(ph * 0.12, 64.0), 2)],
        "text": text,
        "source": "pymupdf_slide_offpage_text_stream",
        "confidence": 0.72,
        "meta": {
            "slide_role": role,
            "layout_reliability": "offpage_bbox",
            "summary_priority": "high" if block_type == "title" else "medium",
            "summary_exclude": False,
            "slide_reading_order": 0,
        },
    }]


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

        chart_like_aspect = 0.5 <= aspect <= 3.0 and area_ratio >= 0.005
        wide_panel_chart_like = 0.35 <= aspect <= 4.5 and area_ratio >= 0.03 and fragment_count >= 2
        visual_type = "chart" if chart_like_aspect or wide_panel_chart_like else "image"

        result_blocks.append({
            "id": f"p{page_idx+1}_img{len(result_blocks)}",
            "type": visual_type,
            "bbox": [round(v, 2) for v in bbox],
            "text": " " if wide_panel_chart_like else "",
            "source": "pymupdf_slide_image",
            "confidence": 1.0,
            "meta": {
                "slide_role": "visual_panel",
                "summary_priority": "low",
                "visual_fragment_count": fragment_count,
                "visual_area_ratio": round(area_ratio, 5),
                "visual_reclass_guard": wide_panel_chart_like,
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
        text = "\n".join(row for row in flattened if row).strip()
        empty_grid = not text
        tables.append({
            "id": f"p{page_idx+1}_t{i}",
            "type": "table",
            "bbox": list(tbl.get("bbox", [0, 0, pw, ph])),
            "text": text,
            "source": "pdfplumber_slide",
            "confidence": 1.0,
            "meta": {
                "slide_role": "table_panel",
                "normalized_table": cells,
                "table_empty_grid": empty_grid,
                "summary_priority": "high",
            },
        })
    return tables


def _infer_slide_role(text: str, bbox: list[float], pw: float, ph: float, font_size: float) -> str:
    clean = text.strip()
    y0, y1 = bbox[1], bbox[3]
    width = bbox[2] - bbox[0]
    compact_len = _compact_text_len(clean)
    sentence_like = bool(re.search(r"[.!?。]", clean)) and compact_len > 55

    if y0 < ph * 0.16 and sentence_like and compact_len > 120:
        return "body_note"
    if y0 < ph * 0.16 and compact_len > 180 and _text_digit_ratio(clean) >= 0.12:
        return "body_note"
    if y0 < ph * 0.16 and (font_size >= 14 or (len(clean) <= 80 and width > pw * 0.18)):
        return "title"
    if y1 > ph * 0.92 and len(clean) <= 40:
        return "footer"
    if re.search(r'[\d,]+(?:\.\d+)?\s*(%|원|억원|조원|USD|배|x|pt|bps)', clean, re.I) and len(clean) <= 48:
        return "kpi_metric"
    if clean.startswith(("▪", "•", "-", "□", "◦", "*", "+")):
        return "bullet_item"
    if len(clean) <= 8 and not re.search(r'[A-Za-z가-힣]{2,}', clean):
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
    if role in ("kpi_metric", "kpi_group"):
        meta["summary_priority"] = "high"
        meta["preserve_atomic"] = True
        signals.append("kpi_detected")
    if role == "offpage_text_stream":
        meta["summary_role"] = "body_text"
        meta["summary_priority"] = "medium"
        signals.append("offpage_text_stream_preserved")
    if role == "bullet_item":
        signals.append("bullet_list_detected")
    if role == "footer" or (block["type"] in ("text", "title") and block["bbox"][3] > ph * 0.94):
        block["type"] = "footer"
        meta["summary_exclude"] = True
        meta["summary_priority"] = "low"
        signals.append("footer_detected")


def _bbox_gap(a: list[float], b: list[float]) -> tuple[float, float]:
    h_gap = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    v_gap = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return h_gap, v_gap


def _bbox_union(rects: list[list[float]]) -> list[float]:
    return [
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    ]


def _expand_bbox(bbox: list[float], x_pad: float, y_pad: float, pw: float, ph: float) -> list[float]:
    return [
        max(0.0, bbox[0] - x_pad),
        max(0.0, bbox[1] - y_pad),
        min(pw, bbox[2] + x_pad),
        min(ph, bbox[3] + y_pad),
    ]


def _block_center(block: dict) -> tuple[float, float]:
    bbox = block.get("bbox") or [0, 0, 0, 0]
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _is_context_text_block(block: dict) -> bool:
    if block.get("type") not in ("text", "title"):
        return False
    meta = block.get("meta", {}) or {}
    if meta.get("summary_exclude") or meta.get("slide_role") == "footer":
        return False
    return bool(str(block.get("text", "") or "").strip())


def _has_unreliable_slide_layout(block: dict) -> bool:
    meta = block.get("meta", {}) or {}
    return meta.get("layout_reliability") == "offpage_bbox" or meta.get("slide_role") == "offpage_text_stream"


def _is_page_wide_text_stream(block: dict, pw: float, ph: float) -> bool:
    if not _is_context_text_block(block):
        return False
    bbox = block.get("bbox") or [0, 0, 0, 0]
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    text_len = _compact_text_len(block.get("text", ""))
    top_band = bbox[1] <= ph * 0.20 and bbox[3] <= ph * 0.36
    return text_len > 140 and width >= pw * 0.65 and top_band and height <= ph * 0.24


def _slide_page_profile(blocks: list[dict], pw: float, ph: float) -> dict[str, bool | int | float]:
    text_blocks = [block for block in blocks if _is_context_text_block(block)]
    reliable_local = [
        block for block in text_blocks
        if not _has_unreliable_slide_layout(block) and not _is_page_wide_text_stream(block, pw, ph)
    ]
    global_streams = [
        block for block in text_blocks
        if _has_unreliable_slide_layout(block) or _is_page_wide_text_stream(block, pw, ph)
    ]
    global_text = "\n".join(str(block.get("text", "") or "") for block in global_streams)
    local_len = sum(_compact_text_len(block.get("text", "")) for block in reliable_local)
    global_len = _compact_text_len(global_text)
    visual_count = sum(1 for block in blocks if block.get("type") in ("chart", "image"))
    punctuation_count = len(re.findall(r"[.!?。]", global_text))
    global_digit_ratio = _text_digit_ratio(global_text)
    global_text_only = global_len >= 40 and local_len < 40
    legal_like = global_text_only and punctuation_count >= 1 and global_len > 55 and global_digit_ratio < 0.18 and visual_count <= 4
    sparse_index_like = global_text_only and global_len <= 260 and global_digit_ratio >= 0.08 and visual_count <= 4
    return {
        "global_text_only": global_text_only,
        "legal_like": legal_like,
        "sparse_index_like": sparse_index_like,
        "local_text_len": local_len,
        "global_text_len": global_len,
        "visual_count": visual_count,
    }


def _slide_region_id(block: dict) -> str:
    meta = block.get("meta", {}) or {}
    return str(meta.get("slide_region_id") or meta.get("structural_region_id") or "")


def _blocks_share_slide_region(a: dict, b: dict) -> bool:
    aid = _slide_region_id(a)
    bid = _slide_region_id(b)
    return bool(aid and bid and aid == bid)


def _infer_slide_text_subrole(block: dict, pw: float, ph: float, page_profile: dict) -> str:
    meta = block.get("meta", {}) or {}
    text = str(block.get("text", "") or "").strip()
    bbox = block.get("bbox") or [0, 0, 0, 0]
    compact_len = _compact_text_len(text)
    role = meta.get("slide_role", "")
    if role == "offpage_text_stream" or _is_page_wide_text_stream(block, pw, ph):
        if page_profile.get("legal_like"):
            return "disclaimer"
        if page_profile.get("sparse_index_like"):
            return "contents_stream"
        return "page_text_stream"
    if block.get("type") == "title" or role == "title":
        return "title"
    if block.get("type") == "footer" or role == "footer" or bbox[3] > ph * 0.92:
        return "footer"
    if role in ("kpi_metric", "kpi_group") or (_metric_tokens(text) and compact_len <= 110):
        return "metric"
    if role == "bullet_item" or text.startswith(("▪", "•", "-", "□", "◦", "*", "+")):
        return "bullet"
    if compact_len <= 28 and not re.search(r"[.!?。]", text):
        return "label"
    if bbox[1] < ph * 0.24 and compact_len <= 120:
        return "subtitle"
    return "body"


def _slide_panel_match_score(anchor: dict, candidate: dict, pw: float, ph: float) -> float | None:
    ab = anchor.get("bbox") or [0, 0, 0, 0]
    cb = candidate.get("bbox") or [0, 0, 0, 0]
    expanded = _expand_bbox(ab, pw * 0.035, ph * 0.065, pw, ph)
    ccx, ccy = _block_center(candidate)
    inside = expanded[0] <= ccx <= expanded[2] and expanded[1] <= ccy <= expanded[3]
    overlap = _bbox_intersection(cb, expanded) / max(1.0, _bbox_area(cb))
    h_gap, v_gap = _bbox_gap(cb, ab)
    x_overlap = _horizontal_overlap_ratio(cb, ab)
    same_band = ab[1] - ph * 0.06 <= ccy <= ab[3] + ph * 0.06 and h_gap <= pw * 0.06
    caption_above = cb[3] <= ab[1] + 8.0 and ab[1] - cb[3] <= ph * 0.10 and x_overlap >= 0.12
    caption_below = cb[1] >= ab[3] - 8.0 and cb[1] - ab[3] <= ph * 0.09 and x_overlap >= 0.12
    if not inside and overlap < 0.18 and not same_band and not caption_above and not caption_below:
        return None
    score = overlap * 100.0
    if inside:
        score += 35.0
    if caption_above or caption_below:
        score += 25.0
    if same_band:
        score += 12.0
    score -= (h_gap / max(1.0, pw)) * 20.0
    score -= (v_gap / max(1.0, ph)) * 24.0
    return score


def _apply_slide_region_metadata(region_id: str, members: list[dict], order: int) -> None:
    region_bbox = _bbox_union([member.get("bbox") or [0, 0, 0, 0] for member in members])
    for member in members:
        meta = member.setdefault("meta", {})
        meta["slide_region_id"] = region_id
        meta["slide_region_order"] = order
        meta["slide_region_bbox"] = [round(v, 2) for v in region_bbox]
        meta["slide_region_member_count"] = len(members)


def _prepare_slide_structure_regions(blocks: list[dict], pw: float, ph: float, quality_notes: list) -> None:
    page_profile = _slide_page_profile(blocks, pw, ph)
    for block in blocks:
        meta = block.setdefault("meta", {})
        subrole = _infer_slide_text_subrole(block, pw, ph, page_profile) if block.get("type") in ("text", "title", "footer") else ""
        if subrole:
            meta["slide_subrole"] = subrole
        if subrole in ("disclaimer", "footer"):
            meta["summary_priority"] = "low"
            meta["summary_exclude"] = True
        elif subrole in ("metric", "title"):
            meta["summary_priority"] = "high"
        elif subrole in ("label", "caption", "subtitle", "contents_stream", "page_text_stream"):
            meta["summary_priority"] = meta.get("summary_priority", "medium")

    anchors = [
        block for block in blocks
        if block.get("type") in ("table", "chart", "image")
        and block.get("meta", {}).get("summary_exclude_reason") != "empty_table_geometry"
    ]
    anchors.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    local_text = [
        block for block in blocks
        if _is_context_text_block(block)
        and not _has_unreliable_slide_layout(block)
        and not _is_page_wide_text_stream(block, pw, ph)
    ]

    region_members: dict[str, list[dict]] = {}
    text_assignments: dict[str, tuple[float, str]] = {}
    for idx, anchor in enumerate(anchors):
        ab = anchor.get("bbox") or [0, 0, 0, 0]
        area_ratio = _bbox_area(ab) / max(1.0, pw * ph)
        meta = anchor.setdefault("meta", {})
        if anchor.get("type") in ("chart", "image") and (
            area_ratio < 0.012 or (ab[3] <= ph * 0.16 and area_ratio < 0.03)
        ):
            meta["slide_visual_role"] = "decorative_candidate"
            meta["summary_priority"] = "low"
        region_id = f"slide_region_{idx + 1}"
        region_members[region_id] = [anchor]
        meta["slide_region_id"] = region_id
        meta["slide_region_order"] = idx + 1
        for candidate in local_text:
            score = _slide_panel_match_score(anchor, candidate, pw, ph)
            if score is None:
                continue
            cid = str(candidate.get("id") or id(candidate))
            if score > text_assignments.get(cid, (-9999.0, ""))[0]:
                text_assignments[cid] = (score, region_id)

    id_to_text = {str(block.get("id") or id(block)): block for block in local_text}
    for cid, (_, region_id) in text_assignments.items():
        candidate = id_to_text.get(cid)
        if candidate is not None:
            region_members.setdefault(region_id, []).append(candidate)

    for order, region_id in enumerate(sorted(region_members, key=lambda rid: min(b["bbox"][1] for b in region_members[rid])), start=1):
        _apply_slide_region_metadata(region_id, region_members[region_id], order)

    clustered = 0
    text_regions: list[list[dict]] = []
    for block in sorted(local_text, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        if _slide_region_id(block):
            continue
        placed = False
        for region in reversed(text_regions):
            last = region[-1]
            h_gap, v_gap = _bbox_gap(last["bbox"], block["bbox"])
            x_overlap = _horizontal_overlap_ratio(last["bbox"], block["bbox"])
            center_gap = abs(_block_center(last)[0] - _block_center(block)[0])
            same_column = x_overlap >= 0.18 or center_gap <= pw * 0.16 or h_gap <= pw * 0.045
            if same_column and v_gap <= max(ph * 0.055, 26.0):
                region.append(block)
                placed = True
                break
        if not placed:
            text_regions.append([block])

    base_order = len(region_members) + 1
    for idx, region in enumerate(text_regions, start=base_order):
        _apply_slide_region_metadata(f"slide_text_region_{idx}", region, idx)
        clustered += len(region)

    if anchors or clustered:
        quality_notes.append(f"slide_structure_regions_prepared:{len(region_members)}:{clustered}")


def _nearby_slide_text_blocks(
    anchor: dict,
    text_blocks: list[dict],
    pw: float,
    ph: float,
    limit: int = 6,
    allow_unreliable_layout: bool = True,
    allow_page_wide: bool = True,
    prefer_same_region: bool = False,
    require_same_region: bool = False,
) -> list[dict]:
    bbox = anchor.get("bbox") or [0, 0, 0, 0]
    expanded = _expand_bbox(bbox, pw * 0.035, ph * 0.055, pw, ph)
    candidates: list[tuple[float, dict]] = []
    for text_block in text_blocks:
        if not _is_context_text_block(text_block):
            continue
        if not allow_unreliable_layout and _has_unreliable_slide_layout(text_block):
            continue
        if not allow_page_wide and _is_page_wide_text_stream(text_block, pw, ph):
            continue
        same_region_candidate = _blocks_share_slide_region(anchor, text_block)
        if require_same_region and not same_region_candidate:
            continue
        tb = text_block.get("bbox") or [0, 0, 0, 0]
        if tb == bbox:
            continue
        overlap = _bbox_intersection(tb, expanded) / max(1.0, _bbox_area(tb))
        h_gap, v_gap = _bbox_gap(tb, bbox)
        x_overlap = _horizontal_overlap_ratio(tb, bbox)
        tcy = (tb[1] + tb[3]) / 2.0
        same_band = bbox[1] - ph * 0.055 <= tcy <= bbox[3] + ph * 0.055 and h_gap <= pw * 0.055
        caption_above = tb[3] <= bbox[1] + 8.0 and bbox[1] - tb[3] <= ph * 0.09 and x_overlap >= 0.12
        caption_below = tb[1] >= bbox[3] - 8.0 and tb[1] - bbox[3] <= ph * 0.08 and x_overlap >= 0.12
        if overlap < 0.20 and not same_band and not caption_above and not caption_below:
            continue
        dist = h_gap + v_gap + abs(tcy - ((bbox[1] + bbox[3]) / 2.0)) * 0.25
        if same_region_candidate:
            dist -= ph * 0.08
        elif prefer_same_region and _slide_region_id(anchor) and _slide_region_id(text_block):
            dist += ph * 0.18
        if text_block.get("type") == "title":
            dist += ph * 0.04
        candidates.append((dist, text_block))
    candidates.sort(key=lambda item: (item[0], item[1]["bbox"][1], item[1]["bbox"][0]))
    return [item[1] for item in candidates[:limit]]


def _nearest_slide_title(
    anchor: dict,
    text_blocks: list[dict],
    ph: float,
    pw: float | None = None,
    allow_unreliable_layout: bool = True,
    allow_page_wide: bool = True,
) -> str:
    bbox = anchor.get("bbox") or [0, 0, 0, 0]
    titles = []
    for block in text_blocks:
        if block.get("type") != "title":
            continue
        if not allow_unreliable_layout and _has_unreliable_slide_layout(block):
            continue
        if pw is not None and not allow_page_wide and _is_page_wide_text_stream(block, pw, ph):
            continue
        text = str(block.get("text", "") or "").strip()
        if not text:
            continue
        tb = block.get("bbox") or [0, 0, 0, 0]
        vertical_dist = abs(bbox[1] - tb[3]) if tb[3] <= bbox[1] + ph * 0.12 else ph
        titles.append((vertical_dist, tb[1], text))
    if not titles:
        return ""
    titles.sort(key=lambda item: (item[0], item[1]))
    return titles[0][2]


def _flatten_context_lines(blocks: list[dict]) -> list[str]:
    lines: list[str] = []
    for block in sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        for line in str(block.get("text", "") or "").splitlines():
            clean = re.sub(r"\s+", " ", line).strip()
            if clean:
                lines.append(clean)
    return lines


def _metric_tokens(text: str) -> list[str]:
    pattern = r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|bp|bps|원|억원|조원|조|USD|pt|x)?"
    return [m.strip() for m in re.findall(pattern, text or "", flags=re.I) if m.strip()]


def _key_value_rows_from_lines(lines: list[str]) -> list[list[str]]:
    rows = [["item", "values"]]
    for line in lines:
        nums = _metric_tokens(line)
        if not nums:
            continue
        label = line
        for token in nums:
            label = label.replace(token, " ")
        label = re.sub(r"^[\s\-\*\+\u2022\uf0a7\u25aa\u25cf\u25cb]+", "", label)
        label = re.sub(r"\s+", " ", label).strip(" :;,")
        if not label:
            label = "metric"
        rows.append([label, ", ".join(nums)])
    return rows if len(rows) > 1 else []


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [
        ["" if cell is None else str(cell).strip() for cell in row] + [""] * (width - len(row))
        for row in rows
    ]
    header = padded[0]
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in padded[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _table_shape(rows: list[list[str]]) -> dict[str, int]:
    return {"rows": len(rows), "cols": max((len(row) for row in rows), default=0)}


def _enrich_slide_table_context(blocks: list[dict], pw: float, ph: float, quality_notes: list) -> int:
    text_blocks = [block for block in blocks if _is_context_text_block(block)]
    enriched = 0
    for block in blocks:
        if block.get("type") != "table":
            continue
        meta = block.setdefault("meta", {})
        cells = meta.get("normalized_table")
        cell_text = ""
        has_cells = False
        if isinstance(cells, list):
            cell_text = " ".join(
                str(cell).strip()
                for row in cells if isinstance(row, list)
                for cell in row if cell is not None and str(cell).strip()
            )
            has_cells = bool(re.search(r"[A-Za-z0-9\uac00-\ud7a3]", cell_text))
        text = str(block.get("text", "") or "").strip()
        context_blocks = []
        expanded = _expand_bbox(block["bbox"], pw * 0.02, ph * 0.025, pw, ph)
        nearby_candidates = _nearby_slide_text_blocks(
            block,
            text_blocks,
            pw,
            ph,
            limit=10,
            allow_unreliable_layout=False,
            allow_page_wide=False,
            prefer_same_region=True,
            require_same_region=True,
        )
        if not nearby_candidates and not _slide_region_id(block):
            nearby_candidates = _nearby_slide_text_blocks(
                block,
                text_blocks,
                pw,
                ph,
                limit=10,
                allow_unreliable_layout=False,
                allow_page_wide=False,
                prefer_same_region=True,
            )
        for candidate in nearby_candidates:
            cb = candidate.get("bbox") or [0, 0, 0, 0]
            ccx, ccy = _block_center(candidate)
            inside = expanded[0] <= ccx <= expanded[2] and expanded[1] <= ccy <= expanded[3]
            overlap = _bbox_intersection(cb, expanded) / max(1.0, _bbox_area(cb))
            if inside or overlap >= 0.18:
                context_blocks.append(candidate)
        context_lines = _flatten_context_lines(context_blocks)
        title = next((str(candidate.get("text", "") or "").strip() for candidate in context_blocks if candidate.get("type") == "title"), "")
        if not title and not _slide_region_id(block):
            title = _nearest_slide_title(
                block,
                text_blocks,
                ph,
                pw=pw,
                allow_unreliable_layout=False,
                allow_page_wide=False,
            )
        if title:
            meta["associated_title"] = title
        if context_blocks:
            meta["context_block_ids"] = [candidate["id"] for candidate in context_blocks]

        if has_cells and text:
            meta["table_summary"] = text
            meta["table_markdown"] = meta.get("table_markdown") or _rows_to_markdown(cells)
            enriched += 1
            continue

        kv_rows = _key_value_rows_from_lines(context_lines)
        if kv_rows:
            summary_lines = [f"{row[0]}: {row[1]}" for row in kv_rows[1:]]
            summary = "\n".join(summary_lines)
            block["text"] = summary
            meta["table_summary"] = summary
            meta["key_value_rows"] = [{"item": row[0], "values": row[1]} for row in kv_rows[1:]]
            meta["normalized_table"] = {
                "headers": kv_rows[0],
                "rows": kv_rows,
                "shape": _table_shape(kv_rows),
                "source": "slide_text_context",
                "markdown": _rows_to_markdown(kv_rows),
            }
            meta["table_markdown"] = meta["normalized_table"]["markdown"]
            meta["table_reconstruction_method"] = "nearby_text_key_value"
            meta["summary_priority"] = "high"
            enriched += 1
        elif not has_cells and (not text or not re.search(r"[A-Za-z0-9\uac00-\ud7a3]", text)):
            block["type"] = "unknown"
            block["text"] = ""
            meta["summary_exclude"] = True
            meta["summary_priority"] = "low"
            meta["summary_exclude_reason"] = "empty_table_geometry"
            meta["classification_reason"] = "empty_pdfplumber_grid_without_text_context"
    if enriched:
        quality_notes.append(f"slide_table_context_enriched:{enriched}")
    return enriched


def _enrich_slide_visual_context(blocks: list[dict], pw: float, ph: float, quality_notes: list) -> int:
    text_blocks = [block for block in blocks if _is_context_text_block(block)]
    page_profile = _slide_page_profile(blocks, pw, ph)
    enriched = 0
    for block in blocks:
        if block.get("type") not in ("chart", "image"):
            continue
        meta = block.setdefault("meta", {})
        if meta.get("visual_reclass_guard"):
            block["text"] = ""
        bbox = block.get("bbox") or [0, 0, 0, 0]
        area_ratio = _bbox_area(bbox) / max(1.0, pw * ph)
        context_blocks = _nearby_slide_text_blocks(
            block,
            text_blocks,
            pw,
            ph,
            limit=6,
            allow_unreliable_layout=False,
            allow_page_wide=False,
            prefer_same_region=True,
            require_same_region=True,
        )
        if not context_blocks and not _slide_region_id(block):
            context_blocks = _nearby_slide_text_blocks(
                block,
                text_blocks,
                pw,
                ph,
                limit=6,
                allow_unreliable_layout=False,
                allow_page_wide=False,
                prefer_same_region=True,
        )
        context_lines = _flatten_context_lines(context_blocks)
        title = next((str(candidate.get("text", "") or "").strip() for candidate in context_blocks if candidate.get("type") == "title"), "")
        if not title and not _slide_region_id(block):
            title = _nearest_slide_title(
                block,
                text_blocks,
                ph,
                pw=pw,
                allow_unreliable_layout=False,
                allow_page_wide=False,
            )
        if (
            not context_lines
            and not title
            and area_ratio >= 0.02
            and not page_profile.get("global_text_only")
            and not _slide_region_id(block)
        ):
            nearest = []
            bx, by = _block_center(block)
            for text_block in text_blocks:
                if _has_unreliable_slide_layout(text_block) or _is_page_wide_text_stream(text_block, pw, ph):
                    continue
                tx, ty = _block_center(text_block)
                nearest.append((abs(by - ty) + abs(bx - tx) * 0.35, text_block))
            nearest.sort(key=lambda item: item[0])
            if nearest:
                context_blocks = [item[1] for item in nearest[:1]]
                context_lines = _flatten_context_lines(context_blocks)
        metrics = []
        for line in context_lines:
            metrics.extend(_metric_tokens(line))
        metrics = metrics[:12]

        if not context_lines and not title:
            if meta.get("slide_visual_role") == "decorative_candidate" or page_profile.get("global_text_only") or area_ratio < 0.035:
                meta["summary_exclude"] = True
                meta["summary_priority"] = "low"
                meta["summary_exclude_reason"] = "decorative_or_unanchored_visual_without_local_context"
                meta["slide_panel_kind"] = "decorative_visual"
                continue
            meta["summary_priority"] = "low"
            meta["summary_exclude_reason"] = "visual_without_local_text_context"
            meta["slide_panel_kind"] = "unanchored_visual"
            continue

        if context_lines or title or metrics:
            visual_kind = "chart" if block.get("type") == "chart" or len(metrics) >= 2 else "image"
            block["type"] = visual_kind
            summary_parts = []
            if title:
                summary_parts.append(f"title: {title}")
            if context_lines:
                summary_parts.append("context: " + " / ".join(context_lines[:4]))
            if metrics:
                summary_parts.append("visible_metrics: " + ", ".join(metrics))
            summary = "\n".join(summary_parts).strip()
            meta["chart_summary" if visual_kind == "chart" else "visual_summary"] = summary
            meta["caption_text"] = summary
            meta["context_block_ids"] = [b["id"] for b in context_blocks]
            meta["associated_title"] = title
            meta["visible_key_metrics"] = metrics
            meta["summary_priority"] = "high" if visual_kind == "chart" and metrics else "medium"
            enriched += 1
    if enriched:
        quality_notes.append(f"slide_visual_context_enriched:{enriched}")
    return enriched


def _assign_slide_panel_order(blocks: list[dict], pw: float, ph: float) -> None:
    anchors = [
        block for block in blocks
        if block.get("type") in ("table", "chart", "image")
        and not block.get("meta", {}).get("summary_exclude")
    ]
    text_blocks = [block for block in blocks if _is_context_text_block(block)]

    anchor_groups: dict[str, list[dict]] = {}
    for idx, anchor in enumerate(anchors):
        region_id = _slide_region_id(anchor) or f"anchor_region_{idx + 1}"
        anchor_groups.setdefault(region_id, []).append(anchor)

    def _group_key(item: tuple[str, list[dict]]) -> tuple[float, float]:
        members = item[1]
        return (
            min(member.get("meta", {}).get("slide_region_order", 500) for member in members),
            min(member["bbox"][1] for member in members),
        )

    ordered_anchor_groups = sorted(anchor_groups.items(), key=_group_key)
    for idx, (region_id, group) in enumerate(ordered_anchor_groups):
        panel_id = f"slide_panel_{idx + 1}"
        panel_order = idx + 1
        panel_bbox = _bbox_union([member.get("bbox") or [0, 0, 0, 0] for member in group])
        attached_text: dict[str, dict] = {}

        for anchor in group:
            meta = anchor.setdefault("meta", {})
            meta["slide_panel_id"] = panel_id
            meta["slide_panel_order"] = panel_order
            meta["slide_panel_bbox"] = [round(v, 2) for v in panel_bbox]
            meta["slide_panel_source_region"] = region_id
            meta["slide_panel_kind"] = meta.get(
                "slide_panel_kind",
                "table_panel" if anchor.get("type") == "table" else "visual_panel",
            )
            local_context = _nearby_slide_text_blocks(
                anchor,
                text_blocks,
                pw,
                ph,
                limit=8,
                allow_unreliable_layout=False,
                allow_page_wide=False,
                prefer_same_region=True,
                require_same_region=True,
            )
            if not local_context:
                local_context = _nearby_slide_text_blocks(
                    anchor,
                    text_blocks,
                    pw,
                    ph,
                    limit=8,
                    allow_unreliable_layout=False,
                    allow_page_wide=False,
                    prefer_same_region=True,
                )
            for text_block in local_context:
                attached_text[str(text_block.get("id") or id(text_block))] = text_block

        if attached_text:
            panel_bbox = _bbox_union([panel_bbox] + [block.get("bbox") or [0, 0, 0, 0] for block in attached_text.values()])
            for anchor in group:
                anchor.setdefault("meta", {})["slide_panel_bbox"] = [round(v, 2) for v in panel_bbox]
        for text_block in attached_text.values():
            tmeta = text_block.setdefault("meta", {})
            if "slide_panel_id" not in tmeta:
                tmeta["slide_panel_id"] = panel_id
                tmeta["slide_panel_order"] = panel_order
                tmeta["slide_panel_bbox"] = [round(v, 2) for v in panel_bbox]
                tmeta["slide_panel_source_region"] = region_id
                tmeta["slide_panel_kind"] = "panel_text"

    fallback_order = len(ordered_anchor_groups) + 1
    existing_regions: dict[str, list[dict]] = {}
    for block in text_blocks:
        if block.get("meta", {}).get("slide_panel_order") is not None:
            continue
        region_id = _slide_region_id(block)
        if region_id and not _has_unreliable_slide_layout(block) and not _is_page_wide_text_stream(block, pw, ph):
            existing_regions.setdefault(region_id, []).append(block)

    for region_id, region in sorted(
        existing_regions.items(),
        key=lambda item: (min(b.get("meta", {}).get("slide_region_order", 500) for b in item[1]), min(b["bbox"][1] for b in item[1])),
    ):
        panel_id = f"slide_text_region_{fallback_order}"
        panel_bbox = _bbox_union([block.get("bbox") or [0, 0, 0, 0] for block in region])
        for block in region:
            meta = block.setdefault("meta", {})
            meta["slide_panel_id"] = panel_id
            meta["slide_panel_order"] = fallback_order
            meta["slide_panel_bbox"] = [round(v, 2) for v in panel_bbox]
            meta["slide_panel_source_region"] = region_id
            meta["slide_panel_kind"] = "text_region"
        fallback_order += 1

    text_regions: list[list[dict]] = []
    for block in sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        meta = block.setdefault("meta", {})
        if "slide_panel_order" in meta or block.get("type") != "text":
            continue
        if _has_unreliable_slide_layout(block) or _is_page_wide_text_stream(block, pw, ph):
            meta["slide_panel_id"] = f"slide_text_stream_{fallback_order}"
            meta["slide_panel_order"] = fallback_order
            meta["slide_panel_kind"] = "page_text_stream"
            fallback_order += 1
            continue

        placed = False
        for region in reversed(text_regions):
            last = region[-1]
            if _has_unreliable_slide_layout(last) or _is_page_wide_text_stream(last, pw, ph):
                continue
            h_gap, v_gap = _bbox_gap(last["bbox"], block["bbox"])
            x_overlap = _horizontal_overlap_ratio(last["bbox"], block["bbox"])
            center_gap = abs(_block_center(last)[0] - _block_center(block)[0])
            same_region = x_overlap >= 0.18 or center_gap <= pw * 0.16 or h_gap <= pw * 0.045
            if same_region and v_gap <= max(ph * 0.055, 26.0):
                region.append(block)
                placed = True
                break
        if not placed:
            text_regions.append([block])

    for region in text_regions:
        panel_id = f"slide_text_region_{fallback_order}"
        for block in region:
            meta = block.setdefault("meta", {})
            meta["slide_panel_id"] = panel_id
            meta["slide_panel_order"] = fallback_order
            meta["slide_panel_kind"] = "text_region"
        fallback_order += 1

    for block in blocks:
        meta = block.setdefault("meta", {})
        if block.get("type") == "title":
            meta["slide_panel_id"] = meta.get("slide_panel_id", "slide_title")
            meta["slide_panel_order"] = 0
            meta["slide_panel_kind"] = meta.get("slide_panel_kind", "slide_title")
        elif block.get("type") == "footer":
            meta["slide_panel_id"] = meta.get("slide_panel_id", "slide_footer")
            meta["slide_panel_order"] = 999
            meta["slide_panel_kind"] = meta.get("slide_panel_kind", "slide_footer")
        elif "slide_panel_order" not in meta:
            fallback_kind = "decorative_visual" if block.get("type") in ("chart", "image") else "unassigned_region"
            meta["slide_panel_id"] = f"slide_text_{fallback_order}"
            meta["slide_panel_order"] = fallback_order
            meta["slide_panel_kind"] = meta.get("slide_panel_kind", fallback_kind)
            fallback_order += 1
        meta["slide_reading_order"] = meta["slide_panel_order"] * 1000 + int(block["bbox"][1] * 10) + int(block["bbox"][0])


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

    _prepare_slide_structure_regions(out, pw, ph, quality_notes)
    _enrich_slide_table_context(out, pw, ph, quality_notes)
    _enrich_slide_visual_context(out, pw, ph, quality_notes)
    before_empty_drop = len(out)
    out = [
        block for block in out
        if not (
            block.get("type") == "unknown"
            and block.get("meta", {}).get("summary_exclude_reason") == "empty_table_geometry"
            and not str(block.get("text", "") or "").strip()
        )
    ]
    dropped_empty = before_empty_drop - len(out)
    if dropped_empty:
        quality_notes.append(f"slide_empty_table_geometry_dropped:{dropped_empty}")
    _assign_slide_panel_order(out, pw, ph)
    out.sort(key=lambda item: (
        item.get("meta", {}).get("slide_panel_order", 500),
        item["bbox"][1],
        item["bbox"][0],
    ))
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
