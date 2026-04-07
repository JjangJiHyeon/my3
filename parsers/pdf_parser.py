"""
PDF parser – block-level extraction pipeline.

Extraction order
────────────────
1. PyMuPDF  ``get_text("dict")``  → text / title / footer blocks with bbox
2. pdfplumber table regions       → table blocks with bbox + cell data
3. Camelot fallback               → table blocks (when pdfplumber finds none)
4. Image / chart-like heuristics  → image & chart_like blocks via xref
5. OCR boxes (only when needed)   → text blocks from OCR for text-poor regions

Preview images are rendered via PyMuPDF pixmap and saved as PNG under
``<RESULT_DIR>/previews/<doc_id>/``.

Every failure is caught and recorded in ``parser_debug.parse_warnings``
so that partial results are always returned.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import uuid
from typing import Any

import fitz  # PyMuPDF
import numpy as np

logger = logging.getLogger(__name__)

# ── tunables ─────────────────────────────────────────────────────────

OCR_TEXT_THRESHOLD = 60       # chars per page below which OCR is considered
OCR_IMAGE_MIN = 1             # page must have ≥1 image to trigger OCR
PREVIEW_ZOOM = 1.5            # zoom factor for preview PNGs
PREVIEW_FORMAT = "png"
BLOCK_ZOOM = 2.0              # zoom for OCR rendering (higher → better quality)

CHART_MIN_ASPECT = 0.3        # aspect-ratio lower bound for chart heuristic
CHART_MAX_ASPECT = 3.0        # aspect-ratio upper bound
CHART_MIN_AREA_RATIO = 0.02   # image area / page area lower bound

# ── Camelot filters ──────────────────────────────────────────────────
CAM_TABLE_MIN_ROWS = 2
CAM_TABLE_MIN_COLS = 2
CAM_TABLE_MIN_FILLED_RATIO = 0.35
CAM_TABLE_MAX_PAGE_AREA_RATIO = 0.70
CAM_TABLE_MIN_ACCURACY = 60.0

# ── result dir (will be set from app.py via module-level helper) ─────

_result_dir: str = ""


def set_result_dir(path: str) -> None:
    global _result_dir
    _result_dir = path


def _preview_dir(doc_id: str) -> str:
    d = os.path.join(_result_dir, "previews", doc_id)
    os.makedirs(d, exist_ok=True)
    return d


# ── public entry point ──────────────────────────────────────────────

def parse_pdf(filepath: str) -> dict[str, Any]:
    import hashlib
    doc_id = hashlib.md5(filepath.encode("utf-8")).hexdigest()

    doc = fitz.open(filepath)
    
    # ── 1. Early Routing (Phase 1) ──────────────────────────
    from .document_router import pre_route_document, route_document
    initial_doc_type = pre_route_document(doc)
    
    # ── 2. Pipeline Dispatch (Phase 2) ──────────────────────
    # Import specialized pipelines
    from .pdf.slide_pipeline import process_slide_ir
    from .pdf.report_pipeline import process_text_report
    from .pdf.dashboard_pipeline import process_dashboard_brief
    from .pdf.utils import extract_metadata, extract_tables_via_pdfplumber
    
    # Common shared data
    metadata = extract_metadata(doc)
    plumber_tables = extract_tables_via_pdfplumber(filepath, doc.page_count)
    
    pages: list[dict[str, Any]] = []
    ocr_page_nums: list[int] = []
    
    # Decide which pipeline to use
    pipeline_fn = process_text_report
    pipeline_name = "text_report_pipeline"
    fallback_reason = None
    
    if initial_doc_type == "slide_ir":
        pipeline_fn = process_slide_ir
        pipeline_name = "slide_ir_pipeline"
    elif initial_doc_type == "dashboard_brief":
        pipeline_fn = process_dashboard_brief
        pipeline_name = "dashboard_brief_pipeline"
    elif initial_doc_type == "table_heavy":
        # Temporary fallback until table_heavy_pipeline is implemented
        pipeline_fn = process_text_report
        pipeline_name = "text_report_pipeline"
        fallback_reason = "table_heavy_pipeline_not_implemented"
        
    # Process all pages using the chosen pipeline
    for i in range(doc.page_count):
        try:
            page_result = pipeline_fn(doc, i, plumber_tables.get(i, []))
            
            # Common Normalization & Preview Generation
            final_page = _finalize_page_results(doc, page_result, i, doc_id)
            final_page.setdefault("parser_debug", {})["pipeline_used"] = pipeline_name
            if fallback_reason:
                final_page["parser_debug"]["pipeline_fallback_reason"] = fallback_reason
                
            if final_page.get("ocr_applied"):
                ocr_page_nums.append(i + 1)
            pages.append(final_page)
        except Exception as exc:
            logger.warning(f"Pipeline {pipeline_name} failed on page {i+1}: {exc}")
            pages.append({"page_num": i+1, "error": str(exc), "blocks": []})

    # ── 3. Final Routing Refinement ───────────────────────
    final_routing = route_document(doc, pages)
    # The user requires pipeline_used and document_type to ALWAYS match.
    # We use the pipeline_name (without '_pipeline' suffix) as the definitive document_type.
    doc_type_from_pipeline = pipeline_name.replace("_pipeline", "")
    refined_doc_type = final_routing.get("document_type", doc_type_from_pipeline)
    routing_confidence = final_routing.get("confidence", 0.0)
    page_type_distribution = final_routing.get("page_type_distribution")
    routing_reasons = final_routing.get("routing_reasons")
    
    doc.close()
    
    # Post-processing
    _apply_cross_page_header_footer_penalties(pages)
    
    empty_pages = sum(1 for p in pages if not p.get("blocks"))
    quality = _assess_quality(pages, ocr_page_nums, empty_pages)

    # NEW: Aggregate extraction source stats
    source_stats = {
        "pdf4llm_pages": sum(1 for p in pages if p.get("parser_debug", {}).get("report_text_source") == "pdf4llm"),
        "ppt_recovery_pages": sum(1 for p in pages if p.get("parser_debug", {}).get("chosen_text_source") == "ppt_export_native"),
        "ocr_skipped_pages": sum(1 for p in pages if p.get("parser_debug", {}).get("ocr_skipped_due_to_native_recovery")),
        "native_pages": sum(1 for p in pages if p.get("parser_debug", {}).get("chosen_text_source") == "native"),
    }

    mismatch = (refined_doc_type != doc_type_from_pipeline)

    metadata.update({
        "parser_used": "Modular Pipeline (" + pipeline_name + ")",
        "ocr_pages": ocr_page_nums,
        "text_quality": quality,
        "empty_pages": empty_pages,
        "document_type": doc_type_from_pipeline,
        "refined_document_type": refined_doc_type,
        "doc_type": refined_doc_type,
        "pipeline_used": pipeline_name,
        "routing_confidence": routing_confidence,
        "page_type_distribution": page_type_distribution,
        "routing_reasons": routing_reasons,
        "routing_signals": final_routing.get("routing_signals"),
        "routing_reason": final_routing.get("routing_reason"),
        "routing_mismatch_flag": mismatch,
        "extraction_source_stats": source_stats,
    })
    
    if fallback_reason:
        metadata["pipeline_fallback_reason"] = fallback_reason

    if mismatch:
        metadata["initial_routing_type"] = initial_doc_type
        metadata["refined_routing_type"] = refined_doc_type
        metadata["routing_mismatch_flag"] = True

    return {"pages": pages, "metadata": metadata, "status": "success"}


def _finalize_page_results(doc: fitz.Document, page_result: dict, page_idx: int, doc_id: str) -> dict:
    """Consolidate pipeline output into common UI/Downstream schema."""
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    
    # 1. Preview Rendering
    pdir = _preview_dir(doc_id)
    pname = f"page_{page_idx+1}.png"
    ppath = os.path.join(pdir, pname)
    
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(PREVIEW_ZOOM, PREVIEW_ZOOM))
        pix.save(ppath)
        page_result["preview_image"] = f"/api/documents/{doc_id}/pages/{page_idx+1}/preview"
        page_result["preview_width"] = pix.width
        page_result["preview_height"] = pix.height
    except Exception as e:
        logger.warning(f"Failed to render preview for page {page_idx+1}: {e}")
        page_result.setdefault("parser_debug", {})["preview_error"] = str(e)

    page_result["page_width"] = pw
    page_result["page_height"] = ph
    page_result["coord_space"] = "points"
    
    # 2. Block Validation & Normalization
    raw_blocks = page_result.get("blocks", [])
    valid_blocks = []
    dropped = []
    warnings = []
    
    for b in raw_blocks:
        vb = _validate_and_format_block(b, pw, ph, page_idx+1, dropped, warnings)
        if vb:
            valid_blocks.append(vb)

    # ── Advanced Heuristics Integrated (Normalization Phase) ──
    quality_notes = []
    merge_events = []
    classification_overrides = []
    dedup_stats = {}
    text_merge_stats = {}
    pipeline_used_early = page_result.get("parser_debug", {}).get("pipeline_used", "")
    skip_paragraph_merge = pipeline_used_early == "slide_ir_pipeline"
    
    # 2.1 Reclassify Images (based on text density)
    valid_blocks = _reclassify_images(valid_blocks, pw, ph, quality_notes, classification_overrides, dropped)
    
    # 2.2 Deduplicate (Native vs OCR overlap)
    valid_blocks = _deduplicate_blocks(valid_blocks, merge_events, dedup_stats, quality_notes)
    
    # 2.3 Merge Adjacent Text (Paragraph restoration) — skipped for slide_ir; card merge runs later
    if not skip_paragraph_merge:
        valid_blocks = _merge_adjacent_text_blocks(valid_blocks, merge_events, text_merge_stats, pw, ph)
    
    # 2.4 Layout Inference
    layout_hint = _infer_page_layout_hint(valid_blocks, pw, ph)
    pipeline_postprocess = page_result.pop("_pipeline_postprocess", None)
    if callable(pipeline_postprocess):
        valid_blocks, layout_hint = pipeline_postprocess(
            valid_blocks,
            pw,
            ph,
            layout_hint,
            merge_events,
            quality_notes,
        )
    
    # 2.5 Pipeline-specific Post-processing
    pipeline_used = page_result.get("parser_debug", {}).get("pipeline_used", "")
    if pipeline_used == "slide_ir_pipeline" or layout_hint == "slide_like":
        valid_blocks = _merge_slide_cards(valid_blocks, merge_events)
    elif pipeline_used in ("text_report_pipeline", "dashboard_brief_pipeline"):
        valid_blocks = _absorb_tiny_fragments(valid_blocks, merge_events)
    
    # 2.6 Reading Order
    valid_blocks, actual_strategy, strategy_basis = _sort_reading_order(valid_blocks, layout_hint, pw, ph)
    valid_blocks, table_debug = _enrich_table_blocks(valid_blocks, pw, ph, quality_notes)
    
    page_result["blocks"] = valid_blocks
    dbg = page_result.setdefault("parser_debug", {})
    dbg["dropped_blocks"] = dropped
    dbg["bbox_warnings"] = warnings
    dbg["page_layout_hint"] = layout_hint
    dbg["reading_order_strategy"] = actual_strategy
    dbg["reading_order_basis"] = strategy_basis
    dbg["quality_notes"] = quality_notes
    dbg["merge_events"] = merge_events
    dbg["classification_overrides"] = classification_overrides
    dbg["dedup_stats"] = dedup_stats
    dbg["text_merge_stats"] = text_merge_stats
    dbg["table_enrichment"] = table_debug
    
    # 3. OCR Fallback for text-poor pages in modular path
    consolidated_text = "\n\n".join(b["text"] for b in valid_blocks if b.get("text") and b["type"] != "footer")
    image_block_count = sum(1 for b in valid_blocks if b["type"] in ("image", "chart"))
    page_result["ocr_applied"] = False
    modular_ocr_allowed = True
    if pipeline_used == "slide_ir_pipeline":
        chosen_source = dbg.get("chosen_text_source")
        if chosen_source in ("native", "native_salvage", "ppt_export_native") and len(consolidated_text) >= 40:
            modular_ocr_allowed = False
            dbg["ocr_fallback_skipped_reason"] = "slide_structured_native_text_present"

    if modular_ocr_allowed and len(consolidated_text) < OCR_TEXT_THRESHOLD and image_block_count >= OCR_IMAGE_MIN:
        try:
            from .ocr_utils import run_ocr_on_image
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            ocr_result = run_ocr_on_image(img_arr, page_width=pw, page_height=ph, zoom=2.0)
            if ocr_result.get("success") and ocr_result.get("text", "").strip():
                ocr_text = ocr_result["text"].strip()
                if len(ocr_text) > len(consolidated_text):
                    valid_blocks.append({
                        "id": f"p{page_idx+1}_ocr0",
                        "type": "text",
                        "bbox": [0, 0, round(pw, 2), round(ph, 2)],
                        "text": ocr_text,
                        "page_num": page_idx + 1,
                        "source": "ocr_fallback",
                        "score": 0.6,
                        "meta": {"recovery_source": "ocr", "summary_priority": "medium"},
                    })
                    page_result["ocr_applied"] = True
                    consolidated_text = "\n\n".join(
                        b["text"] for b in valid_blocks if b.get("text") and b["type"] != "footer"
                    )
                    dbg["ocr_fallback_chars"] = len(ocr_text)
                    quality_notes.append("ocr_fallback_applied")
        except Exception as ocr_exc:
            dbg["ocr_fallback_error"] = str(ocr_exc)

    if page_result.get("ocr_applied"):
        valid_blocks, actual_strategy, strategy_basis = _sort_reading_order(valid_blocks, layout_hint, pw, ph)
        dbg["reading_order_strategy"] = f"{actual_strategy}+post_ocr"
        dbg["reading_order_basis"] = strategy_basis

    valid_blocks, page_rag_structure = _apply_page_level_rag_structure(valid_blocks, ph)
    page_result["blocks"] = valid_blocks
    page_result["page_title"] = page_rag_structure.get("page_title", "")
    dbg["page_rag_structure"] = page_rag_structure

    consolidated_text = "\n\n".join(
        b["text"] for b in valid_blocks
        if b.get("text") and b["type"] != "footer"
    )
    page_result["text"] = consolidated_text

    # RAG-Ready Text Stream Generation
    rag_text = _generate_rag_text(valid_blocks, page_title=page_result.get("page_title"))
    page_result["rag_text"] = rag_text
    
    # Extract tables for router
    tables = [b for b in valid_blocks if b["type"] == "table"]
    page_result["tables"] = tables
    
    # Count images/charts
    images = [b for b in valid_blocks if b["type"] in ("image", "chart")]
    page_result["image_count"] = len(images)
    
    # Debug counts
    type_counts = {}
    for b in valid_blocks:
        t = b["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    page_result["parser_debug"]["block_type_counts"] = type_counts
    
    return page_result


def _enrich_table_blocks(blocks: list[dict], pw: float, ph: float, quality_notes: list) -> tuple[list[dict], dict[str, Any]]:
    from .table_utils import detect_table_candidates, normalize_table_candidate, score_table_quality, choose_best_table_candidate

    if not blocks:
        return blocks, {"candidate_count": 0, "kept_table_count": 0, "rejected_candidate_count": 0, "reject_reason_counts": {}}

    detect_table_candidates(pw, ph, blocks)
    table_candidates = []
    reject_reason_counts: dict[str, int] = {}
    rejected_candidate_count = 0

    for block in blocks:
        meta = block.setdefault("meta", {})
        if block["type"] != "table" and meta.get("table_candidate_score", 0) <= 1.0:
            continue

        source_name = str(block.get("source", "native")).lower()
        source_type = "camelot" if "camelot" in source_name else "pdfplumber" if "pdfplumber" in source_name else "ocr" if "ocr" in source_name else "native"
        raw_data = (
            meta.get("normalized_table")
            or meta.get("rows")
            or meta.get("cells")
            or meta.get("normalized_table_rows")
            or block.get("text", "")
        )
        structured_fallback = bool(meta.get("table_summary") or meta.get("key_value_rows") or meta.get("table_markdown"))
        if isinstance(raw_data, dict) and raw_data.get("rows"):
            norm_tbl = raw_data
            norm_tbl.setdefault("source", source_type)
            norm_tbl.setdefault("headers", norm_tbl.get("rows", [[]])[0] if norm_tbl.get("rows") else [])
            norm_tbl.setdefault("shape", {
                "rows": len(norm_tbl.get("rows", [])),
                "cols": max((len(r) for r in norm_tbl.get("rows", []) if isinstance(r, list)), default=0),
            })
            norm_tbl.setdefault("markdown", meta.get("table_markdown", ""))
        else:
            norm_tbl = normalize_table_candidate(raw_data, source_type, block.get("text"))
        quality_score = score_table_quality(norm_tbl)
        if structured_fallback and norm_tbl.get("rows"):
            quality_score = max(quality_score, 0.75)
            norm_tbl.setdefault("quality", {})["structured_fallback_preserved"] = True

        reject_signals = meta.setdefault("table_reject_signals", [])
        for reason in norm_tbl.get("reject_signals", []):
            if reason not in reject_signals:
                reject_signals.append(reason)

        severe_rejects = {
            "giant_dump_table",
            "collapsed_row_table",
            "mixed_narrative_table",
            "non_rectangular_sparse_table",
            "single_column_pseudo_table",
            "symbolic_pseudo_table",
            "long_text_dump_table",
            "mostly_empty_cells",
        }
        if block["type"] == "table" and not structured_fallback and (quality_score < 0.5 or any(reason in severe_rejects for reason in reject_signals)):
            block["type"] = "unknown"
            meta["classification_reason"] = "low_quality_table_candidate"
            rejected_candidate_count += 1
            for reason in reject_signals:
                reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1
            continue

        meta["normalized_table"] = norm_tbl
        meta["table_quality"] = quality_score
        meta["table_quality_details"] = norm_tbl.get("quality", {})
        meta["table_markdown"] = norm_tbl.get("markdown", "")
        meta["table_selection_reason"] = norm_tbl.get("quality", {}).get("selection_reason") or (
            f"kept_quality={quality_score:.2f}, rejects={','.join(reject_signals) if reject_signals else 'none'}"
        )
        table_candidates.append(block)

    if not table_candidates:
        return blocks, {
            "candidate_count": rejected_candidate_count,
            "kept_table_count": 0,
            "rejected_candidate_count": rejected_candidate_count,
            "reject_reason_counts": reject_reason_counts,
        }

    dropped_ids = set()
    for table_block in table_candidates:
        if table_block["id"] in dropped_ids:
            continue

        group = [table_block]
        for other in table_candidates:
            if other["id"] == table_block["id"] or other["id"] in dropped_ids:
                continue
            if _rects_overlap(table_block["bbox"], other["bbox"], 0.5):
                group.append(other)
                dropped_ids.add(other["id"])

        if len(group) <= 1:
            continue

        norm_candidates = [candidate.get("meta", {}).get("normalized_table") for candidate in group if candidate.get("meta", {}).get("normalized_table")]
        best_norm = choose_best_table_candidate(norm_candidates)
        if not best_norm:
            continue

        for candidate in group:
            meta = candidate.setdefault("meta", {})
            if meta.get("normalized_table") != best_norm:
                candidate["type"] = "unknown"
                meta["classification_reason"] = "inferior_overlapping_table_candidate"
                dropped_ids.add(candidate["id"])
            else:
                meta["table_selection_reason"] = best_norm.get("quality", {}).get("selection_reason")

    kept_blocks = [block for block in blocks if block["id"] not in dropped_ids or block["type"] != "unknown"]
    if any(block.get("meta", {}).get("table_quality") is not None for block in kept_blocks):
        quality_notes.append("table_quality_enrichment_applied")
    kept_table_count = sum(1 for block in kept_blocks if block.get("type") == "table")
    candidate_count = len(table_candidates) + rejected_candidate_count
    return kept_blocks, {
        "candidate_count": candidate_count,
        "kept_table_count": kept_table_count,
        "rejected_candidate_count": max(0, candidate_count - kept_table_count),
        "reject_reason_counts": reject_reason_counts,
    }



def _apply_cross_page_header_footer_penalties(pages: list[dict[str, Any]]) -> None:
    from collections import defaultdict
    import re
    
    def _norm(t):
        return re.sub(r'[\d\s\.\-\|]', '', str(t).lower())

    cand_map = defaultdict(list)
    
    for p_idx, p in enumerate(pages):
        blocks = p.get("blocks", [])
        ph = p.get("page_height", 800)
        for b in blocks:
            text = b.get("text", "").strip()
            if not text or len(text) > 60:
                continue
            
            y0, y1 = b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[3]
            
            if y0 < ph * 0.15 or y1 > ph * 0.85:
                nt = _norm(text)
                if len(nt) >= 3:
                    cand_map[nt].append((p_idx, b))

    for nt, occurrences in cand_map.items():
        if len(occurrences) >= 3:
            for p_idx, b in occurrences:
                meta = b.setdefault("meta", {})
                if meta.get("summary_exclude"):
                    continue
                meta["summary_exclude"] = True
                meta["summary_exclude_reason"] = "repeated_header_footer"
                meta["summary_priority"] = "low"
                
    noise_kws = ["end of document", "본 자료는", "투자권유", "법적 책임", "법적책임", "compliance notice"]
    for p in pages:
        for b in p.get("blocks", []):
            text_lower = str(b.get("text", "")).lower()
            meta = b.setdefault("meta", {})
            if meta.get("summary_exclude"):
                continue
            if len(text_lower) < 100:
                if any(kw in text_lower for kw in noise_kws):
                     meta["summary_exclude"] = True
                     meta["summary_exclude_reason"] = "noise_keyword"
                     meta["summary_priority"] = "low"

# ── block validation/normalization ──────────────────────────────────

def _validate_and_format_block(
    blk: dict,
    page_width: float,
    page_height: float,
    page_num: int,
    dropped_blocks: list,
    bbox_warnings: list
) -> dict | None:
    """Normalize block schema and validate/clamp bbox."""
    blkid = blk.get("id", "unknown")
    source = blk.get("source", "unknown")
    raw_bbox = blk.get("bbox")
    
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        dropped_blocks.append({"id": blkid, "source": source, "reason": "invalid_bbox_length"})
        return None
    
    try:
        x0, y0, x1, y1 = float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])
    except (ValueError, TypeError):
        dropped_blocks.append({"id": blkid, "source": source, "reason": "invalid_bbox_type"})
        return None

    # fix reversed bbox
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
        
    original_coords = (x0, y0, x1, y1)

    # clamp to page bounds
    x0 = max(0.0, min(float(page_width), x0))
    x1 = max(0.0, min(float(page_width), x1))
    y0 = max(0.0, min(float(page_height), y0))
    y1 = max(0.0, min(float(page_height), y1))

    if (x0, y0, x1, y1) != original_coords:
        reason = "negative_bbox_clamped" if any(v < 0 for v in original_coords) else "page_bounds_clamped"
        bbox_warnings.append(f"Block {blkid} ({source}): {reason} from {original_coords} to {(x0, y0, x1, y1)}")

    width = x1 - x0
    height = y1 - y0
    
    # drop if too small (< 2x2 points)
    if width < 2 or height < 2:
        dropped_blocks.append({"id": blkid, "source": source, "reason": "too_small_dropped"})
        return None
        
    btype = blk.get("type", "unknown")
    if btype == "chart_like":
        btype = "chart"
    if btype not in ("text", "table", "image", "chart", "title", "footer", "unknown"):
        btype = "unknown"

    # Support both 'meta' (new modular pipelines) and 'extra' (legacy internal)
    meta = blk.get("meta") or blk.get("extra") or {}

    return {
        "id": blkid,
        "type": btype,
        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "text": blk.get("text", ""),
        "page_num": page_num,
        "source": source,
        "score": blk.get("confidence", 0.0),
        "meta": meta
    }


# ── Phase 2 Heuristics ──────────────────────────────────────────────

import difflib

def _text_similarity(a: str, b: str) -> float:
    an = "".join(a.split()).lower()
    bn = "".join(b.split()).lower()
    if not an or not bn: return 0.0
    set_a, set_b = set(an), set(bn)
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))

def _get_source_priority(src: str) -> int:
    s = str(src).lower()
    if "pymupdf" in s: return 3
    if "pdfplumber" in s: return 3
    if "camelot" in s: return 3
    if "ocr" in s: return 1
    return 0

def _reclassify_images(blocks: list[dict], pw: float, ph: float, quality_notes: list, classification_overrides: list, dropped: list) -> list[dict]:
    out = []
    page_area = pw * ph if pw * ph > 0 else 1
    reclassified = False
    
    # Calculate contained text length for each image block
    for b in blocks:
        if b["type"] in ("image", "chart", "chart_like"):
            x0, y0, x1, y1 = b["bbox"]
            area = (x1 - x0) * (y1 - y0)
            area_ratio = area / page_area
            
            if area_ratio > 0.85:
                dropped.append({"id": b["id"], "reason": "background_dropped", "source": b["source"]})
                continue
                
            contained_text_len = len(b.get("text", ""))
            
            # Fallback: check other text blocks inside it
            if contained_text_len == 0:
                for tb in blocks:
                    if tb["type"] in ("text", "title", "footer"):
                        tx0, ty0, tx1, ty1 = tb["bbox"]
                        # if tb is mostly inside b
                        if tx0 >= x0 - 10 and ty0 >= y0 - 10 and tx1 <= x1 + 10 and ty1 <= y1 + 10:
                            contained_text_len += len(tb.get("text", ""))
                            
            text_density = contained_text_len / (area + 1)
            
            if contained_text_len > 50 and text_density > 0.0005:
                b["type"] = "text"
                b["meta"]["classification_reason"] = "high_text_density"
                classification_overrides.append({
                    "block_id": b["id"],
                    "from": "image",
                    "to": "text",
                    "reason": "high_text_density"
                })
                reclassified = True
            else:
                b["meta"]["classification_reason"] = "area_ratio_check"
                
        out.append(b)
        
    if reclassified:
        quality_notes.append("image_reclassified_to_text")
        
    return out

def _deduplicate_blocks(blocks: list[dict], merge_events: list, dedup_stats: dict, quality_notes: list) -> list[dict]:
    out = []
    dropped_count = 0
    sorted_blocks = sorted(blocks, key=lambda x: _get_source_priority(x["source"]), reverse=True)
    
    for b in sorted_blocks:
        x0, y0, x1, y1 = b["bbox"]
        b_area = (x1-x0)*(y1-y0)
        if b_area <= 0: continue
        
        is_duplicate = False
        for ex in out:
            if b["type"] not in ("text", "title", "footer", "unknown") or ex["type"] not in ("text", "title", "footer", "unknown"):
                continue
            
            ex0, ey0, ex1, ey1 = ex["bbox"]
            ix0, iy0 = max(x0, ex0), max(y0, ey0)
            ix1, iy1 = min(x1, ex1), min(y1, ey1)
            iarea = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            
            overlap_ratio = iarea / min(b_area, max(1, (ex1-ex0)*(ey1-ey0)))
            
            if overlap_ratio >= 0.5:
                sim = _text_similarity(b["text"], ex["text"])
                if sim >= 0.75 or (len(b["text"]) < 10 and overlap_ratio >= 0.75):
                    is_duplicate = True
                    merge_events.append({
                        "kept": ex["id"],
                        "dropped": b["id"],
                        "reason": "duplicate_text_native_preferred" if _get_source_priority(ex["source"]) > _get_source_priority(b["source"]) else "duplicate_text_merged"
                    })
                    dropped_count += 1
                    break
                    
        if not is_duplicate:
            out.append(b)
            
    dedup_stats["dropped_duplicates"] = dropped_count
    if dropped_count > 0:
        quality_notes.append("duplicate_text_removed")
        
    return sorted(out, key=lambda x: int(x["id"].split("_b")[-1]) if "_b" in x["id"] else 0)

def _merge_adjacent_text_blocks(blocks: list[dict], merge_events: list, text_merge_stats: dict, pw: float, ph: float) -> list[dict]:
    # Group text blocks into paragraph-like clumps
    out = []
    merged_count = 0
    
    # Sort blocks natively by vertical Y then X
    # Using center points helps
    sorted_b = sorted(blocks, key=lambda b: (b["bbox"][1] + (b["bbox"][3]-b["bbox"][1])/2, b["bbox"][0]))
    
    merged = []
    skip = set()
    
    for i in range(len(sorted_b)):
        if i in skip: continue
        b1 = sorted_b[i]
        
        # We only try to merge text-like blocks generated natively or OCR (not tables/images)
        if b1["type"] != "text":
            merged.append(b1)
            continue
            
        current = b1.copy()
        
        for j in range(i+1, len(sorted_b)):
            if j in skip: continue
            b2 = sorted_b[j]
            
            if b2["type"] != "text":
                continue
                
            y_gap = b2["bbox"][1] - current["bbox"][3]
            x_diff = abs(b2["bbox"][0] - current["bbox"][0])
            width1 = current["bbox"][2] - current["bbox"][0]
            width2 = b2["bbox"][2] - b2["bbox"][0]
            w_diff_ratio = abs(width1 - width2) / max(1, width1)
            
            import re
            def get_num_ratio(t):
                if not getattr(get_num_ratio, 'compiled', False):
                    get_num_ratio.pat = re.compile(r'[\d%\.,\-]')
                    get_num_ratio.compiled = True
                nums = len(get_num_ratio.pat.findall(t))
                return nums / max(1, len(t))
                
            num_ratio_1 = get_num_ratio(current["text"])
            num_ratio_2 = get_num_ratio(b2["text"])
            
            is_numeric_dump = (num_ratio_1 > 0.4 and len(current["text"]) < 50) or (num_ratio_2 > 0.4 and len(b2["text"]) < 50)
            
            # --- DS Issue 1: Header Meta & KPI vs Narrative Barrier ---
            text1, text2 = current["text"], b2["text"]
            
            # 1. Header/Meta Separation (30% area for safety)
            is_header_1 = current["bbox"][1] < ph * 0.3
            is_header_2 = b2["bbox"][1] < ph * 0.3
            
            meta_kws = ["연구원", "Analyst", "Tel", "com", "@", "팀장", "리서치", "연락처", "이메일", "작성자"]
            has_meta_1 = any(k in text1 for k in meta_kws) or re.search(r'20[2-3]\d\.\s?\d{1,2}\.', text1)
            has_meta_2 = any(k in text2 for k in meta_kws) or re.search(r'20[2-3]\d\.\s?\d{1,2}\.', text2)
            
            is_meta_1 = is_header_1 and has_meta_1 and len(text1.strip()) < 100
            is_meta_2 = is_header_2 and has_meta_2 and len(text2.strip()) < 100
            
            # 2. KPI / Dashboard numbers vs Narrative Isolation (강화: len < 80, num > 0.25)
            is_kpi_1 = (num_ratio_1 > 0.25) and (len(text1) < 80) and any(c in text1 for c in "%+-.,")
            is_kpi_2 = (num_ratio_2 > 0.25) and (len(text2) < 80) and any(c in text2 for c in "%+-.,")
            
            is_narrative_1 = len(text1) > 80 and num_ratio_1 < 0.25
            is_narrative_2 = len(text2) > 80 and num_ratio_2 < 0.25
            
            # 3. Section-like line barrier: 짧은 독립 헤더성 라인은 merge 차단
            _sec_kws = ["이슈", "이벤트", "동향", "전략", "요약", "현황", "실적", "종합",
                        "주간", "일간", "테마", "업종", "섹터", "상승", "하락", "시황",
                        "[한국]", "[미국]", "[원전]", "[게임]", "[바이오", "[통신"]
            def _is_section_like(t):
                t_strip = t.strip()
                if len(t_strip) > 50 or len(t_strip) < 2:
                    return False
                # 불렛 포인트로 시작하는 짧은 줄도 독립 유지 (•, -, □ 등)
                if t_strip[0] in "•-*+◦□\u25e6\u2022\u25cb":
                    return True
                lines = t_strip.split('\n')
                if len(lines) > 2:
                    return False
                return any(kw in t_strip for kw in _sec_kws)
            
            is_sec_1 = _is_section_like(text1)
            is_sec_2 = _is_section_like(text2)
            
            skip_merge = False
            
            # 메타 정보와 일반 본문은 합치지 않음
            if (is_meta_1 != is_meta_2):
                skip_merge = True
                
            # KPI 조각과 일반 내러티브 본문은 합치지 않음
            if (is_kpi_1 and is_narrative_2) or (is_kpi_2 and is_narrative_1):
                skip_merge = True
            
            # KPI 조각끼리도 서로 다른 카테고리면 합치지 않음 (수치 블록 독립 유지)
            if is_kpi_1 and is_kpi_2 and y_gap > 5:
                skip_merge = True
            
            # section-like line은 다른 블록과 merge 차단
            if is_sec_1 or is_sec_2:
                skip_merge = True
                
            if skip_merge:
                continue
            
            # Adjust thresholds based on numeric density
            max_y_gap = 10 if is_numeric_dump else 25
            max_x_diff = 10 if is_numeric_dump else 30
            
            # Rules: 
            # - Must be close vertically (-10 to 20 pts)
            # - Must be roughly same left margin (< 30 pts diff)
            # - Width difference shouldn't be huge (> 50% diff)
            # - Neither should be tiny footer-like or title isolated (we only checked 'text' type anyway, title/footer are separate)
            if -15 <= y_gap <= max_y_gap and x_diff < max_x_diff and w_diff_ratio < 0.6:
                # Merge current and b2
                cx0 = min(current["bbox"][0], b2["bbox"][0])
                cy0 = min(current["bbox"][1], b2["bbox"][1])
                cx1 = max(current["bbox"][2], b2["bbox"][2])
                cy1 = max(current["bbox"][3], b2["bbox"][3])
                
                current["bbox"] = [round(cx0, 2), round(cy0, 2), round(cx1, 2), round(cy1, 2)]
                current["text"] = current["text"].strip() + "\n" + b2["text"].strip()
                current["score"] = min(current.get("score", 1.0), b2.get("score", 1.0))
                
                merge_events.append({"kept": current["id"], "dropped": b2["id"], "reason": "paragraph_merge"})
                merged_count += 1
                skip.add(j)
            else:
                # If it's way below, we break search as they are sorted by Y.
                if y_gap > 100:
                    break
                    
        merged.append(current)
        
    text_merge_stats["merged_pairs"] = merged_count
    return merged

def _merge_slide_cards(blocks: list[dict], merge_events: list) -> list[dict]:
    """Slide-specific aggressive merging for titles+bullets and KPI boxes."""
    if not blocks: return []
    out = []
    skip = set()
    
    for i, b1 in enumerate(blocks):
        if i in skip: continue
        if b1["type"] not in ("text", "title"):
            out.append(b1)
            continue
            
        current = b1.copy()
        
        for j in range(i + 1, len(blocks)):
            if j in skip: continue
            b2 = blocks[j]
            if b2["type"] not in ("text", "title"): continue
            
            # Check proximity and category compatibility
            y_gap = b2["bbox"][1] - current["bbox"][3]
            x_diff = abs(b2["bbox"][0] - current["bbox"][0])
            
            # Horizontal alignment for bullets or labels
            is_same_card = False
            
            meta1, meta2 = current.get("meta", {}), b2.get("meta", {})
            r_1, r_2 = meta1.get("slide_role"), meta2.get("slide_role")
            p_1, p_2 = meta1.get("slide_panel_id"), meta2.get("slide_panel_id")
            if p_1 and p_2 and p_1 != p_2:
                continue
            
            # 1. Title + Bullet merging (x_diff within 40, small y gap)
            if current["type"] == "title" and r_2 == "bullet_item" and -10 <= y_gap <= 25 and x_diff < 40:
                is_same_card = True
            elif r_1 == "bullet_item" and r_2 == "bullet_item" and -10 <= y_gap <= 20 and x_diff < 30:
                is_same_card = True
                
            # 2. KPI merging (Label + Value etc)
            elif (r_1 in ("kpi_metric", "kpi_group") or r_2 in ("kpi_metric", "kpi_group")) and -10 <= y_gap <= 30 and x_diff < 50:
                is_same_card = True
                
            # 3. Same slide_role body fragments only (avoid cross-column merges)
            elif (
                r_1 == r_2
                and r_1 in ("body_note", "bullet_item", "kpi_metric", "kpi_group")
                and -5 <= y_gap <= 12
                and x_diff < 15
            ):
                is_same_card = True
                
            if is_same_card:
                cx0 = min(current["bbox"][0], b2["bbox"][0])
                cy0 = min(current["bbox"][1], b2["bbox"][1])
                cx1 = max(current["bbox"][2], b2["bbox"][2])
                cy1 = max(current["bbox"][3], b2["bbox"][3])
                current["bbox"] = [cx0, cy0, cx1, cy1]
                current["text"] = current["text"].strip() + "\n" + b2["text"].strip()
                merge_events.append({"kept": current["id"], "dropped": b2["id"], "reason": "slide_card_merge"})
                skip.add(j)
            elif y_gap > 100:
                break
                
        out.append(current)
    return out

def _absorb_tiny_fragments(blocks: list[dict], merge_events: list) -> list[dict]:
    """Absorb small singleton text chunks into nearest larger blocks to reduce fragmentation."""
    if not blocks: return []
    out = []
    
    # Identify tiny blocks (<30 chars, <2 lines) and normal blocks
    tiny_blocks = []
    normal_blocks = []
    
    for b in blocks:
        if b["type"] == "text" and len(b.get("text", "")) < 30 and "\n\n" not in b.get("text", ""):
            tiny_blocks.append(b)
        else:
            normal_blocks.append(b)
            
    if not normal_blocks:
        return blocks # If everything is tiny, just return
        
    absorbed_ids = set()
    
    for tb in tiny_blocks:
        # Avoid absorbing purely structural headers
        if len(tb["text"]) < 5 and tb["text"].strip().isdigit():
            normal_blocks.append(tb)
            continue
            
        best_candidate = None
        min_dist = float('inf')
        
        tx0, ty0, tx1, ty1 = tb["bbox"]
        tcy = (ty0 + ty1) / 2
        tcx = (tx0 + tx1) / 2
        
        for nb in normal_blocks:
            nx0, ny0, nx1, ny1 = nb["bbox"]
            ncy = (ny0 + ny1) / 2
            ncx = (nx0 + nx1) / 2
            
            # Distance metric (prioritize Y axis closeness with some X tolerance)
            dist = abs(tcy - ncy) + (abs(tcx - ncx) * 0.5)
            
            # Only absorb if reasonably close (< 60 points away)
            if dist < 60 and dist < min_dist:
                min_dist = dist
                best_candidate = nb
                
        if best_candidate and best_candidate["type"] == "text":
            best_candidate["text"] += "\n" + tb["text"].strip()
            # Expand bbox
            bx0, by0, bx1, by1 = best_candidate["bbox"]
            best_candidate["bbox"] = [min(bx0, tx0), min(by0, ty0), max(bx1, tx1), max(by1, ty1)]
            merge_events.append({"kept": best_candidate["id"], "dropped": tb["id"], "reason": "tiny_absorption"})
            absorbed_ids.add(tb["id"])
        else:
            # Leave it as is if no text block is close
            normal_blocks.append(tb)
            
    # Sort again logically using original order or approx spatial order
    normal_blocks.sort(key=lambda b: (b["bbox"][1] + (b["bbox"][3]-b["bbox"][1])/2, b["bbox"][0]))
    return normal_blocks

def _infer_page_layout_hint(blocks: list[dict], pw: float, ph: float) -> str:
    # Multi-column vs single vs slide
    text_blocks = [b for b in blocks if b["type"] in ("text", "title", "footer")]
    img_blocks = [b for b in blocks if b["type"] in ("image", "chart", "table")]
    
    if pw > ph * 1.1: 
        return "slide_like"
        
    # Check for multi-column by looking at x0 coordinates
    x0s = [b["bbox"][0] for b in text_blocks if (b["bbox"][2]-b["bbox"][0]) < pw * 0.45]
    if len(x0s) > 3:
        left_col = sum(1 for x in x0s if x < pw * 0.3)
        right_col = sum(1 for x in x0s if x > pw * 0.5)
        if left_col > 1 and right_col > 1:
            return "multi_column"
            
    if len(img_blocks) > len(text_blocks) / 2 and len(img_blocks) > 0:
        return "mixed_visual"
        
    return "single_column"

def _sort_reading_order(blocks: list[dict], hint: str, pw: float, ph: float) -> tuple[list[dict], str, str]:
    if hint == "multi_column":
        strategy = "column_aware_top_down"
        basis = "x grouped by columns, then by y"
        mid_x = pw / 2
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
            min(1, int((b["bbox"][0] + (b["bbox"][2]-b["bbox"][0])/2) / (pw/2 + 1))) if (b["bbox"][2]-b["bbox"][0]) < pw * 0.6 else 0, # column
            b["bbox"][1] # y
        ))
        return blocks_sorted, strategy, basis
        
    elif hint == "slide_like":
        has_panel_order = any("slide_panel_order" in (b.get("meta") or {}) for b in blocks)
        if has_panel_order:
            strategy = "slide_panel_aware"
            basis = "panel order from slide postprocess, then local y/x order"
            blocks_sorted = sorted(blocks, key=lambda b: (
                0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
                (b.get("meta") or {}).get("slide_panel_order", 500),
                (b.get("meta") or {}).get("slide_reading_order", 999999),
                b["bbox"][1],
                b["bbox"][0],
            ))
        else:
            strategy = "slide_title_priority"
            basis = "title first, then absolute vertical sort"
            blocks_sorted = sorted(blocks, key=lambda b: (
                0 if b["type"] == "title" else 1,
                b["bbox"][1]
            ))
        return blocks_sorted, strategy, basis
        
    # single_column / mixed
    strategy = "y_bucket_sorted"
    basis = "title first, footer last, 20px grouping, then x"
    blocks_sorted = sorted(blocks, key=lambda b: (
        0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
        b["bbox"][1] // 20, 
        b["bbox"][0]
    ))
    return blocks_sorted, strategy, basis

def _is_meaningful_table_header(header_row: list[str]) -> bool:
    import re
    if not header_row: return False
    clean_cells = [str(c).strip() for c in header_row if c is not None]
    filled = [c for c in clean_cells if c]
    if len(filled) < 1:
        return False
        
    symbols_only = 0
    meaningful = 0
    contents_keywords = ["contents", "목차", "appendix", "index"]
    for c in filled:
        cl = c.lower()
        if any(k in cl for k in contents_keywords):
            return False
            
        letters = re.findall(r'[a-zA-Z0-9가-힣]', c)
        if letters:
            meaningful += 1
        else:
            symbols_only += 1
            
    if meaningful == 0:
        return False
        
    tlen = sum(len(c) for c in filled)
    slen = sum(len(re.findall(r'[▪%\(\)\.\-\/\|]', c)) for c in filled)
    if tlen > 0 and (slen / tlen) > 0.8:
        return False
        
    return True

def _has_meaningful_rows(rows: list[list[str]]) -> bool:
    import re
    if not rows: return False
    valid_row_count = 0
    for r in rows:
        if not isinstance(r, list): continue
        clean = [str(c).strip() for c in r if c is not None and str(c).strip()]
        if not clean:
            continue
        if len(clean) == 1 and re.fullmatch(r'[\d\.,%\-]+', clean[0]):
            continue
        letters = sum(len(re.findall(r'[a-zA-Z0-9가-힣]', c)) for c in clean)
        if letters == 0:
            continue
        valid_row_count += 1
    return valid_row_count > 0

def _is_visual_noise_text(text: str) -> bool:
    import re
    if not text: return True
    clean = text.strip()
    if not clean: return True
    
    noise_patterns = ["2025년", "20252", "qoq", "%", "b+", "a", "contents", "본 자", "태메젯증권",
                      "1분기", "2분기", "3분기", "4분기", "2025", "2024", "2026", "사업별 주", "분기별", "결산"]
    lower_clean = clean.lower()
    
    # 하드 게이트 정규식: 연도 조각, 특정 불용어
    if re.fullmatch(r'^(20[2-3]\d)(년|말|분기|\.|2)?$', lower_clean):
        return True
    if re.fullmatch(r'^(본\s?자|사업별\s?주|순영업수익|국내주식|해외주식\s?수|운용손의|감사합니다|eso등|규제비율|규제비율\s?1c|일간\s?상승|주요\s?이벤|전월\s?대비)$', lower_clean):
        return True
    
    if lower_clean in noise_patterns:
        return True
    
    # Micro text (1-8 chars) with no meaningful syllables/letters ≥4 (상향)
    if len(clean) <= 8 and not re.search(r'[가-힣a-zA-Z]{4,}', clean):
        return True
        
    meaningful = re.findall(r'[a-zA-Z가-힣]', clean)
    if not meaningful or len(meaningful) < 2:
        return True
        
    # Check for garbage OCR repeating characters (e.g. "I I I " or "---")
    if len(clean) > 5 and len(set(clean.replace(" ", ""))) <= 2:
        return True
        
    # Check for too many symbols/numbers vs letters (summary용으로는 품질 미달)
    if len(meaningful) < len(clean) * 0.3 and len(clean) < 40:
        return True
    
    # 강화: ≤12자 이하에서 의미있는 글자 ≤6이면 noise
    if len(clean) <= 12 and len(meaningful) <= 6:
        return True
    
    # 강화: ≤15자 이하에서 의미있는 한글/영문 ≤3이면 noise
    if len(clean) <= 15 and len(meaningful) <= 3:
        return True
        
    tokens = clean.split()
    if len(tokens) <= 2 and len(clean) <= 20:
        if any(p in lower_clean for p in noise_patterns):
            return True
        letters = sum(len(re.findall(r'[a-zA-Z가-힣]', t)) for t in tokens)
        if letters < 5:
            return True
            
    return False

def _has_disclaimer_context(*texts) -> bool:
    keywords = ["본자료는", "본 자료는", "투자권유", "법적 책임", "법적책임", "증빙자료", "광고물로 활용", "면책"]
    for t in texts:
        if not t: continue
        t_low = str(t).lower().replace(" ", "")
        for k in keywords:
            if k.replace(" ", "") in t_low:
                return True
    return False

def _has_toc_context(*texts) -> bool:
    keywords = ["contents", "목차", "appendix", "index"]
    for t in texts:
        if not t: continue
        t_low = str(t).lower()
        if any(k in t_low for k in keywords):
            return True
    return False

def _classify_page_type(page_text: str, blocks: list[dict], pw: float, ph: float, page_num: int) -> tuple[str, list[str]]:
    import re
    signals = []
    text_lower = page_text.lower().replace(" ", "")
    
    # 1. Cover or Disclaimer
    disclaimer_kws = ["본자료는", "투자권유", "법적책임", "증빙자료", "광고물로활용", "면책", "투자판단", "endofdocument"]
    d_hits = sum(1 for k in disclaimer_kws if k in text_lower)
    
    title_blocks = [b for b in blocks if b.get("type") in ("title",)]
    text_blocks  = [b for b in blocks if b.get("type") == "text"]
    long_paras = [b for b in blocks if b.get("type") == "text" and len(str(b.get("text", ""))) > 60 and "다." in str(b.get("text", ""))]
    
    if d_hits > 0: signals.append(f"disclaimer_found({d_hits})")
    if page_num <= 2: signals.append("early_page")
    if len(long_paras) == 0: signals.append("no_long_paragraph")
    if len(title_blocks) >= 2 and len(text_blocks) <= 3: signals.append("title_heavy_layout")
    
    date_hits = len(re.findall(r'20\d{2}[.년/]?\d{1,2}[.월]?', page_text))
    if date_hits >= 2: signals.append(f"date_pattern({date_hits})")
    
    cover_score = 0
    if "disclaimer_found" in str(signals): cover_score += 3
    if "no_long_paragraph" in signals:     cover_score += 1
    if "early_page" in signals:            cover_score += 1
    if "title_heavy_layout" in signals:    cover_score += 1
    if "date_pattern" in str(signals):     cover_score += 1
    
    if cover_score >= 4 or (d_hits >= 2):
        signals.append(f"classified:cover_or_disclaimer_like_score({cover_score})")
        return "cover_or_disclaimer_like", signals

    # 2. Appendix or Table Heavy
    img_blocks = [b for b in blocks if b.get("type") in ("image", "chart", "table") or b.get("meta", {}).get("table_candidate_score", 0) > 0.5]
    if len(img_blocks) > len(text_blocks) and len(img_blocks) >= 2 and len(long_paras) == 0:
        signals.append("classified:appendix_or_table_heavy")
        return "appendix_or_table_heavy", signals
        
    # 3. Dashboard KPI Like
    kw_hits = sum(1 for kw in ["1d", "1w", "1m", "수익률", "순매수", "종목", "업종", "테마", "ytd", "qoq", "yoy"] if kw in text_lower)
    if kw_hits >= 3: signals.append(f"dashboard_keywords_found({kw_hits})")
    if len(blocks) > 15: signals.append(f"many_blocks({len(blocks)})")
    
    num_count = sum(len(re.findall(r'\d+', str(b.get("text", "")))) for b in text_blocks)
    pct_count = sum(str(b.get("text", "")).count('%') for b in text_blocks)
    if num_count > 40 and pct_count > 3: signals.append("dense_metrics")
    
    if len(img_blocks) >= 2: signals.append(f"multiple_visuals({len(img_blocks)})")
    if long_paras and len(img_blocks) > 0: signals.append("narrative_and_tables_coexist")
    
    dash_score = 0
    if "dashboard_keywords_found" in str(signals): dash_score += 2
    if "dense_metrics" in signals: dash_score += 1
    if "many_blocks" in str(signals): dash_score += 1
    if "multiple_visuals" in str(signals): dash_score += 1
    if "narrative_and_tables_coexist" in signals: dash_score += 1
    
    if dash_score >= 3 and num_count > 30:
        signals.append(f"classified:dashboard_kpi_like_score({dash_score})")
        return "dashboard_kpi_like", signals
        
    # 4. Slide Like
    if pw > ph * 1.05:
        signals.append("classified:slide_like")
        return "slide_like", signals
        
    # 5. Default
    signals.append("classified:text_heavy")
    return "text_heavy", signals

def _rag_clean_text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value if v is not None)
    text = str(value).replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if limit and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _rag_normalize_key(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _rag_is_duplicate(text: str, seen_keys: set[str], threshold: float = 0.92) -> bool:
    key = _rag_normalize_key(text)
    if not key:
        return True
    if key in seen_keys:
        return True
    for old in seen_keys:
        if len(key) >= 18 and len(old) >= 18 and (key in old or old in key):
            return True
        if len(key) >= 30 and len(old) >= 30 and difflib.SequenceMatcher(None, key, old).ratio() >= threshold:
            return True
    return False


def _rag_mark_seen(text: str, seen_keys: set[str]) -> None:
    key = _rag_normalize_key(text)
    if key:
        seen_keys.add(key)


def _rag_strip_report_prefix(text: str) -> str:
    clean = _rag_clean_text(text)
    stripped = re.sub(r"^(20\d{2}년[^\s]{0,24}?보고서)+", "", clean).strip()
    return stripped if len(stripped) >= 3 else clean


def _rag_focus_quarter_heading(text: str) -> str:
    match = re.search(r"\[[1-4]Q20\d{2}\].*", text)
    if not match or match.start() <= 0:
        return text
    prefix = text[:match.start()]
    metricish = sum(1 for ch in prefix if ch.isdigit() or ch in "%()., ")
    if metricish >= max(3, int(len(prefix) * 0.35)):
        return text[match.start():].strip()
    return text


def _rag_compact_title(text: str, limit: int = 140) -> str:
    clean = _rag_focus_quarter_heading(_rag_strip_report_prefix(text))
    if not clean or _has_disclaimer_context(clean):
        return ""
    return _rag_clean_text(clean, limit)


def _rag_compact_body_text(text: str, limit: int = 700) -> str:
    clean = _rag_focus_quarter_heading(_rag_strip_report_prefix(text))
    if not clean or _has_disclaimer_context(clean):
        return ""
    return _rag_clean_text(clean, limit)


def _rag_metric_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_tokens = re.split(r"[,;|\s]+", value)
    elif isinstance(value, (list, tuple)):
        raw_tokens = [str(v) for v in value]
    else:
        raw_tokens = [str(value)]

    out: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        t = _rag_clean_text(token).strip(" ,;")
        if not t:
            continue
        compact = t.replace(",", "")
        if re.fullmatch(r"20[0-3]\d년?", compact):
            continue
        if re.fullmatch(r"[1-4](?:q|Q|분기)?", compact) or re.fullmatch(r"(?:q|Q)[1-4]", compact):
            continue
        if compact in {"24", "25", "2024", "2025"}:
            continue
        if len(compact) >= 5 and "2025" in compact and "%" not in compact and "." not in compact:
            continue
        if not (re.search(r"\d", compact) and ("%" in compact or "." in compact or len(compact) >= 2)):
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= 12:
            break
    return out


def _rag_split_visual_summary_line(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "", _rag_clean_text(line)
    prefix, value = line.split(":", 1)
    return prefix.strip().lower().replace(" ", "_"), _rag_clean_text(value)


def _rag_context_duplicate(text: str, contexts: list[str]) -> bool:
    key = _rag_normalize_key(text)
    if not key:
        return True
    for ctx in contexts:
        ctx_key = _rag_normalize_key(ctx)
        if not ctx_key:
            continue
        if len(key) >= 8 and (key in ctx_key or ctx_key in key):
            return True
        if len(key) >= 12 and len(ctx_key) >= 12 and difflib.SequenceMatcher(None, key, ctx_key).ratio() >= 0.90:
            return True
    return False


def _rag_clean_visual_summary(block: dict, context_texts: list[str] | None = None) -> str:
    meta = block.get("meta", {}) or {}
    btype = block.get("type")
    raw = meta.get("chart_summary") or meta.get("visual_summary") or meta.get("caption_text") or ""
    raw = _rag_clean_text(raw, 1200)
    block_text = _rag_clean_text(block.get("text", ""), 400)
    context_texts = list(context_texts or [])
    assoc_title = _rag_clean_text(meta.get("associated_title", ""), 240)
    if assoc_title:
        context_texts.append(assoc_title)

    if not raw:
        return ""
    if meta.get("summary_exclude") and not block_text:
        return ""
    if _has_disclaimer_context(raw, assoc_title) or _has_toc_context(raw) and not block_text:
        return ""

    metrics = _rag_metric_tokens(meta.get("visible_key_metrics"))
    lines: list[str] = []
    seen_local: set[str] = set()
    saw_informative_line = False

    for raw_line in raw.splitlines():
        prefix, value = _rag_split_visual_summary_line(raw_line)
        if not value:
            continue
        if prefix in {"title", "context"}:
            segments = [seg.strip() for seg in re.split(r"\s*/\s*", value) if seg.strip()]
            kept_segments = [seg for seg in segments if not _rag_context_duplicate(seg, context_texts)]
            if not kept_segments:
                continue
            value = _rag_clean_text(" / ".join(kept_segments), 360)
            if not value:
                continue
            label = "related_title" if prefix == "title" else "context"
            line = f"{label}: {value}"
        elif prefix == "visible_metrics":
            line_metrics = metrics or _rag_metric_tokens(value)
            if len(line_metrics) < 2:
                continue
            line = "visible_metrics: " + ", ".join(line_metrics[:12])
        else:
            line = _rag_clean_text(raw_line, 500)
            saw_informative_line = True

        if _rag_is_duplicate(line, seen_local, threshold=0.98):
            continue
        _rag_mark_seen(line, seen_local)
        lines.append(line)

    if not lines:
        prefixed_lines = [_rag_split_visual_summary_line(line)[0] for line in raw.splitlines() if line.strip()]
        if prefixed_lines and all(p in {"title", "context", "visible_metrics"} for p in prefixed_lines):
            return ""
        if not _rag_context_duplicate(raw, context_texts):
            return _rag_clean_text(raw, 500)
        return ""

    area_ratio = meta.get("visual_area_ratio")
    try:
        area_ratio_float = float(area_ratio)
    except (TypeError, ValueError):
        area_ratio_float = 0.0

    priority = str(meta.get("summary_priority") or "").lower()
    only_generic_lines = not saw_informative_line and all(
        line.startswith(("related_title:", "context:", "visible_metrics:")) for line in lines
    )
    if not block_text and only_generic_lines:
        if priority == "low":
            return ""
        if area_ratio_float and area_ratio_float < 0.015 and len(metrics) < 3:
            return ""
        if btype == "image" and len(metrics) < 3:
            return ""

    return "\n".join(lines)


def _rag_table_rows(meta: dict) -> list[list[Any]]:
    table_data = meta.get("normalized_table") or meta.get("normalized_table_rows") or meta.get("rows") or []
    if isinstance(table_data, dict):
        return table_data.get("rows", []) or []
    return table_data if isinstance(table_data, list) else []


def _rag_format_table(block: dict) -> str:
    meta = block.get("meta", {}) or {}
    table_summary = _rag_clean_text(meta.get("table_summary"), 1000)
    table_md = _rag_clean_text(meta.get("table_markdown"), 1800)
    key_values = meta.get("key_value_rows")
    parts: list[str] = []

    if table_summary:
        parts.append(f"[TABLE SUMMARY]\n{table_summary}")
    if key_values and isinstance(key_values, list):
        kv_lines = []
        for row in key_values[:12]:
            if not isinstance(row, dict):
                continue
            item = _rag_clean_text(row.get("item"), 120)
            values = _rag_clean_text(row.get("values") or row.get("value"), 180)
            if item and values:
                kv_lines.append(f"- {item}: {values}")
        if kv_lines:
            parts.append("[TABLE KEY VALUES]\n" + "\n".join(kv_lines))
    if table_md:
        parts.append(f"[TABLE]\n{table_md}")
    if parts:
        return "\n\n".join(parts)

    rows = _rag_table_rows(meta)
    if rows:
        row_count = len(rows)
        col_count = max((len(r) for r in rows if isinstance(r, list)), default=0)
        preview_rows = []
        for row in rows[:3]:
            if isinstance(row, list):
                preview_rows.append(" | ".join(_rag_clean_text(cell, 80) for cell in row))
        detail = f"{row_count} rows, {col_count} columns"
        if preview_rows:
            detail += "\n" + "\n".join(preview_rows)
        return f"[TABLE]\n{detail}"

    text = _rag_clean_text(block.get("text", ""), 500)
    return f"[TABLE]\n{text}" if text else ""


def _rag_add_part(parts: list[str], label: str, text: str, seen_keys: set[str], context_texts: list[str]) -> bool:
    clean = _rag_clean_text(text)
    if not clean or _rag_is_duplicate(clean, seen_keys):
        return False
    _rag_mark_seen(clean, seen_keys)
    context_texts.append(clean)
    parts.append(f"[{label}]\n{clean}")
    return True


def _rag_priority_rank(block: dict) -> int:
    meta = block.get("meta", {}) or {}
    priority = str(meta.get("summary_priority") or "").lower()
    if priority == "high":
        return 3
    if priority == "medium":
        return 2
    if priority == "low":
        return 0
    return 1


def _rag_panel_order(block: dict) -> int:
    meta = block.get("meta", {}) or {}
    for key in ("slide_panel_order", "panel_order", "reading_order", "slide_reading_order"):
        value = meta.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 500


def _rag_bbox_xy(block: dict) -> tuple[float, float]:
    bbox = block.get("bbox") or [0, 0, 0, 0]
    if len(bbox) < 2:
        return 0.0, 0.0
    return float(bbox[1] or 0), float(bbox[0] or 0)


def _rag_has_structured_payload(block: dict) -> bool:
    meta = block.get("meta", {}) or {}
    if block.get("type") == "table":
        return True
    return bool(
        meta.get("table_summary")
        or meta.get("key_value_rows")
        or meta.get("table_markdown")
        or meta.get("chart_summary")
        or meta.get("visual_summary")
        or meta.get("caption_text")
    )


def _rag_meaningful_char_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9가-힣]", text or ""))


def _rag_is_weak_text_block(block: dict, text: str) -> bool:
    meta = block.get("meta", {}) or {}
    if not text:
        return True
    if block.get("type") == "footer" or meta.get("summary_exclude_reason") in {
        "footer",
        "disclaimer",
        "date_metadata",
        "contact_metadata",
        "axis_label",
    }:
        return True
    if _has_disclaimer_context(text):
        return True
    if _rag_priority_rank(block) >= 3:
        return False
    meaningful = _rag_meaningful_char_count(text)
    has_metric = bool(re.search(r"\d|%|bp|p\)", text, flags=re.IGNORECASE))
    if meaningful < 3 and not has_metric:
        return True
    if len(text) <= 5 and not has_metric and _rag_priority_rank(block) < 2:
        return True
    return False


def _rag_title_candidate_score(block: dict, title: str, source: str, page_height: float) -> float:
    meta = block.get("meta", {}) or {}
    if not title or _has_disclaimer_context(title):
        return -9999.0

    btype = block.get("type")
    role = str(meta.get("summary_role") or "")
    slide_role = str(meta.get("slide_role") or "")
    y0, _ = _rag_bbox_xy(block)
    text_len = len(title)

    score = 0.0
    if source == "block_text":
        if btype == "title":
            score += 80
        if role in {"title", "section_header"}:
            score += 45
        if slide_role == "title":
            score += 45
    elif source == "associated_title":
        score += 45
    elif source == "summary_title":
        score += 35

    score += _rag_priority_rank(block) * 8
    panel_order = _rag_panel_order(block)
    score += max(0, 20 - min(panel_order, 20))
    if page_height > 0:
        if y0 <= page_height * 0.22:
            score += 12
        elif y0 >= page_height * 0.82:
            score -= 20
    if 4 <= text_len <= 160:
        score += 12
    elif text_len > 240:
        score -= 24
    if btype == "footer" or meta.get("summary_exclude"):
        score -= 80
    return score


def _rag_extract_summary_title(meta: dict) -> str:
    raw = meta.get("chart_summary") or meta.get("visual_summary") or meta.get("caption_text") or ""
    for raw_line in str(raw).splitlines():
        prefix, value = _rag_split_visual_summary_line(raw_line)
        if prefix == "title":
            return value
    return ""


def _rag_select_page_title(blocks: list[dict], page_height: float = 0) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
    for idx, block in enumerate(blocks):
        meta = block.get("meta", {}) or {}
        text = _rag_compact_title(block.get("text", ""), 180)
        btype = block.get("type")
        role = str(meta.get("summary_role") or "")
        slide_role = str(meta.get("slide_role") or "")
        priority_rank = _rag_priority_rank(block)

        if text and (
            btype == "title"
            or role in {"title", "section_header"}
            or slide_role == "title"
            or (priority_rank >= 3 and btype == "text" and _rag_panel_order(block) <= 2)
        ):
            score = _rag_title_candidate_score(block, text, "block_text", page_height)
            candidates.append((score, idx, text, {"source": "block_text", "block_id": block.get("id")}))

        assoc = _rag_compact_title(meta.get("associated_title", ""), 180)
        if assoc and _rag_has_structured_payload(block):
            score = _rag_title_candidate_score(block, assoc, "associated_title", page_height)
            candidates.append((score, idx, assoc, {"source": "associated_title", "block_id": block.get("id")}))

        summary_title = _rag_compact_title(_rag_extract_summary_title(meta), 180)
        if summary_title and _rag_has_structured_payload(block):
            score = _rag_title_candidate_score(block, summary_title, "summary_title", page_height)
            candidates.append((score, idx, summary_title, {"source": "summary_title", "block_id": block.get("id")}))

    if not candidates:
        return "", {"source": "none"}

    score, idx, title, debug = max(candidates, key=lambda item: (item[0], -item[1]))
    if score < 20:
        return "", {"source": "none", "best_score": round(score, 2)}
    debug.update({"score": round(score, 2), "candidate_index": idx})
    return title, debug


def _rag_block_summary_bucket(block: dict, selected_title_id: str | None = None) -> int:
    meta = block.get("meta", {}) or {}
    btype = block.get("type")
    if selected_title_id and block.get("id") == selected_title_id:
        return 0
    if btype == "title" or meta.get("summary_role") == "title" or meta.get("slide_role") == "title":
        return 1
    if btype == "text":
        text = _rag_compact_body_text(block.get("text", ""), 240)
        if _rag_is_weak_text_block(block, text):
            return 7
        return 2
    if btype == "table":
        return 3
    if btype in {"chart", "image"}:
        return 4
    if btype == "unknown":
        return 6
    if btype == "footer":
        return 8
    return 5


def _rag_structure_order_key(block: dict, selected_title_id: str | None = None) -> tuple:
    y0, x0 = _rag_bbox_xy(block)
    return (
        _rag_block_summary_bucket(block, selected_title_id),
        _rag_panel_order(block),
        -_rag_priority_rank(block),
        y0,
        x0,
    )


def _apply_page_level_rag_structure(
    blocks: list[dict],
    page_height: float = 0,
) -> tuple[list[dict], dict[str, Any]]:
    page_title, title_debug = _rag_select_page_title(blocks, page_height)
    selected_title_id = title_debug.get("block_id")

    ordered = sorted(blocks, key=lambda block: _rag_structure_order_key(block, selected_title_id))
    for rag_order, block in enumerate(ordered):
        meta = block.setdefault("meta", {})
        text = _rag_clean_text(block.get("text", ""))
        meta["rag_order"] = rag_order
        meta["rag_bucket"] = _rag_block_summary_bucket(block, selected_title_id)
        if page_title:
            meta["page_title"] = page_title
            if block.get("type") in {"table", "chart", "image"} and not _rag_clean_text(meta.get("associated_title", "")):
                meta["associated_title"] = page_title
        if _rag_has_structured_payload(block) and meta.get("summary_priority") != "low":
            meta["summary_priority"] = "high" if block.get("type") == "table" else meta.get("summary_priority", "medium")
        if block.get("type") in {"text", "title", "footer", "unknown"} and _rag_is_weak_text_block(block, text):
            meta["rag_exclude"] = True
            meta.setdefault("rag_exclude_reason", "weak_or_non_summary_text")
        else:
            meta.pop("rag_exclude", None)
            meta.pop("rag_exclude_reason", None)

    return ordered, {
        "page_title": page_title,
        "page_title_source": title_debug,
        "rag_order_strategy": "page_title_body_table_visual_note",
    }


def _rag_body_group_key(block: dict, page_title: str) -> str:
    meta = block.get("meta", {}) or {}
    for key in ("associated_title", "slide_panel_id", "panel_id"):
        value = _rag_clean_text(meta.get(key), 80)
        if value:
            return value
    panel_order = meta.get("slide_panel_order") or meta.get("panel_order")
    if panel_order is not None:
        return f"panel:{panel_order}"
    return page_title or "page"


def _generate_rag_text(blocks: list[dict], page_title: str | None = None) -> str:
    """Generate a clean text stream optimized for summary and RAG retrieval."""
    titles: list[str] = []
    body: list[str] = []
    tables: list[str] = []
    visuals: list[str] = []
    notes: list[str] = []
    seen_keys: set[str] = set()
    context_texts: list[str] = []
    page_title = _rag_compact_title(page_title or _rag_select_page_title(blocks)[0], 180)
    ordered_blocks = sorted(blocks, key=lambda block: (
        (block.get("meta", {}) or {}).get("rag_order", 999),
        _rag_structure_order_key(block),
    ))
    has_structured_table = any((b.get("meta", {}) or {}).get("table_summary") or (b.get("meta", {}) or {}).get("table_markdown") for b in blocks)
    pending_body_key = ""
    pending_body_lines: list[str] = []

    if page_title:
        _rag_add_part(titles, "TITLE", page_title, seen_keys, context_texts)

    def flush_body() -> None:
        nonlocal pending_body_key, pending_body_lines
        if pending_body_lines:
            _rag_add_part(body, "BODY", "\n".join(pending_body_lines), seen_keys, context_texts)
        pending_body_key = ""
        pending_body_lines = []

    for block in ordered_blocks:
        btype = block.get("type")
        text = _rag_clean_text(block.get("text", ""))
        meta = block.get("meta", {}) or {}
        priority = str(meta.get("summary_priority") or "").lower()
        slide_role = str(meta.get("slide_role") or "")

        if btype == "footer":
            footer_note = _rag_compact_body_text(text, 220)
            if footer_note and not _has_disclaimer_context(footer_note) and _rag_priority_rank(block) >= 2:
                notes.append(f"[NOTE]\nfooter: {footer_note}")
            continue
        if meta.get("summary_exclude") and btype in ("image", "chart", "unknown", "footer"):
            continue

        if btype == "title":
            title = _rag_compact_title(text)
            if title and not meta.get("summary_exclude") and not _rag_context_duplicate(title, [page_title]):
                flush_body()
                _rag_add_part(titles, "TITLE", title, seen_keys, context_texts)
            continue

        if btype == "text":
            body_text = _rag_compact_body_text(text, 700 if priority == "high" else 520)
            if (
                not body_text
                or meta.get("summary_exclude")
                or meta.get("rag_exclude")
                or priority == "low"
                or _rag_is_weak_text_block(block, body_text)
            ):
                continue
            if page_title and _rag_context_duplicate(body_text, [page_title]) and len(body_text) <= len(page_title) + 20:
                continue
            if slide_role in {"offpage_text_stream", "kpi_group"} and has_structured_table and len(body_text) > 80:
                flush_body()
                _rag_add_part(notes, "NOTE", f"page text excerpt: {body_text[:360]}", seen_keys, context_texts)
            else:
                group_key = _rag_body_group_key(block, page_title)
                if pending_body_key and group_key != pending_body_key:
                    flush_body()
                pending_body_key = group_key
                pending_body_lines.append(body_text)
            continue

        if btype == "table":
            flush_body()
            table_text = _rag_format_table(block)
            if table_text:
                clean_table = _rag_clean_text(table_text)
                if not _rag_is_duplicate(clean_table, seen_keys, threshold=0.95):
                    _rag_mark_seen(clean_table, seen_keys)
                    context_texts.append(clean_table)
                    tables.append(clean_table)
            continue

        if btype in ("image", "chart"):
            flush_body()
            visual_summary = _rag_clean_visual_summary(block, context_texts)
            if visual_summary and not _rag_is_duplicate(visual_summary, seen_keys, threshold=0.90):
                _rag_mark_seen(visual_summary, seen_keys)
                label = "CHART SUMMARY" if btype == "chart" else "VISUAL SUMMARY"
                visuals.append(f"[{label}]\n{visual_summary}")
            continue

        if btype == "unknown" and text and not meta.get("summary_exclude"):
            flush_body()
            note = _rag_clean_text(text, 360)
            if note and priority != "low" and not _has_disclaimer_context(note):
                _rag_add_part(notes, "NOTE", note, seen_keys, context_texts)

    flush_body()
    return "\n\n".join(titles + body + tables + visuals + notes).strip()

def _assign_block_scores(blocks: list[dict]):
    for b in blocks:
        if b["type"] in ("title", "footer", "text"):
            # length-based penalization for native
            slen = len(b["text"].strip())
            if slen < 3:
                sc = min(b.get("score", 1.0), 0.5)
                b["score"] = sc
                b["meta"]["score_reason"] = "too_short_text"
                b["meta"]["score_reason"] = "text_completeness_ok"
        else:
            b["meta"]["score_reason"] = f"type_heuristic_{b['type']}"

def _apply_summary_normalization(blocks: list[dict], page_height: float, source: str, page_layout_hint: str = "", summary_debug: dict = None) -> None:
    """Assigns summary roles to blocks and flags items for exclusion from the final summary."""
    if summary_debug is None: summary_debug = {}
    import re
    
    for i, b in enumerate(blocks):
        meta = b.setdefault("meta", {})
        
        src = str(b.get("source", "")).lower()
        if "ocr" in src:
            meta["recovery_source"] = "ocr"
        elif source == "hybrid":
            meta["recovery_source"] = "native+ocr"
        else:
            meta["recovery_source"] = "native"
            
        btype = b.get("type", "unknown")
        text = str(b.get("text", "")).strip()
        
        role = btype
        if btype == "title":
            role = "title"
        elif btype == "footer":
            role = "footer_like"
        elif btype == "table":
            role = "table_like"
        elif btype in ("chart", "image"):
            if btype == "chart":
                role = "chart_like"
            else:
                role = "image_like"
            meta["visual_subtype"] = btype
        elif btype == "text":
            bullet_patterns = ("▪", "•", "-", "○", "1.", "2.", "3.", "가.", "나.")
            if any(text.startswith(p) for p in bullet_patterns):
                role = "bullet_list"
            else:
                is_bold = b.get("extra", {}).get("weight", 0) >= 700
                if len(text) < 40 and not text.endswith((".", "!", "?")) and "\n" not in text and (b.get("score", 1.0) > 0.8 or is_bold):
                    role = "section_header"
                else:
                    role = "body_text"
                
        meta["summary_role"] = role
        meta["summary_exclude"] = False
        meta["summary_exclude_reason"] = None
        meta["summary_priority"] = "medium"
        
        clean_text = text.replace(" ", "").replace("\n", "")
        meaningful = re.findall(r'[a-zA-Z0-9가-힣]', clean_text)
        meaningful_ratio = len(meaningful) / len(clean_text) if len(clean_text) > 0 else 0
        
        if role in ("body_text", "bullet_list") and len(clean_text) > 0 and meaningful_ratio < 0.2:
            meta["summary_exclude"] = True
            meta["summary_exclude_reason"] = "symbol_noise"
            meta["summary_priority"] = "low"
            
        bbox = b.get("bbox", [0, 0, 0, 0])
        y0, y1 = bbox[1], bbox[3]
            
        if role in ("body_text", "footer_like") and re.fullmatch(r'[\d\.,%\-]+', clean_text) and len(meaningful) > 0:
            if y1 > page_height * 0.92:
                meta["summary_exclude"] = True
                meta["summary_exclude_reason"] = "page_number_or_footer_num"
                meta["summary_priority"] = "low"
            else:
                is_axis = False
                for j in range(max(0, i-2), min(len(blocks), i+3)):
                    if j != i and blocks[j]["type"] in ("chart", "image", "table"):
                        is_axis = True
                        break
                if is_axis:
                    meta["summary_exclude"] = True
                    meta["summary_exclude_reason"] = "axis_label"
                    meta["summary_priority"] = "low"
                else:
                    meta["summary_exclude_reason"] = "kpi_number_candidate"
                    meta["summary_priority"] = "high"
                    
        # --- DS-FIX: 날짜/연구원/연락처 블록 제외 ---
        if role in ("body_text", "section_header"):
            _txt_strip = text.strip()
            # 날짜 전용 라인 (2026.03.16 등)
            if re.fullmatch(r'\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}', _txt_strip):
                meta["summary_exclude"] = True
                meta["summary_exclude_reason"] = "date_metadata"
                meta["summary_priority"] = "low"
            # 연구원 연락처 블록 (이메일 또는 전화번호 포함, 짧은 텍스트)
            elif len(_txt_strip) < 120 and (re.search(r'[\w.+-]+@[\w.-]+\.\w+', _txt_strip) or re.search(r'\d{2,4}[-)\s]\d{3,4}[-\s]\d{3,4}', _txt_strip)):
                meta["summary_exclude"] = True
                meta["summary_exclude_reason"] = "contact_metadata"
                meta["summary_priority"] = "low"

        if role == "body_text":
            is_disclaimer_keyword = any(k in text for k in ["본 자료는", "투자자에게", "법적 책임을", "투자권유", "면책"]) or ("compliance" in text.lower())
            is_bottom = (y1 > page_height * 0.8)
            is_long_enough = (len(text) > 20)
            if is_disclaimer_keyword and (is_bottom or is_long_enough):
                meta["summary_role"] = "disclaimer"
                meta["summary_exclude"] = True
                meta["summary_exclude_reason"] = "disclaimer"
                meta["summary_priority"] = "low"
            
        if role in ("title", "section_header"):
            meta["summary_priority"] = "high"
        elif role in ("footer_like", "disclaimer"):
            meta["summary_priority"] = "low"
            
        if role in ("table_like", "chart_like", "image_like"):
            meta["visual_subtype"] = btype
            for j in range(i-1, -1, -1):
                if blocks[j]["type"] == "title":
                    meta["associated_title"] = blocks[j].get("text", "").strip()
                    break
            for j in range(i-1, -1, -1):
                t = blocks[j].get("text", "").strip()
                if blocks[j]["type"] == "text" and len(t) > 5 and not blocks[j].get("meta", {}).get("summary_exclude"):
                    meta["nearby_text_before"] = t
                    break
            for j in range(i+1, min(len(blocks), i+3)):
                t = blocks[j].get("text", "").strip()
                if blocks[j]["type"] == "text" and len(t) > 5:
                    meta["nearby_text_after"] = t
                    break

            if role in ("image_like", "chart_like"):
                ass_title = meta.get("associated_title", "")
                nbt_bef = meta.get("nearby_text_before", "")
                nbt_aft = meta.get("nearby_text_after", "")
                reject_signals = meta.get("table_reject_signals", [])
                area_ratio = b.get("extra", {}).get("area_ratio", 1.0)
                
                ex_reason = None

                if role == "chart_like":
                    meta["summary_generation_method"] = "visual_ocr"
                else:
                    meta["summary_generation_method"] = "visual_caption"
                
                txt_len = len(text)
                
                if "dashboard_like" in page_layout_hint:
                    has_date = bool(re.search(r'\b(20[2-3]\d)[./-]([0-1]?\d)\b', text))
                    if txt_len < 30 or has_date:
                        ex_reason = "dashboard_visual_noise"
                        summary_debug["dashboard_visual_noise_skips"] = summary_debug.get("dashboard_visual_noise_skips", 0) + 1
                
                if not ex_reason:
                    w = b.get("bbox", [0,0,0,0])[2] - b.get("bbox", [0,0,0,0])[0]
                    h = b.get("bbox", [0,0,0,0])[3] - b.get("bbox", [0,0,0,0])[1]
                    area_pts = w * h
                    aspect = b.get("extra", {}).get("aspect_ratio", 1.0)
                    
                    if txt_len == 0 and area_pts < 200:
                        ex_reason = "decorative_visual"
                    elif area_ratio < 0.015:
                        ex_reason = "decorative_visual_small"
                    elif area_ratio < 0.03 and len(ass_title) < 2 and txt_len == 0:
                        ex_reason = "decorative_meaningless_small"
                    elif aspect > 8 or aspect < 0.12:
                        ex_reason = "decorative_visual_line"
                
                if ex_reason:
                    meta["summary_exclude"] = True
                    meta["summary_exclude_reason"] = ex_reason
                    meta["summary_priority"] = "low"
                    
                if ass_title and _has_disclaimer_context(ass_title) and len(ass_title) > 10:
                    meta["associated_title"] = ""

            if role == "table_like":
                reject_signals = meta.get("table_reject_signals", [])
                norm_tbl = meta.get("normalized_table") or b.get("extra", {}).get("normalized_table")
                t_qual = meta.get("table_quality")
                
                ass_title = meta.get("associated_title", "").lower()
                nbt_bef = meta.get("nearby_text_before", "").lower()
                txt_lower = text.lower()
                
                has_toc_kw = any(k in ass_title or k in nbt_bef or k in txt_lower for k in ["contents", "목차", "appendix", "index"])
                
                ex_reason = None
                if isinstance(reject_signals, list) and "toc_or_appendix_keywords" in reject_signals:
                    ex_reason = "toc_table"
                elif has_toc_kw:
                    ex_reason = "toc_table"
                elif isinstance(reject_signals, list) and "paragraph_like_dump" in reject_signals:
                    ex_reason = "paragraph_dump_table"
                elif isinstance(reject_signals, list) and "small_kpi_or_logo_area" in reject_signals:
                    ex_reason = "small_kpi_table"
                elif isinstance(reject_signals, list) and "collapsed_row_table" in reject_signals:
                    ex_reason = "collapsed_row_table"
                elif isinstance(reject_signals, list) and "mixed_narrative_table" in reject_signals:
                    ex_reason = "mixed_narrative_table"
                elif isinstance(reject_signals, list) and "micro_fragment_table" in reject_signals:
                    ex_reason = "micro_fragment_table"
                elif isinstance(reject_signals, list) and "non_rectangular_sparse_table" in reject_signals:
                    ex_reason = "non_rectangular_sparse_table"
                elif isinstance(reject_signals, list) and "no_row_labels" in reject_signals:
                    ex_reason = "no_row_labels"
                elif page_layout_hint == "dashboard_like" and isinstance(reject_signals, list) and "dashboard_mixed_table" in reject_signals:
                    ex_reason = "dashboard_mixed_table"
                elif not norm_tbl:
                    ex_reason = "empty_normalized_table"
                else:
                    from .table_utils import is_summary_ready_table
                    summary_debug["summary_ready_checks"] = summary_debug.get("summary_ready_checks", 0) + 1
                    is_ready, not_ready_reason = is_summary_ready_table(norm_tbl)
                    
                    if not is_ready and not_ready_reason != "giant_dump_table":
                        ex_reason = not_ready_reason
                        summary_debug["summary_ready_failures"] = summary_debug.get("summary_ready_failures", 0) + 1
                    elif (t_qual is None or t_qual < 1.0) and not_ready_reason != "giant_dump_table":
                        ex_reason = "low_table_quality"
                
                if ex_reason:
                    meta["summary_exclude"] = True
                    meta["summary_exclude_reason"] = ex_reason
                    meta["summary_priority"] = "low"

def _generate_summary_blocks(
    blocks: list[dict],
    page: fitz.Page,
    page_width: float,
    page_height: float,
    skip_counts: dict[str, int],
    skip_events: list[dict],
    summary_debug: dict[str, Any] = None
) -> list[dict]:
    import numpy as np
    import re
    from .ocr_utils import run_ocr_on_image
    
    summary_blocks = []
    used_contexts = set()
    
    for b in blocks:
        meta = b.get("meta", {})
        role = meta.get("summary_role")
        if role not in ("table_like", "chart_like", "image_like"):
            continue
            
        ass_title = meta.get("associated_title", "")
        ctx_before = meta.get("nearby_text_before", "")
        ctx_after = meta.get("nearby_text_after", "")
        
        # --- SKIP GATES ---
        if meta.get("summary_exclude"):
            rsn = meta.get("summary_exclude_reason", "summary_exclude")
            skip_counts[rsn] = skip_counts.get(rsn, 0) + 1
            skip_events.append({"id": b["id"], "type": role, "reason": rsn})
            continue
        
        # --- FINAL HARD GATE: cover_like / dashboard_like ---
        _layout = summary_debug.get("_page_layout_hint", "")
        
        if "cover_like" in _layout and role in ("image_like", "chart_like"):
            # 커버 페이지: 이미지/차트 예외 없이 전부 차단
            skip_counts["cover_visual_noise"] = skip_counts.get("cover_visual_noise", 0) + 1
            skip_events.append({"id": b["id"], "type": role, "reason": "cover_visual_noise"})
            summary_debug["cover_visual_noise_skips"] = summary_debug.get("cover_visual_noise_skips", 0) + 1
            summary_debug["dashboard_final_gate_skips"] = summary_debug.get("dashboard_final_gate_skips", 0) + 1
            continue
        
        if "dashboard_like" in _layout and role == "table_like":
            # 대시보드 페이지: is_dashboard_summary_ready_table 통과 못하면 무조건 차단
            from .table_utils import is_dashboard_summary_ready_table
            _norm_tbl = meta.get("normalized_table") or b.get("extra", {}).get("normalized_table")
            _ready, _reason = is_dashboard_summary_ready_table(_norm_tbl) if _norm_tbl else (False, "empty_table")
            if not _ready:
                skip_counts["dashboard_not_allowlisted"] = skip_counts.get("dashboard_not_allowlisted", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": _reason})
                summary_debug["dashboard_final_gate_skips"] = summary_debug.get("dashboard_final_gate_skips", 0) + 1
                summary_debug["dashboard_allowlist_rejected"] = summary_debug.get("dashboard_allowlist_rejected", 0) + 1
                continue

        if role == "table_like":
            norm_tbl = meta.get("normalized_table") or b.get("extra", {}).get("normalized_table")
            t_qual = meta.get("table_quality", 0.0)
            txt_lower = b.get("text", "").lower()
            
            has_toc_kw = any(k in ass_title.lower() or k in ctx_before.lower() or k in txt_lower 
                             for k in ["contents", "목차", "appendix", "index"])
            
            if not norm_tbl:
                skip_counts["empty_table"] = skip_counts.get("empty_table", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "empty_normalized_table"})
                meta["summary_exclude"] = True
                continue
            
            rows = norm_tbl.get("rows", [])
            if not rows or not _has_meaningful_rows(rows):
                skip_counts["empty_table"] = skip_counts.get("empty_table", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "empty_table_rows"})
                meta["summary_exclude"] = True
                continue
                
            hdr = rows[0] if isinstance(rows[0], list) else []
            if not _is_meaningful_table_header(hdr):
                skip_counts["symbolic_header_table"] = skip_counts.get("symbolic_header_table", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "symbolic_header_table"})
                meta["summary_exclude"] = True
                continue
                
            if t_qual < 1.0:
                skip_counts["low_quality_table"] = skip_counts.get("low_quality_table", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "low_quality_table"})
                meta["summary_exclude"] = True
                continue
                
            if has_toc_kw:
                skip_counts["toc_table"] = skip_counts.get("toc_table", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "toc_table"})
                meta["summary_exclude"] = True
                continue

        # Deduplication check
        def _get_context_str(t: str) -> str:
            if not t or len(t) < 20: return t
            h = hash(t)
            if h in used_contexts:
                return "(앞서 언급된 내용과 동일)"
            used_contexts.add(h)
            return t
            
        ctx_before_dedup = _get_context_str(ctx_before)
        ctx_after_dedup = _get_context_str(ctx_after)
        
        sblock = {
            "id": f"s_{b['id']}",
            "type": f"{role}_summary",
            "source_block_id": b["id"],
            "bbox": b["bbox"],
            "confidence": b.get("confidence", 1.0),
            "text": "",
            "meta": {
                "source_type": b.get("type"),
                "visual_subtype": meta.get("visual_subtype"),
                "associated_title": ass_title,
                "used_context_blocks": bool(ctx_before or ctx_after),
            }
        }
        
        if role == "table_like":
            sblock["meta"]["summary_generation_method"] = "table_compaction"
            
            meta_obj = b.get("meta", {})
            extra_obj = b.get("extra", {})
            norm = meta_obj.get("normalized_table") or extra_obj.get("normalized_table") or {}
            
            rows = norm.get("rows")
            if rows is None:
                rows = meta_obj.get("rows") or extra_obj.get("rows", [])
            
            sblock["meta"]["table_quality"] = norm.get("quality", {}).get("overall_table_quality", 0.0)
            
            clean_rows = []
            for r in rows:
                if not isinstance(r, list): continue
                r_clean = [str(c).strip().replace("\n", " ") for c in r if c is not None]
                if any(r_clean):
                    clean_rows.append(r_clean)
                    
            if not clean_rows:
                continue
            else:
                is_giant = "giant_dump_table" in meta_obj.get("table_reject_signals", [])
                hdr = clean_rows[0]
                
                if is_giant:
                    if summary_debug is not None:
                        summary_debug["table_reconstruction_attempts"] = summary_debug.get("table_reconstruction_attempts", 0) + 1
                        
                    all_cells = [c for r in clean_rows for c in r]
                    chunks = []
                    current_chunk = []
                    for c in all_cells:
                        current_chunk.append(c)
                        if len(current_chunk) > 1 and re.search(r'\d', c):
                            if sum(1 for x in current_chunk if re.search(r'\d', x)) >= 5:
                                chunks.append(current_chunk)
                                current_chunk = []
                    if current_chunk:
                        chunks.append(current_chunk)
                        
                    if len(chunks) > 0 and len(chunks[0]) > 0:
                        hdr = chunks[0][:10]
                        disp_rows = chunks[1:4]
                        sblock["meta"]["compact_reconstruction"] = True
                        if summary_debug is not None:
                            summary_debug.setdefault("table_reconstruction_success", []).append(b["id"])
                    else:
                        if summary_debug is not None:
                            summary_debug.setdefault("table_reconstruction_failures", []).append(b["id"])
                        skip_counts["giant_dump_table"] = skip_counts.get("giant_dump_table", 0) + 1
                        skip_events.append({"id": b["id"], "type": role, "reason": "giant_dump_table_reconstruct_fail"})
                        continue
                else:
                    sblock["meta"]["header_row"] = hdr
                    sblock["meta"]["num_rows"] = len(clean_rows)
                    sblock["meta"]["num_cols"] = len(hdr)
                    
                    compaction = False
                    if len(clean_rows) > 7:
                        compaction = True
                        disp_rows = clean_rows[1:4] + [["..."]] + clean_rows[-2:]
                    else:
                        disp_rows = clean_rows[1:]
                        
                    sblock["meta"]["table_compaction"] = compaction
                
                t_lines = []
                if ass_title: t_lines.append(f"[TABLE] 제목: {ass_title}")
                if ctx_before_dedup: t_lines.append(f"이전 맥락: {ctx_before_dedup}")
                t_lines.append(f"컬럼: {' | '.join(hdr[:10])}")
                t_lines.append("주요 행:")
                for r in disp_rows:
                    t_lines.append("  - " + " | ".join(r[:10]))
                if ctx_after_dedup: t_lines.append(f"이후 맥락: {ctx_after_dedup}")
                
                sblock["text"] = "\n".join(t_lines)
                
        elif role in ("chart_like", "image_like"):
            sblock["meta"]["summary_generation_method"] = "visual_ocr"
            x0, y0, x1, y1 = b["bbox"]
            clip = fitz.Rect(max(0, x0-2), max(0, y0-2), min(page_width, x1+2), min(page_height, y1+2))
            
            ocr_text = ""
            used_ocr = False
            
            try:
                mat = fitz.Matrix(3.0, 3.0)
                pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                
                ocr_result = run_ocr_on_image(img_arr, page_width=clip.width, page_height=clip.height, zoom=3.0)
                if ocr_result["success"]:
                    used_ocr = True
                    ocr_tokens = []
                    legend_c = []
                    axis_c = []
                    num_c = []
                    
                    for ob in ocr_result.get("boxes", []):
                        t = ob["text"].strip()
                        if len(t) < 2: continue
                        ocr_tokens.append(t)
                        if re.search(r'\d', t): num_c.append(t)
                        elif len(t) <= 15: legend_c.append(t)
                        else: axis_c.append(t)
                        
                    ocr_text = " ".join(ocr_tokens)
                    sblock["meta"]["visual_ocr_text"] = ocr_text
                    sblock["meta"]["legend_candidates"] = legend_c[:10]
                    sblock["meta"]["numeric_tokens"] = num_c[:10]
                    
            except Exception as e:
                sblock["meta"]["summary_warnings"] = f"OCR failed: {e}"
                
            sblock["meta"]["used_ocr"] = used_ocr
            
            ctype = "CHART" if role == "chart_like" else "IMAGE"
            t_lines = [f"[{ctype}] 제목: {ass_title if ass_title else '(없음)'}"]
            if ctx_before_dedup: t_lines.append(f"주변 설명(상): {ctx_before_dedup}")
            if ocr_text:
                if len(ocr_text) > 200: ocr_text = ocr_text[:200] + "..."
                t_lines.append(f"내부 텍스트: {ocr_text}")
            if ctx_after_dedup: t_lines.append(f"주변 설명(하): {ctx_after_dedup}")
            
            # area_ratio thresholds for noise suppression
            area_ratio = b.get("extra", {}).get("area_ratio", 1.0)
            
            # Record judgment values in parser_debug
            sblock["meta"]["judgment_values"] = {
                "ocr_text_length": len(ocr_text),
                "area_ratio": round(area_ratio, 4),
                "meaningful_letters": sum(len(re.findall(r'[가-힣a-zA-Z]', t)) for t in ocr_text.split()),
                "has_ass_title": bool(ass_title),
                "has_context": bool(ctx_before or ctx_after)
            }
            
            meaningful_letters = sblock["meta"]["judgment_values"]["meaningful_letters"]
            
            # --- ENHANCED NOISE SUPPRESSION (fixes 미래에셋 micro-summary) ---
            # 1. Pure visual noise text → always skip regardless of ass_title/context
            if used_ocr and _is_visual_noise_text(ocr_text):
                skip_counts["visual_noise_unconditional"] = skip_counts.get("visual_noise_unconditional", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "visual_noise_unconditional"})
                if summary_debug is not None:
                    summary_debug["micro_crop_skips"] = summary_debug.get("micro_crop_skips", 0) + 1
                continue
            
            # 2. Too few meaningful letters with short OCR text → noise (강화: 6, 20)
            if (meaningful_letters < 6 and len(ocr_text) < 20):
                skip_counts["micro_noise_letters"] = skip_counts.get("micro_noise_letters", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "micro_noise_few_letters"})
                if summary_debug is not None:
                    summary_debug["micro_crop_skips"] = summary_debug.get("micro_crop_skips", 0) + 1
                continue
            
            # 장식성 요소 (area_ratio가 매우 작고 제목도 없는 경우)
            if area_ratio < 0.005 and not ass_title:
                skip_counts["tiny_decorative_crop"] = skip_counts.get("tiny_decorative_crop", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "tiny_decorative_crop"})
                continue
            
            # 3. Empty OCR — no text at all → worthless summary
            if not ocr_text.strip():
                skip_counts["empty_ocr_visual"] = skip_counts.get("empty_ocr_visual", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "empty_ocr_visual"})
                continue
            
            sblock["text"] = "\n".join(t_lines)
            
            if not ocr_text and not ctx_before and not ctx_after and not ass_title:
                skip_counts["meaningless_visual"] = skip_counts.get("meaningless_visual", 0) + 1
                skip_events.append({"id": b["id"], "type": role, "reason": "meaningless_visual"})
                continue
                
        summary_blocks.append(sblock)
        
    return summary_blocks
# ── per-page processing ─────────────────────────────────────────────

def _process_page(
    doc: fitz.Document,
    idx: int,
    doc_id: str,
    plumber_table_data: list[dict[str, Any]],
) -> dict[str, Any]:
    page: fitz.Page = doc[idx]
    rect = page.rect
    pw, ph = rect.width, rect.height
    dims = f"{round(pw, 1)} x {round(ph, 1)}"

    warnings: list[str] = []
    extraction_order: list[str] = []
    blocks: list[dict[str, Any]] = []
    block_counter = 0
    
    dropped_blocks: list[dict[str, Any]] = []
    bbox_warnings: list[str] = []

    # ── 0. Preview image ──────────────────────────────────────────
    preview_path, preview_w, preview_h, scale_x, scale_y, preview_err = _render_preview(page, pw, ph, doc_id, idx, warnings)

    # ── 1. PyMuPDF text blocks (dict mode) ────────────────────────
    native_text = ""
    try:
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for bk in text_dict.get("blocks", []):
            if bk.get("type") == 0:  # text block
                bbox = [round(bk["bbox"][i], 2) for i in range(4)]
                block_text = ""
                for line in bk.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                    block_text += "\n"
                block_text = block_text.strip()
                if not block_text:
                    continue

                btype = _classify_text_block(bk, bbox, ph)
                blocks.append({
                    "id": f"p{idx+1}_b{block_counter}",
                    "type": btype,
                    "bbox": bbox,
                    "text": block_text,
                    "confidence": 1.0,
                    "source": "pymupdf",
                    "extra": {},
                })
                block_counter += 1
                native_text += block_text + "\n"
        extraction_order.append("pymupdf_blocks")
    except Exception as exc:
        warnings.append(f"PyMuPDF dict extraction failed: {exc}")
        # fallback: plain text
        try:
            native_text = page.get_text("text").strip()
            extraction_order.append("pymupdf_plain_fallback")
        except Exception as exc2:
            warnings.append(f"PyMuPDF plain text also failed: {exc2}")

    native_text = native_text.strip()

    # ── 2. pdfplumber table blocks ────────────────────────────────
    table_bboxes: list[list[float]] = []
    tables_for_page: list[list[list[str]]] = []
    if plumber_table_data:
        for ti, tinfo in enumerate(plumber_table_data):
            tbl_bbox = tinfo.get("bbox", [0, 0, pw, ph])
            rows = tinfo.get("rows", [])
            table_bboxes.append(tbl_bbox)
            tables_for_page.append(rows)
            blocks.append({
                "id": f"p{idx+1}_b{block_counter}",
                "type": "table",
                "bbox": [round(v, 2) for v in tbl_bbox],
                "text": _table_to_text(rows),
                "confidence": 1.0,
                "source": "pdfplumber",
                "extra": {"rows": rows},
            })
            block_counter += 1
        extraction_order.append("pdfplumber_tables")

    # ── 3. Camelot fallback (only when pdfplumber found no tables) ─
    if not plumber_table_data:
        camelot_tables = _extract_tables_via_camelot(doc.name, idx, pw, ph, warnings)
        accepted_camelot_tables = []
        for ct in camelot_tables:
            is_accepted, reason = _accept_camelot_table(
                rows=ct["rows"],
                bbox=ct["bbox"],
                page_width=pw,
                page_height=ph,
                accuracy=ct.get("accuracy", 0.0)
            )
            if is_accepted:
                accepted_camelot_tables.append(ct)
            else:
                warnings.append(f"Camelot rejected on page {idx+1}: {reason}")

        if accepted_camelot_tables:
            for ct in accepted_camelot_tables:
                table_bboxes.append(ct["bbox"])
                tables_for_page.append(ct["rows"])
                blocks.append({
                    "id": f"p{idx+1}_b{block_counter}",
                    "type": "table",
                    "bbox": ct["bbox"],
                    "text": _table_to_text(ct["rows"]),
                    "confidence": ct.get("accuracy", 0.0) / 100.0,
                    "source": "camelot",
                    "extra": {"rows": ct["rows"], "accuracy": ct.get("accuracy")},
                })
                block_counter += 1
            extraction_order.append("camelot_tables")

    # ── 4. Image / chart-like heuristics ──────────────────────────
    image_list = page.get_images(full=True)
    image_count = len(image_list)
    try:
        for img_info in image_list:
            xref = img_info[0]
            img_rect = _find_image_rect(page, xref)
            if img_rect is None:
                continue
            ibbox = [round(img_rect.x0, 2), round(img_rect.y0, 2),
                     round(img_rect.x1, 2), round(img_rect.y1, 2)]
            iw = ibbox[2] - ibbox[0]
            ih = ibbox[3] - ibbox[1]
            if iw <= 0 or ih <= 0:
                continue
            aspect = iw / ih
            area_ratio = (iw * ih) / (pw * ph) if pw * ph else 0

            if (CHART_MIN_ASPECT <= aspect <= CHART_MAX_ASPECT
                    and area_ratio >= CHART_MIN_AREA_RATIO):
                btype = "chart_like"
            else:
                btype = "image"

            blocks.append({
                "id": f"p{idx+1}_b{block_counter}",
                "type": btype,
                "bbox": ibbox,
                "text": "",
                "confidence": 1.0,
                "source": "heuristic",
                "extra": {"xref": xref, "aspect_ratio": round(aspect, 3),
                          "area_ratio": round(area_ratio, 4)},
            })
            block_counter += 1
        extraction_order.append("image_heuristics")
    except Exception as exc:
        warnings.append(f"Image heuristic failed: {exc}")

    # ── 5. Remove text blocks that overlap table regions ──────────
    # Only use pdfplumber table bboxes for text suppression to avoid Camelot false positives
    # suppressing valid native text blocks.
    if plumber_table_data:
        plumber_bboxes = [tb.get("bbox", [0, 0, pw, ph]) for tb in plumber_table_data]
        blocks = _filter_overlapping_text_blocks(blocks, plumber_bboxes)

    # ── 6. OCR (only when needed) ─────────────────────────────────
    early_layout_hint = _infer_page_layout_hint(blocks, pw, ph)
    need_ocr, ocr_reason = _should_ocr(native_text, image_count, early_layout_hint)
    ocr_applied = False
    ocr_confidence = 0.0
    ocr_engine: str | None = None
    fallback_reason: str | None = None
    merge_strategy = "native_only"
    final_text = native_text
    text_source = "native"

    if need_ocr:
        try:
            from .ocr_utils import run_ocr_on_image

            mat = fitz.Matrix(BLOCK_ZOOM, BLOCK_ZOOM)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            ocr_result = run_ocr_on_image(
                img_arr, page_width=pw, page_height=ph, zoom=BLOCK_ZOOM
            )
            ocr_applied = ocr_result["success"]
            ocr_confidence = ocr_result.get("confidence", 0.0)
            ocr_engine = ocr_result.get("engine")
            fallback_reason = ocr_result.get("fallback_reason")

            if ocr_applied:
                # Add OCR boxes as blocks
                for ob in ocr_result.get("boxes", []):
                    blocks.append({
                        "id": f"p{idx+1}_b{block_counter}",
                        "type": "text",
                        "bbox": ob["bbox"],
                        "text": ob["text"],
                        "confidence": ob["confidence"],
                        "source": f"ocr_{ocr_engine}",
                        "extra": {},
                    })
                    block_counter += 1

                ocr_text = ocr_result.get("text", "")
                if len(native_text) < OCR_TEXT_THRESHOLD:
                    # Native text too sparse → replace
                    final_text = ocr_text
                    text_source = "ocr"
                    merge_strategy = "ocr_replace"
                else:
                    # Native text exists → append OCR supplement
                    final_text = native_text + "\n\n[OCR]\n" + ocr_text
                    text_source = "hybrid"
                    merge_strategy = "hybrid_merge"

                extraction_order.append("ocr_boxes")
            else:
                if ocr_result.get("error"):
                    warnings.append(f"OCR failed: {ocr_result['error']}")
        except Exception as exc:
            warnings.append(f"OCR pipeline error: {exc}")
    # ── 7. Phase 2 Heuristics Post-Processing ────────────────────
    validated_pre = []
    for bk in blocks:
        validated = _validate_and_format_block(bk, pw, ph, idx + 1, dropped_blocks, bbox_warnings)
        if validated: validated_pre.append(validated)
        
    # --- TABLE PIPELINE V1 ---
    from .table_utils import (
        detect_table_candidates,
        segment_dashboard_subtables,
        merge_subtable_fragments,
        normalize_table_candidate,
        score_table_quality,
        choose_best_table_candidate
    )
    
    # 1. Detect candidates (assign candidate scores & drop false tables)
    detect_table_candidates(pw, ph, validated_pre)
    
    # 2. Segment dashboards (now with table scores)
    segment_dashboard_subtables(validated_pre)
    
    # 2.5 Merge subtable fragments
    validated_pre = merge_subtable_fragments(validated_pre, pw, ph)
    
    # 3. Normalize tables
    table_candidates = []
    for b in validated_pre:
        if b["type"] == "table" or b.get("meta", {}).get("table_candidate_score", 0) > 1.0:
            source = str(b.get("source", "native"))
            
            # Note: _validate_and_format_block moves "extra" to "meta"
            meta_obj = b.get("meta", {})
            raw_data = (
                meta_obj.get("normalized_table")
                or meta_obj.get("rows")
                or meta_obj.get("cells")
                or b.get("text", "")
            )
            
            norm_tbl = normalize_table_candidate(raw_data, source, b.get("text"))
            quality_score = score_table_quality(norm_tbl)
            
            rskt = b["meta"].setdefault("table_reject_signals", [])
            for r in norm_tbl.get("reject_signals", []):
                if r not in rskt:
                    rskt.append(r)
            
            # --- PATCH V1: Quality Gate ---
            if quality_score < 0.5 and "giant_dump_table" not in rskt:
                continue
            
            b["meta"]["normalized_table"] = norm_tbl
            b["meta"]["table_quality"] = quality_score
            b["meta"]["table_quality_details"] = norm_tbl.get("quality", {})
            b["meta"]["table_markdown"] = norm_tbl.get("markdown", "")
            table_candidates.append(b)

    # 4. Filter overlapping tables / Multi-engine selection
    dropped_table_ids = set()
    for t in table_candidates:
        if t["id"] in dropped_table_ids: continue
        group = [t]
        for other_t in table_candidates:
            if t["id"] != other_t["id"] and other_t["id"] not in dropped_table_ids:
                if _rects_overlap(t["bbox"], other_t["bbox"], 0.5):
                    group.append(other_t)
                    dropped_table_ids.add(other_t["id"])
        
        if len(group) > 1:
            norm_cands = [g["meta"]["normalized_table"] for g in group if "normalized_table" in g.get("meta", {})]
            best_norm = choose_best_table_candidate(norm_cands)
            
            for g in group:
                # Keep only the block that generated the best_norm
                if "normalized_table" not in g.get("meta", {}) or g["meta"]["normalized_table"] != best_norm:
                    g["type"] = "unknown" 
                    dropped_table_ids.add(g["id"])

    # Clean up rejected tables
    validated_pre = [b for b in validated_pre if b["id"] not in dropped_table_ids or b["type"] != "unknown"]
    # ---------------------------
        
    quality_notes = []
    merge_events = []
    classification_overrides = []
    dedup_stats = {}
    text_merge_stats = {}
    
    valid_blocks = _reclassify_images(validated_pre, pw, ph, quality_notes, classification_overrides, dropped_blocks)
    valid_blocks = _deduplicate_blocks(valid_blocks, merge_events, dedup_stats, quality_notes)
    valid_blocks = _merge_adjacent_text_blocks(valid_blocks, merge_events, text_merge_stats, pw, ph)
    if text_merge_stats.get("merged_pairs", 0) > 0:
        quality_notes.append(f"text_blocks_merged:{text_merge_stats['merged_pairs']}")
    
    page_layout_hint = _infer_page_layout_hint(valid_blocks, pw, ph)
    valid_blocks, strategy, basis = _sort_reading_order(valid_blocks, page_layout_hint, pw, ph)
    
    if not quality_notes:
        quality_notes.append("no_major_adjustments")
        
    summary_debug = {
        "summary_ready_checks": 0,
        "summary_ready_failures": 0,
        "table_reconstruction_attempts": 0,
        "table_reconstruction_success": [],
        "table_reconstruction_failures": []
    }
    
    is_dash, dash_signals = False, [] # legacy, handled in page_type below
    is_cover, cover_signals = False, [] # legacy, handled in page_type below
    
    page_type, page_type_signals = _classify_page_type(native_text, valid_blocks, pw, ph, idx + 1)
    
    if page_type == "dashboard_kpi_like":
        page_layout_hint = "dashboard_like"
        summary_debug["dashboard_like_pages"] = summary_debug.get("dashboard_like_pages", 0) + 1
    elif page_type == "cover_or_disclaimer_like":
        page_layout_hint = page_layout_hint + ",cover_like" if page_layout_hint else "cover_like"
        summary_debug["cover_like_pages"] = summary_debug.get("cover_like_pages", 0) + 1
        
    _assign_block_scores(valid_blocks)
    _apply_summary_normalization(valid_blocks, ph, text_source, page_layout_hint, summary_debug)
    summary_debug["_page_layout_hint"] = page_layout_hint  # for final gate use
    
    block_type_counts = {"text": 0, "table": 0, "image": 0, "chart": 0, "title": 0, "footer": 0, "unknown": 0}
    for b in valid_blocks:
        t = b["type"]
        block_type_counts[t] = block_type_counts.get(t, 0) + 1
        
    candidate_counts = {
        "pymupdf_text_blocks": sum(1 for b in blocks if b["source"] == "pymupdf"),
        "pdfplumber_tables": sum(1 for b in blocks if b["source"] == "pdfplumber"),
        "camelot_tables": sum(1 for b in blocks if b["source"] == "camelot"),
        "image_candidates": sum(1 for b in blocks if b["source"] == "heuristic"),
        "ocr_boxes": sum(1 for b in blocks if str(b["source"]).startswith("ocr_")),
        "final_blocks": len(valid_blocks)
    }

    summary_skip_counts = {}
    summary_skip_events = []
    
    summary_blocks = _generate_summary_blocks(valid_blocks, page, pw, ph, summary_skip_counts, summary_skip_events, summary_debug)

    parser_debug = {
        "preview_generated": preview_path is not None,
        "preview_error": preview_err,
        "native_text_chars": len(native_text),
        "ocr_used": ocr_applied,
        "ocr_trigger_reason": ocr_reason,
        "ocr_engine_used": ocr_engine,
        "merge_strategy": merge_strategy,
        "fallback_reason": fallback_reason,
        "candidate_counts": candidate_counts,
        "block_type_counts": block_type_counts,
        "dropped_blocks": dropped_blocks,
        "bbox_warnings": bbox_warnings,
        "parse_warnings": warnings,
        "extraction_order": extraction_order,
        "quality_notes": quality_notes,
        "merge_events": merge_events,
        "classification_overrides": classification_overrides,
        "dedup_stats": dedup_stats,
        "text_merge_stats": text_merge_stats,
        "page_layout_hint": page_layout_hint,
        "reading_order_strategy": strategy,
        "reading_order_basis": basis,
        "summary_skip_counts": summary_skip_counts,
        "summary_skip_events": summary_skip_events,
        "summary_debug": summary_debug,
        "page_type": page_type,
        "page_type_reason": page_type_signals
    }

    return {
        "page_num": idx + 1,
        "page_width": pw,
        "page_height": ph,
        "preview_width": preview_w,
        "preview_height": preview_h,
        "preview_scale_x": scale_x,
        "preview_scale_y": scale_y,
        "coord_space": "page_points",
        "preview_image": preview_path,
        "text": final_text,
        "tables": tables_for_page,
        "blocks": valid_blocks,
        "summary_blocks": summary_blocks,
        "image_count": image_count,
        "text_source": text_source,
        "ocr_applied": ocr_applied,
        "ocr_confidence": ocr_confidence,
        "parser_debug": parser_debug,
    }


# ── block classification ────────────────────────────────────────────

def _classify_text_block(
    block: dict, bbox: list[float], page_height: float
) -> str:
    """Heuristic classification of a PyMuPDF text block."""
    y0, y1 = bbox[1], bbox[3]
    height = y1 - y0

    # Title heuristic: near top, large font
    lines = block.get("lines", [])
    if lines:
        max_size = max(
            (span.get("size", 0) for line in lines for span in line.get("spans", [])),
            default=0,
        )
        if max_size >= 14 and y0 < page_height * 0.15:
            return "title"

    # Footer heuristic: near bottom, small
    if y1 > page_height * 0.92 and height < 20:
        return "footer"

    return "text"


# ── preview rendering ───────────────────────────────────────────────

def _render_preview(
    page: fitz.Page, pw: float, ph: float, doc_id: str, idx: int, warnings: list[str]
) -> tuple[str | None, float | None, float | None, float | None, float | None, str | None]:
    """Render page to PNG and return (api_path, pw_px, ph_px, scale_x, scale_y, error_msg)."""
    try:
        if not _result_dir:
            msg = "RESULT_DIR not set — preview skipped"
            warnings.append(msg)
            return None, None, None, None, None, msg
        out_dir = _preview_dir(doc_id)
        fname = f"page_{idx + 1}.{PREVIEW_FORMAT}"
        fpath = os.path.join(out_dir, fname)

        mat = fitz.Matrix(PREVIEW_ZOOM, PREVIEW_ZOOM)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(fpath)

        path = f"/api/documents/{doc_id}/pages/{idx + 1}/preview"
        return path, float(pix.width), float(pix.height), float(pix.width)/pw, float(pix.height)/ph, None
    except Exception as exc:
        msg = f"Preview generation failed: {exc}"
        warnings.append(msg)
        return None, None, None, None, None, msg


# ── image rect lookup ────────────────────────────────────────────────

def _find_image_rect(page: fitz.Page, xref: int) -> fitz.Rect | None:
    """Find the bounding rect of an image on the page by its xref."""
    try:
        for img in page.get_image_info(xrefs=True):
            if img.get("xref") == xref:
                bbox = img.get("bbox")
                if bbox:
                    return fitz.Rect(bbox)
        return None
    except Exception:
        return None


# ── table-text overlap filter ────────────────────────────────────────

def _rects_overlap(a: list[float], b: list[float], threshold: float = 0.5) -> bool:
    """Check if box `a` overlaps `b` by ≥ threshold of a's area."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    if area_a <= 0:
        return False
    return (inter / area_a) >= threshold


def _filter_overlapping_text_blocks(
    blocks: list[dict[str, Any]], table_bboxes: list[list[float]]
) -> list[dict[str, Any]]:
    """Remove text blocks whose bbox overlaps ≥50 % with any table bbox."""
    filtered = []
    for blk in blocks:
        if blk["type"] not in ("text", "title", "footer"):
            filtered.append(blk)
            continue
        overlaps = any(_rects_overlap(blk["bbox"], tb) for tb in table_bboxes)
        if not overlaps:
            filtered.append(blk)
    return filtered


# ── OCR trigger logic ────────────────────────────────────────────────

def _should_ocr(native_text: str, image_count: int, page_layout_hint: str = "") -> tuple[bool, str]:
    if len(native_text) == 0:
        return True, "native_text_empty"
        
    import re
    
    clean_text = native_text.replace(" ", "").replace("\n", "")
    if not clean_text:
         return True, "native_text_whitespace_only"
         
    meaningful = re.findall(r'[a-zA-Z0-9가-힣]', clean_text)
    meaningful_count = len(meaningful)
    meaningful_ratio = meaningful_count / len(clean_text)
    symbol_ratio = 1.0 - meaningful_ratio
    
    lines = [ln.strip() for ln in native_text.splitlines() if ln.strip()]
    
    meaningful_lines = 0
    short_tokens = 0
    total_tokens = 0
    
    for ln in lines:
        tokens = ln.split()
        total_tokens += len(tokens)
        short_tokens += sum(1 for t in tokens if len(t) <= 2)
        
        m_chars = re.findall(r'[a-zA-Z0-9가-힣]', ln)
        if len(m_chars) >= 3:
            meaningful_lines += 1
            
    meaningful_line_ratio = meaningful_lines / len(lines) if lines else 0.0
    short_token_ratio = short_tokens / total_tokens if total_tokens > 0 else 0
    
    if len(native_text) >= OCR_TEXT_THRESHOLD:
        if page_layout_hint in ("slide_like", "mixed_visual"):
            if meaningful_ratio < 0.45 and (symbol_ratio > 0.4 or meaningful_line_ratio < 0.5 or short_token_ratio > 0.5):
                return True, f"sensitive_slide_low_density(m_ratio={meaningful_ratio:.2f}, m_lines={meaningful_line_ratio:.2f}, sym={symbol_ratio:.2f})"
                
        if meaningful_ratio < 0.35 and (symbol_ratio > 0.6 or meaningful_line_ratio < 0.4 or short_token_ratio > 0.6):
            return True, f"low_alphanumeric_density(m_ratio={meaningful_ratio:.2f}, m_lines={meaningful_line_ratio:.2f}, sym={symbol_ratio:.2f})"
        return False, "ocr_not_needed"
        
    if image_count >= OCR_IMAGE_MIN:
        return True, "images_present_and_text_weak"
        
    if meaningful_ratio < 0.3:
        return True, "short_and_low_alphanumeric"
        
    return False, "native_text_below_threshold_but_no_images" 


# ── table extraction: pdfplumber ─────────────────────────────────────

def _extract_tables_via_pdfplumber(
    filepath: str, page_count: int
) -> dict[int, list[dict[str, Any]]]:
    """
    Returns ``{page_index: [{"bbox": [...], "rows": [[...], ...]}, ...]}``.
    """
    result: dict[int, list[dict[str, Any]]] = {}
    try:
        import pdfplumber

        with pdfplumber.open(filepath) as pdf:
            for idx, pg in enumerate(pdf.pages):
                try:
                    tables = pg.find_tables()
                    if not tables:
                        continue
                    page_tables = []
                    for tbl in tables:
                        bbox_raw = tbl.bbox  # (x0, top, x1, bottom)
                        rows_raw = tbl.extract()
                        if rows_raw is None:
                            continue
                        cleaned_rows = [
                            [(c.strip() if c else "") for c in row]
                            for row in rows_raw
                        ]
                        page_tables.append({
                            "bbox": [
                                round(bbox_raw[0], 2),
                                round(bbox_raw[1], 2),
                                round(bbox_raw[2], 2),
                                round(bbox_raw[3], 2),
                            ],
                            "rows": cleaned_rows,
                        })
                    if page_tables:
                        result[idx] = page_tables
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed: %s", exc)
    return result


# ── table extraction: Camelot fallback ───────────────────────────────

def _extract_tables_via_camelot(
    filepath: str,
    page_idx: int,
    page_width: float,
    page_height: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Camelot fallback — tries lattice then stream. Returns list of table dicts."""
    tables_out: list[dict[str, Any]] = []
    try:
        import camelot

        page_str = str(page_idx + 1)

        for flavour in ("lattice", "stream"):
            try:
                cam_tables = camelot.read_pdf(
                    filepath, pages=page_str, flavor=flavour,
                    suppress_stdout=True,
                )
                for ct in cam_tables:
                    df = ct.df
                    rows_list = [list(row) for _, row in df.iterrows()]
                    
                    cx0, cy0, cx1, cy1 = ct._bbox  # (x0, y0_bottom, x1, y1_bottom)
                    bbox = [
                        round(cx0, 2),
                        round(page_height - cy1, 2),
                        round(cx1, 2),
                        round(page_height - cy0, 2),
                    ]
                    
                    acc = round(ct.accuracy, 2) if hasattr(ct, "accuracy") else 0.0

                    tables_out.append({
                        "bbox": bbox,
                        "rows": rows_list,
                        "accuracy": acc,
                    })
                if tables_out:
                    break  # got results, no need for second flavour
            except Exception as exc:
                warnings.append(f"Camelot {flavour} failed on page {page_idx+1}: {exc}")

    except ImportError:
        warnings.append("camelot-py not installed — skipping Camelot fallback")
    except Exception as exc:
        warnings.append(f"Camelot error: {exc}")

    return tables_out


def _accept_camelot_table(
    rows: list[list[str]],
    bbox: list[float],
    page_width: float,
    page_height: float,
    accuracy: float
) -> tuple[bool, str]:
    """Validate Camelot table results before acceptance."""
    num_rows = len(rows)
    num_cols = len(rows[0]) if num_rows > 0 else 0

    if num_rows < CAM_TABLE_MIN_ROWS:
        return False, f"too_few_rows:{num_rows}"
    if num_cols < CAM_TABLE_MIN_COLS:
        return False, f"too_few_cols:{num_cols}"

    flat_cells = [c.strip() for r in rows for c in r]
    filled_cells = sum(1 for c in flat_cells if c)
    filled_ratio = filled_cells / (num_rows * num_cols) if (num_rows * num_cols) > 0 else 0
    if filled_ratio < CAM_TABLE_MIN_FILLED_RATIO:
        return False, f"low_filled_ratio:{filled_ratio:.2f}"

    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    page_area = page_width * page_height
    area_ratio = area / page_area if page_area > 0 else 0
    if area_ratio > CAM_TABLE_MAX_PAGE_AREA_RATIO:
        return False, f"bbox_too_large:{area_ratio:.2f}"

    if accuracy < CAM_TABLE_MIN_ACCURACY:
        return False, f"low_accuracy:{accuracy}"

    return True, "accepted"


# ── helper: table rows → plain text ─────────────────────────────────

def _table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row) for row in rows)


# ── metadata ─────────────────────────────────────────────────────────

def _extract_metadata(doc: fitz.Document) -> dict[str, Any]:
    raw = doc.metadata or {}
    return {
        "page_count": doc.page_count,
        "author": raw.get("author", "") or "",
        "title": raw.get("title", "") or "",
        "creator": raw.get("creator", "") or "",
        "producer": raw.get("producer", "") or "",
    }



# ── Type-specific Post-processing ───────────────────────────────────

def _refine_blocks_by_type(pages: list[dict[str, Any]], doc_type: str) -> None:
    """Adjusts block-level metadata and filters based on document-wide classification."""
    if doc_type == "slide_ir":
        # Slide IR: Often contains many decorative crops or repetitive footers.
        for p in pages:
            for b in p.get("blocks", []):
                meta = b.setdefault("meta", {})
                if meta.get("summary_exclude"):
                    continue
                
                # Further suppress very short text on slides (often page nums/decorations)
                text = b.get("text", "").strip()
                if len(text) < 15 and b["type"] == "text" and not meta.get("summary_role") == "title":
                    meta["summary_exclude"] = True
                    meta["summary_exclude_reason"] = "slide_micro_text"
                    
    elif doc_type == "dashboard_brief":
        # Dashboard Brief: Focus on KPI numbers and visual summaries.
        # Be extremely strict about narrative snippets that look like noise.
        for p in pages:
            for b in p.get("blocks", []):
                meta = b.setdefault("meta", {})
                if meta.get("summary_exclude"):
                    continue
                
                text = b.get("text", "").strip()
                num_ratio = sum(1 for c in text if c.isdigit()) / max(1, len(text))
                
                if b["type"] == "text" and len(text) < 50 and num_ratio < 0.1:
                    # Non-numeric short text in a dashboard is usually decorative noise
                    meta["summary_exclude"] = True
                    meta["summary_exclude_reason"] = "dashboard_narrative_noise"
                    
    elif doc_type == "table_heavy":
        # Table Heavy: Ensure tables are marked with high priority.
        for p in pages:
            for b in p.get("blocks", []):
                if b["type"] == "table":
                    meta = b.setdefault("meta", {})
                    meta["summary_priority"] = "high"
                    meta["is_primary_content"] = True

    elif doc_type == "text_report":
        # Text Report: Default behavior, keep balanced.
        pass


# ── quality assessment ───────────────────────────────────────────────

def _assess_quality(
    pages: list[dict],
    ocr_page_nums: list[int],
    empty_pages: int,
) -> str:
    if not pages:
        return "empty"

    total_text = sum(len(p.get("text", "")) for p in pages)
    n = len(pages)
    avg = total_text / n

    ocr_ratio = len(ocr_page_nums) / n if n else 0
    empty_ratio = empty_pages / n if n else 0

    if avg > 200 and empty_ratio < 0.2:
        return "good"
    if avg > 80 or (ocr_ratio > 0 and avg > 40):
        return "partial"
    return "poor"
