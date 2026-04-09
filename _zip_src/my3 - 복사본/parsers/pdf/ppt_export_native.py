import logging
import re
from typing import Any, Dict, List

import fitz

logger = logging.getLogger(__name__)

def recover_ppt_export_blocks(page: fitz.Page, page_idx: int, pw: float, ph: float) -> Dict[str, Any]:
    """
    Recover text blocks from PPT-export style PDF pages using low-level reassembly.
    Tries to rebuild what PyMuPDF's default 'dict' might miss or split excessively.
    """
    try:
        # 1. Get rawdict with flags to preserve everything
        # TEXT_PRESERVE_WHITESPACE is key for some PPT exports
        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        
        # 2. Collect all spans with their metrics
        all_spans = []
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    # In rawdict, spans have 'chars', not 'text'
                    chars = span.get("chars", [])
                    text = "".join(c.get("c", "") for c in chars).strip()
                    if not text:
                        continue
                    
                    bbox = list(span.get("bbox", [0, 0, 0, 0]))
                    all_spans.append({
                        "text": text,
                        "bbox": bbox,
                        "font": span.get("font", ""),
                        "size": round(span.get("size", 0), 2),
                        "color": span.get("color", 0),
                        "origin": span.get("origin", (0, 0))
                    })

        if not all_spans:
            return {"blocks": [], "success": False, "source": "none", "recovered_text_len": 0}

        # 3. Line Reassembly (Group by Y-origin)
        lines = _reassemble_lines(all_spans)
        
        # 4. Block Reassembly (Group lines by X-alignment and Y-gap)
        raw_blocks = _reassemble_blocks(lines, pw, ph)
        
        # 5. Final Block Normalization and Role Assignment
        final_blocks = []
        for i, rb in enumerate(raw_blocks):
            text = rb["text"]
            bbox = rb["bbox"]
            max_font = rb["max_font"]
            
            role = _classify_ppt_block(text, bbox, pw, ph, max_font)
            if role == "decorative_noise" and len(text) < 5:
                continue
                
            block_type = "title" if role == "title" else "footer" if role == "footer" else "text"
            
            final_blocks.append({
                "id": f"p{page_idx+1}_ptr{i}",
                "type": block_type,
                "bbox": bbox,
                "text": text,
                "source": "ppt_export_native_rawdict",
                "confidence": 0.95,
                "meta": {
                    "slide_role": role,
                    "font_size": round(max_font, 2),
                    "summary_priority": "high" if role in ("title", "kpi_metric", "bullet_item") else "medium",
                    "summary_exclude": role in ("footer", "decorative_noise"),
                }
            })

        recovered_len = sum(len(b["text"]) for b in final_blocks)
        
        return {
            "blocks": final_blocks,
            "success": True if final_blocks else False,
            "source": "ppt_export_native_rawdict",
            "recovered_text_len": recovered_len,
            "debug": {
                "attempted_rawdict": True,
                "attempted_mcid": False, # Future placeholder
                "recovered_line_count": len(lines),
                "recovered_block_count": len(final_blocks),
                "span_count": len(all_spans)
            }
        }

    except Exception as e:
        logger.warning(f"PPT native recovery failed on page {page_idx+1}: {e}")
        return {
            "blocks": [], 
            "success": False, 
            "source": "none", 
            "failure_reason": str(e)
        }

def _reassemble_lines(spans: List[Dict]) -> List[Dict]:
    """Sort and group spans into lines based on vertical alignment."""
    if not spans:
        return []
    
    # Sort by Y-origin (top to bottom), then by X-origin (left to right)
    sorted_spans = sorted(spans, key=lambda s: (s["origin"][1], s["origin"][0]))
    
    lines = []
    if not sorted_spans:
        return []
        
    current_line = [sorted_spans[0]]
    
    for i in range(1, len(sorted_spans)):
        s1 = current_line[-1]
        s2 = sorted_spans[i]
        
        # Heuristic for the same line: 
        # Origins are very close OR BBoxes overlap significantly vertically
        y_diff = abs(s1["origin"][1] - s2["origin"][1])
        
        # Tolerance for y-variation (some PPT exports shift characters slightly)
        # Usually 2-3 points is enough
        if y_diff < 2.5:
            current_line.append(s2)
        else:
            # Finalize previous line
            lines.append(_finalize_line(current_line))
            current_line = [s2]
            
    lines.append(_finalize_line(current_line))
    return lines

def _finalize_line(line_spans: List[Dict]) -> Dict:
    # Sort by X-origin just in case
    line_spans.sort(key=lambda s: s["origin"][0])
    
    # Join text with space if there's a significant gap
    text_parts = []
    for i in range(len(line_spans)):
        if i > 0:
            prev = line_spans[i-1]
            curr = line_spans[i]
            gap = curr["bbox"][0] - prev["bbox"][2]
            # If gap is more than 0.2 of font size, add space
            if gap > curr["size"] * 0.2:
                text_parts.append(" ")
        text_parts.append(line_spans[i]["text"])
    
    text = "".join(text_parts).strip()
    
    xs0 = [s["bbox"][0] for s in line_spans]
    ys0 = [s["bbox"][1] for s in line_spans]
    xs1 = [s["bbox"][2] for s in line_spans]
    ys1 = [s["bbox"][3] for s in line_spans]
    
    return {
        "text": text,
        "bbox": [min(xs0), min(ys0), max(xs1), max(ys1)],
        "max_font": max(s["size"] for s in line_spans),
        "spans": line_spans
    }

def _reassemble_blocks(lines: List[Dict], pw: float, ph: float) -> List[Dict]:
    """Group lines into blocks based on vertical flow and horizontal alignment."""
    if not lines:
        return []
    
    # Lines should already be sorted by Y
    blocks = []
    if not lines:
        return []
        
    current_block = [lines[0]]
    
    for i in range(1, len(lines)):
        l1 = current_block[-1]
        l2 = lines[i]
        
        y_gap = l2["bbox"][1] - l1["bbox"][3]
        x_diff = abs(l2["bbox"][0] - l1["bbox"][0])
        
        # Heuristic for the same block
        # Usually 1.2x to 1.6x font size is typical spacing
        max_gap = l1["max_font"] * 1.6
        
        if -4 <= y_gap <= max_gap and x_diff < 40:
            current_block.append(l2)
        else:
            blocks.append(_finalize_block(current_block))
            current_block = [l2]
            
    blocks.append(_finalize_block(current_block))
    return blocks

def _finalize_block(block_lines: List[Dict]) -> Dict:
    text = "\n".join(l["text"] for l in block_lines).strip()
    
    xs0 = [l["bbox"][0] for l in block_lines]
    ys0 = [l["bbox"][1] for l in block_lines]
    xs1 = [l["bbox"][2] for l in block_lines]
    ys1 = [l["bbox"][3] for l in block_lines]
    
    return {
        "text": text,
        "bbox": [round(min(xs0), 2), round(min(ys0), 2), round(max(xs1), 2), round(max(ys1), 2)],
        "max_font": max(l["max_font"] for l in block_lines)
    }

def _classify_ppt_block(text: str, bbox: list, pw: float, ph: float, font_size: float) -> str:
    clean = text.strip()
    y0, y1 = bbox[1], bbox[3]
    width = bbox[2] - bbox[0]

    # Title
    if y0 < ph * 0.18 and (font_size >= 14 or (len(clean) <= 100 and width > pw * 0.2)):
        return "title"
    
    # Footer
    if y1 > ph * 0.92 and len(clean) <= 60:
        return "footer"
    
    # KPI
    if re.search(r'[\d,]+(?:\.\d+)?\s*(%|원|억원|조원|USD|pt|bps|배|x|yo|qoq)', clean, re.I) and len(clean) <= 60:
        return "kpi_metric"
        
    # Bullet
    if clean.startswith(("▪", "•", "-", "□", "◦", "*", "+")):
        return "bullet_item"
        
    # Noise
    if len(clean) <= 4 and not re.search(r'[A-Za-z가-힣]', clean):
        return "decorative_noise"
        
    return "body_note"
