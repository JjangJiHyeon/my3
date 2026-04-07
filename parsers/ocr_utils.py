"""
OCR utilities – EasyOCR primary, pytesseract fallback.

Every public function catches its own exceptions so the caller's
overall parse never crashes due to OCR failure.

Features
────────
* Multi-variant preprocessing with automatic quality-based selection
* Image style estimation for candidate narrowing
* Structural quality scoring (no string-based heuristics)
* Enhanced box cleanup (fragment merging, micro-noise removal)
"""

from __future__ import annotations

import re
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── lazy singletons ──────────────────────────────────────────────────

_easyocr_reader: Any = None
_easyocr_failed: bool = False


def _get_easyocr() -> Any:
    """Return shared EasyOCR Reader (ko+en). Created once."""
    global _easyocr_reader, _easyocr_failed
    if _easyocr_failed:
        return None
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        import easyocr
        _easyocr_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        logger.info("EasyOCR reader initialised (ko+en, CPU)")
        return _easyocr_reader
    except Exception as exc:
        _easyocr_failed = True
        logger.warning("EasyOCR init failed – will try pytesseract: %s", exc)
        return None


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ── image style estimation ──────────────────────────────────────────

def _estimate_image_style(image: np.ndarray) -> dict[str, Any]:
    """
    Analyze image statistics to determine preprocessing strategy.

    Returns dict with:
      mean_brightness, std_brightness, edge_density,
      is_low_contrast, is_dark, is_bright, is_noisy
    """
    import cv2

    if image is None or image.size == 0:
        return {"mean_brightness": 128, "std_brightness": 50, "edge_density": 0.1,
                "is_low_contrast": False, "is_dark": False, "is_bright": False, "is_noisy": False}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Sample center region (avoid borders which may be solid)
    h, w = gray.shape[:2]
    margin_y, margin_x = max(h // 10, 1), max(w // 10, 1)
    center = gray[margin_y:h - margin_y, margin_x:w - margin_x]

    mean_b = float(np.mean(center))
    std_b = float(np.std(center))

    # Edge density via Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / max(1, edges.size)

    return {
        "mean_brightness": round(mean_b, 1),
        "std_brightness": round(std_b, 1),
        "edge_density": round(edge_density, 4),
        "is_low_contrast": std_b < 40,
        "is_dark": mean_b < 100,
        "is_bright": mean_b > 210,
        "is_noisy": edge_density > 0.25,
    }


# ── preprocessing variants ──────────────────────────────────────────

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Light OpenCV preprocessing: grayscale → adaptive threshold. (Legacy single variant)"""
    import cv2
    if image is None or image.size == 0:
        return image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    return processed


def _build_ocr_preprocess_variants(
    image: np.ndarray,
    style: dict[str, Any],
) -> list[tuple[str, np.ndarray]]:
    """
    Build a list of (variant_name, preprocessed_image) candidates
    based on image style analysis. Returns 3-5 variants.
    """
    import cv2

    if image is None or image.size == 0:
        return [("raw", image)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    variants: list[tuple[str, np.ndarray]] = []

    # ── 1. gray_only: no thresholding (baseline) ──
    variants.append(("gray_only", gray.copy()))

    # ── 2. adaptive_gaussian: current default ──
    adaptive_g = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    variants.append(("adaptive_gaussian", adaptive_g))

    # ── 3. otsu_binary: automatic global threshold ──
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_binary", otsu))

    # ── 4-5. style-dependent variants ──
    if style.get("is_low_contrast") or style.get("is_dark"):
        # contrast boost via CLAHE + otsu
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, clahe_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("contrast_boost_otsu", clahe_otsu))

        # Also try CLAHE + adaptive for low-contrast
        clahe_adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6
        )
        variants.append(("contrast_boost_adaptive", clahe_adaptive))
    else:
        # For normal/bright images: light sharpen + adaptive
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        sharp_adaptive = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        variants.append(("sharpen_adaptive", sharp_adaptive))

    if style.get("is_noisy"):
        # Light denoise + adaptive for noisy images
        denoised = cv2.fastNlMeansDenoising(gray, h=8, searchWindowSize=15, templateWindowSize=7)
        denoise_adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        variants.append(("denoise_adaptive", denoise_adaptive))

    return variants


# ── OCR quality scoring ─────────────────────────────────────────────

def _score_ocr_result(
    boxes: list[dict],
    page_width: float,
    page_height: float,
) -> float:
    """
    Score OCR result quality using structural signals only.

    Higher = better. Components:
    - confidence-weighted char count (main signal)
    - valid box ratio bonus
    - micro-fragment penalty
    - low-coverage penalty
    """
    if not boxes:
        return 0.0

    page_area = max(1.0, page_width * page_height)
    total_boxes = len(boxes)

    # Confidence-weighted char count
    weighted_chars = 0.0
    valid_boxes = 0
    micro_fragments = 0
    total_box_area = 0.0
    total_chars = 0

    for box in boxes:
        text = box.get("text", "").strip()
        conf = box.get("confidence", 0.0)
        char_count = len(text)
        total_chars += char_count

        # Meaningful chars with high confidence
        weighted_chars += char_count * conf

        # Valid box: confidence > 0.4 and not trivially short
        if conf > 0.4 and char_count >= 2:
            valid_boxes += 1

        # Micro-fragment: 1-2 chars regardless of confidence
        if char_count <= 2:
            micro_fragments += 1

        # Box area coverage
        bbox = box.get("bbox", [0, 0, 0, 0])
        bw = max(0.0, bbox[2] - bbox[0])
        bh = max(0.0, bbox[3] - bbox[1])
        total_box_area += bw * bh

    # Valid box ratio [0-1]
    valid_ratio = valid_boxes / max(1, total_boxes)

    # Micro-fragment ratio [0-1]
    micro_ratio = micro_fragments / max(1, total_boxes)

    # Coverage ratio [0-1] capped at 0.5
    coverage = min(0.5, total_box_area / page_area)

    # Score composition
    score = (
        weighted_chars                       # main: prefer more text at higher confidence
        * (1.0 + valid_ratio * 0.3)          # bonus for high valid ratio
        * (1.0 + coverage * 0.2)             # small bonus for good coverage
        - micro_ratio * total_chars * 0.3    # penalty for fragmented output
    )

    # If almost no text, heavy penalty
    if total_chars < 10:
        score *= 0.1

    return round(max(0.0, score), 2)


# ── box cleanup ──────────────────────────────────────────────────────

def _cleanup_ocr_boxes(boxes: list[dict]) -> list[dict]:
    """
    Post-OCR box cleanup:
    - Remove very short low-confidence fragments
    - Remove single-char noise boxes
    - Merge horizontally adjacent fragments on same line
    - Remove numeric/symbol-only micro-boxes
    """
    if not boxes:
        return []

    # Step 1: Filter noise
    filtered = []
    for box in boxes:
        text = box.get("text", "").strip()
        conf = box.get("confidence", 0.0)

        # Very short + low confidence
        if len(text) <= 1 and conf < 0.7:
            continue
        if len(text) <= 2 and conf < 0.5:
            continue
        if len(text) <= 3 and conf < 0.35:
            continue

        # Meaningful character check
        if not re.search(r'[가-힣a-zA-Z0-9]', text):
            continue

        # Pure numeric/symbol micro-box (1-3 chars, no letters)
        if len(text) <= 3 and not re.search(r'[가-힣a-zA-Z]', text) and conf < 0.6:
            continue

        filtered.append(box)

    if not filtered:
        return []

    # Step 2: Merge horizontally adjacent boxes on same line
    filtered.sort(key=lambda b: (
        round((b["bbox"][1] + b["bbox"][3]) / 2.0, 0),  # y-center bucketed
        b["bbox"][0]  # x-start
    ))

    merged: list[dict] = []
    i = 0
    while i < len(filtered):
        current = dict(filtered[i])
        current["bbox"] = list(current["bbox"])
        j = i + 1
        while j < len(filtered):
            nxt = filtered[j]
            cur_cy = (current["bbox"][1] + current["bbox"][3]) / 2.0
            nxt_cy = (nxt["bbox"][1] + nxt["bbox"][3]) / 2.0
            cur_h = max(1.0, current["bbox"][3] - current["bbox"][1])

            # Same line check: y-centers close relative to box height
            if abs(cur_cy - nxt_cy) > cur_h * 0.6:
                break

            # Horizontal gap: small relative to box height
            h_gap = nxt["bbox"][0] - current["bbox"][2]
            if h_gap > cur_h * 1.5 or h_gap < -cur_h * 0.3:
                break

            # Merge
            merge_text = current["text"] + ("" if h_gap < cur_h * 0.3 else " ") + nxt["text"]
            merge_conf = (current["confidence"] + nxt["confidence"]) / 2.0
            current["text"] = merge_text
            current["confidence"] = round(merge_conf, 4)
            current["bbox"] = [
                min(current["bbox"][0], nxt["bbox"][0]),
                min(current["bbox"][1], nxt["bbox"][1]),
                max(current["bbox"][2], nxt["bbox"][2]),
                max(current["bbox"][3], nxt["bbox"][3]),
            ]
            j += 1

        merged.append(current)
        i = j

    return merged


# ── public API ───────────────────────────────────────────────────────

def run_ocr_on_image(
    image: np.ndarray,
    *,
    page_width: float = 0.0,
    page_height: float = 0.0,
    zoom: float = 2.0,
    multi_variant: bool = True,
) -> dict[str, Any]:
    """
    Run OCR on a numpy BGR/grayscale image.

    When multi_variant=True (default), generates multiple preprocessing
    variants, runs OCR on each, and selects the best by quality score.

    Returns
    -------
    {
      "boxes": [ { "bbox": [x0,y0,x1,y1], "text": str, "confidence": float } ],
      "text": str,
      "confidence": float,
      "success": bool,
      "engine": str | None,
      "fallback_reason": str | None,
      "error": str | None,
      # Multi-variant fields:
      "preprocess_variant": str | None,
      "variant_candidates": list[str],
      "variant_scores": dict[str, float],
      "selected_reason": str | None,
      "box_count_after_cleanup": int,
    }
    """
    fallback: dict[str, Any] = {
        "boxes": [], "text": "", "confidence": 0.0,
        "success": False, "engine": None, "fallback_reason": None, "error": None,
        "preprocess_variant": None, "variant_candidates": [],
        "variant_scores": {}, "selected_reason": None, "box_count_after_cleanup": 0,
    }

    if image is None or image.size == 0:
        fallback["error"] = "empty_image"
        return fallback

    sx = (page_width / image.shape[1]) if page_width and image.shape[1] else 1.0
    sy = (page_height / image.shape[0]) if page_height and image.shape[0] else 1.0

    if not multi_variant:
        # Legacy single-variant path
        processed = preprocess_for_ocr(image)
        result = _run_easyocr(processed, sx, sy)
        if result["success"]:
            result["boxes"] = _cleanup_ocr_boxes(result["boxes"])
            result["box_count_after_cleanup"] = len(result["boxes"])
            result["preprocess_variant"] = "adaptive_gaussian"
            result["variant_candidates"] = ["adaptive_gaussian"]
            result["variant_scores"] = {}
            result["selected_reason"] = "single_variant_mode"
            # Rebuild text
            result["text"] = "\n".join(b["text"] for b in result["boxes"])
            confs = [b["confidence"] for b in result["boxes"]]
            result["confidence"] = round(sum(confs) / len(confs), 4) if confs else 0.0
            return result

        easyocr_error = result.get("error", "unknown")
        result = _run_pytesseract(processed, sx, sy)
        if result["success"]:
            result["fallback_reason"] = f"easyocr failed: {easyocr_error}"
            result["boxes"] = _cleanup_ocr_boxes(result["boxes"])
            result["box_count_after_cleanup"] = len(result["boxes"])
            result["preprocess_variant"] = "adaptive_gaussian"
            result["variant_candidates"] = ["adaptive_gaussian"]
            result["variant_scores"] = {}
            result["selected_reason"] = "single_variant_pytesseract_fallback"
            result["text"] = "\n".join(b["text"] for b in result["boxes"])
            confs = [b["confidence"] for b in result["boxes"]]
            result["confidence"] = round(sum(confs) / len(confs), 4) if confs else 0.0
            return result

        fallback["error"] = f"easyocr: {easyocr_error}; pytesseract: {result.get('error', 'unknown')}"
        fallback["fallback_reason"] = "all_ocr_engines_failed"
        return fallback

    # ── Multi-variant path ──────────────────────────────────────────
    style = _estimate_image_style(image)
    variants = _build_ocr_preprocess_variants(image, style)

    variant_names = [name for name, _ in variants]
    variant_scores: dict[str, float] = {}
    variant_results: dict[str, dict[str, Any]] = {}

    best_name = ""
    best_score = -1.0

    for name, processed_img in variants:
        # Try EasyOCR on each variant
        ocr_res = _run_easyocr(processed_img, sx, sy)
        if not ocr_res.get("success"):
            variant_scores[name] = 0.0
            continue

        # Cleanup boxes
        cleaned = _cleanup_ocr_boxes(ocr_res["boxes"])
        score = _score_ocr_result(cleaned, page_width, page_height)
        variant_scores[name] = score

        if score > best_score:
            best_score = score
            best_name = name
            ocr_res["boxes"] = cleaned
            variant_results[name] = ocr_res

    # If no variant succeeded via EasyOCR, try pytesseract on best variant
    if not variant_results:
        for name, processed_img in variants[:2]:  # Try first 2 only
            pyt_res = _run_pytesseract(processed_img, sx, sy)
            if pyt_res.get("success"):
                cleaned = _cleanup_ocr_boxes(pyt_res["boxes"])
                score = _score_ocr_result(cleaned, page_width, page_height)
                variant_scores[name] = score
                pyt_res["boxes"] = cleaned
                pyt_res["fallback_reason"] = "easyocr_all_variants_failed"
                if score > best_score:
                    best_score = score
                    best_name = name
                    variant_results[name] = pyt_res

    if not variant_results or best_score <= 0:
        fallback["error"] = "all_variants_produced_no_text"
        fallback["variant_candidates"] = variant_names
        fallback["variant_scores"] = variant_scores
        fallback["selected_reason"] = "no_viable_result"
        return fallback

    # Assemble the winning result
    winner = variant_results[best_name]
    boxes = winner["boxes"]
    texts = [b["text"] for b in boxes]
    confs = [b["confidence"] for b in boxes]

    # Build top scores for debug (top 3)
    sorted_scores = dict(sorted(variant_scores.items(), key=lambda x: x[1], reverse=True)[:3])

    return {
        "boxes": boxes,
        "text": "\n".join(texts),
        "confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "success": True,
        "engine": winner.get("engine"),
        "fallback_reason": winner.get("fallback_reason"),
        "error": None,
        "preprocess_variant": best_name,
        "variant_candidates": variant_names,
        "variant_scores": sorted_scores,
        "selected_reason": f"best_score_{best_score:.1f}_among_{len(variant_names)}_variants",
        "box_count_after_cleanup": len(boxes),
    }


# ── OCR engine runners (unchanged core, enhanced filtering) ─────────

def _run_easyocr(image: np.ndarray, sx: float, sy: float) -> dict[str, Any]:
    reader = _get_easyocr()
    if reader is None:
        return {"success": False, "error": "EasyOCR reader unavailable", "boxes": [],
                "text": "", "confidence": 0.0, "engine": None, "fallback_reason": None}
    try:
        results: list[tuple] = reader.readtext(image, detail=1, paragraph=False)
        boxes = []
        for item in results:
            pts = item[0]
            text_val = item[1] if len(item) > 1 else ""
            conf_val = float(item[2]) if len(item) > 2 else 0.0
            if not text_val.strip() or len(text_val.strip()) < 1:
                continue

            # Pure special-character fragment skip
            if not re.search(r'[가-힣a-zA-Z0-9]', text_val):
                continue

            # Low-confidence short fragment skip
            if (conf_val < 0.5 and len(text_val.strip()) <= 5) or \
               (conf_val < 0.6 and len(text_val.strip()) <= 2):
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [
                round(min(xs) * sx, 2),
                round(min(ys) * sy, 2),
                round(max(xs) * sx, 2),
                round(max(ys) * sy, 2),
            ]
            boxes.append({"bbox": bbox, "text": text_val.strip(), "confidence": round(conf_val, 4)})

        texts = [b["text"] for b in boxes]
        confs = [b["confidence"] for b in boxes]
        avg = sum(confs) / len(confs) if confs else 0.0

        return {
            "boxes": boxes,
            "text": "\n".join(texts),
            "confidence": round(avg, 4),
            "success": True,
            "engine": "easyocr",
            "fallback_reason": None,
            "error": None,
        }
    except Exception as exc:
        logger.warning("EasyOCR execution failed: %s", exc)
        return {"success": False, "error": str(exc), "boxes": [],
                "text": "", "confidence": 0.0, "engine": None, "fallback_reason": None}


def _run_pytesseract(image: np.ndarray, sx: float, sy: float) -> dict[str, Any]:
    try:
        import pytesseract
        from PIL import Image

        pil_img = Image.fromarray(image)
        data = pytesseract.image_to_data(pil_img, lang="kor+eng", output_type=pytesseract.Output.DICT)

        boxes = []
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            conf = int(data["conf"][i]) if data["conf"][i] != -1 else 0
            if not txt or conf < 15:
                continue

            if not re.search(r'[가-힣a-zA-Z0-9]', txt):
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = [
                round(x * sx, 2),
                round(y * sy, 2),
                round((x + w) * sx, 2),
                round((y + h) * sy, 2),
            ]
            boxes.append({"bbox": bbox, "text": txt, "confidence": round(conf / 100.0, 4)})

        texts = [b["text"] for b in boxes]
        confs = [b["confidence"] for b in boxes]
        avg = sum(confs) / len(confs) if confs else 0.0

        return {
            "boxes": boxes,
            "text": "\n".join(texts),
            "confidence": round(avg, 4),
            "success": bool(boxes),
            "engine": "pytesseract",
            "fallback_reason": None,
            "error": None if boxes else "pytesseract returned no text",
        }
    except Exception as exc:
        logger.warning("pytesseract execution failed: %s", exc)
        return {"success": False, "error": str(exc), "boxes": [],
                "text": "", "confidence": 0.0, "engine": None, "fallback_reason": None}


# ── legacy compatibility shim ────────────────────────────────────────

def get_ocr_reader() -> Any:
    """Kept for backward compatibility with any external callers."""
    return _get_easyocr()


def run_easyocr_on_image(image: np.ndarray) -> dict[str, Any]:
    """Legacy wrapper — returns old-style {text, confidence, success}."""
    result = run_ocr_on_image(image, multi_variant=False)
    return {
        "text": result["text"],
        "confidence": result["confidence"],
        "success": result["success"],
        "error": result.get("error"),
    }
