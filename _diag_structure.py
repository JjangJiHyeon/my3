import fitz
import json

doc = fitz.open("documents/미래에셋증권 1분기 실적보고서.pdf")
page = doc[0]

try:
    raw = page.get_text("rawdict")
    with open("_diag_rawdict_p0.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print("Saved rawdict.")
except Exception as e:
    print(f"rawdict failed: {e}")

try:
    # Try getting image/xref info
    imgs = page.get_image_info(xrefs=True)
    with open("_diag_imgs_p0.json", "w", encoding="utf-8") as f:
        json.dump(imgs, f, ensure_ascii=False, indent=2)
    print("Saved image info.")
except Exception as e:
    print(f"image info failed: {e}")

doc.close()
