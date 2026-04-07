import fitz
doc = fitz.open("documents/미래에셋증권 1분기 실적보고서.pdf")
page = doc[0]
raw = page.get_text("rawdict")
print(f"Total blocks in rawdict: {len(raw.get('blocks', []))}")
text_blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]
print(f"Text blocks count: {len(text_blocks)}")
for i, b in enumerate(text_blocks):
    for l in b.get("lines", []):
        for s in l.get("spans", []):
            chars = s.get("chars", [])
            txt = "".join(c.get("c", "") for c in chars)
            print(f"  Block {i} Span text: '{txt}'")
doc.close()
