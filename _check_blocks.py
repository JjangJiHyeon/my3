import fitz
doc = fitz.open("documents/미래에셋증권 1분기 실적보고서.pdf")
page = doc[11]
raw = page.get_text("rawdict")
for i, b in enumerate(raw.get("blocks", [])[:5]):
    print(f"Block {i} type: {b.get('type')}, keys: {list(b.keys())}")
doc.close()
