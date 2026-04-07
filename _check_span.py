import fitz
doc = fitz.open("documents/미래에셋증권 1분기 실적보고서.pdf")
page = doc[11]
raw = page.get_text("rawdict")
for block in raw.get("blocks", []):
    if block.get("type") == 0:
        line = block.get("lines", [])[0]
        span = line.get("spans", [])[0]
        print(f"Span keys: {list(span.keys())}")
        if "chars" in span:
            print(f"First character: {span['chars'][0]}")
        break
doc.close()
