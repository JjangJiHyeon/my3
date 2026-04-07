import fitz
doc = fitz.open("documents/미래에셋증권 4분기 실적보고서.pdf")
page = doc[4] # Page 5
print(f"Page 5 text (first 100): {page.get_text('text')[:100]}")
doc.close()
