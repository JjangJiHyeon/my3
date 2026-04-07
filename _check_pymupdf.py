import fitz
doc = fitz.open("documents/미래에셋증권 1분기 실적보고서.pdf")
page = doc[11] # Page 12
print(f"Page text length: {len(page.get_text('text').strip())}")
print(f"Page dict blocks: {len(page.get_text('dict').get('blocks', []))}")
print(f"Page rawdict blocks: {len(page.get_text('rawdict').get('blocks', []))}")
doc.close()
