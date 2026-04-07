import fitz
from parsers.document_router import _page_feature_bundle

doc = fitz.open("documents/미래에셋증권 4분기 실적보고서.pdf")
page = doc[4] # Page 5
# To use _page_feature_bundle, I need to pass it a dict like the ones in pdf_parser's _finalize_page_results.
# Actually, pdf_parser calls it with a simplified page dict.
# Let's just manually compute the density.
text = page.get_text("text")
compact = text.replace(" ", "").replace("\n", "")
print(f"4Q P5 Text density: {len(compact)}")
doc.close()
