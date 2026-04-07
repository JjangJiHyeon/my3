import fitz
from parsers.pdf.ppt_export_native import recover_ppt_export_blocks

doc = fitz.open("documents/미래에셋증권 1분기 실적보고서.pdf")
page = doc[11] # Page 12
pw, ph = page.rect.width, page.rect.height

res = recover_ppt_export_blocks(page, 11, pw, ph)
print(f"Success: {res.get('success')}")
if not res.get('success'):
    print(f"Failure reason: {res.get('failure_reason')}")
print(f"Recovered text len: {res.get('recovered_text_len')}")
print(f"Block count: {len(res.get('blocks', []))}")
print(f"Debug: {res.get('debug')}")

if res.get('blocks'):
    print("Top blocks text:")
    for b in res['blocks'][:3]:
        print(f"--- {b['type']} ---\n{b['text']}")

doc.close()
