import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import parsed_cache, DOC_DIR, RESULT_DIR, _doc_id, _save_result
from app_support.artifact_cleanup import cleanup_generated_artifacts
from parsers import parse_document, SUPPORTED_EXTENSIONS
from parsers.pdf_parser import set_result_dir

set_result_dir(RESULT_DIR)

print("Batch/full refresh: cleanup_generated_artifacts (same policy as app parse-all)...")
cleanup_generated_artifacts(PROJECT_ROOT, parsed_results_dir=Path(RESULT_DIR))
parsed_cache.clear()

results = []
print("Starting parse-all...")
for fname in sorted(os.listdir(DOC_DIR)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        continue
    fpath = os.path.join(DOC_DIR, fname)
    if not os.path.isfile(fpath):
        continue
        
    did = _doc_id(fpath)
    print(f"Parsing: {fname}")
    try:
        result = parse_document(fpath)
    except Exception as exc:
        result = {
            "id": did,
            "filename": fname,
            "status": "error",
            "error": str(exc),
            "pages": [],
            "metadata": {},
        }
    parsed_cache[did] = result
    _save_result(did, result)
    results.append(result)

print(f"Parsed {len(results)} documents.")

for res in results:
    if res['status'] == 'success' and res.get('pages'):
        print('\n--- SUCCESS PAGE JSON ---')
        p = dict(res['pages'][0])
        p['blocks'] = [p['blocks'][0]] if p.get('blocks') else []
        print(json.dumps(p, indent=2))
        break

print('\n--- SIMULATED ERROR PAGE JSON ---')
from parsers.pdf_parser import _process_page, parse_pdf
import fitz
# Mocking a failed doc[idx] scenario would create the exception we wrote in parse_pdf
# Let's just create a dummy one that parse_pdf outputs natively if we pass an invalid path? No, invalid path raises FileNotFoundError
# We can just manually trigger the catch block by patching a method dynamically
doc = fitz.open()
doc.insert_page(0) # page 0

def mock_process_page(*args, **kwargs):
    raise RuntimeError("Intentional UI Error Test")

import parsers.pdf_parser
parsers.pdf_parser._process_page = mock_process_page

pdf_path = os.path.join(RESULT_DIR, 'dummy.pdf')
doc.save(pdf_path)

try:
    err_res = parsers.pdf_parser.parse_pdf(pdf_path)
    print(json.dumps(err_res['pages'][0], indent=2))
finally:
    doc.close()
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

print("Done")
