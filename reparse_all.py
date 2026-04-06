"""
Batch / full reparse: runs latest-only cleanup once, then reparses every document.

Single-document parse via the API must NOT call cleanup (see app.py parse_single):
other documents' parsed_results, review JSON, and previews must stay intact.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from artifact_cleanup import cleanup_generated_artifacts
from parsers import parse_document, PARSER_VERSION, SUPPORTED_EXTENSIONS
from review_export import save_all_parse_outputs


DOC_DIR = os.path.join(current_dir, "documents")
RESULT_DIR = os.path.join(current_dir, "parsed_results")
PROJECT_ROOT = Path(current_dir).resolve()
os.makedirs(RESULT_DIR, exist_ok=True)


def _doc_id(filepath: str) -> str:
    return hashlib.md5(filepath.encode("utf-8")).hexdigest()


print("Full reparse: latest-only artifact cleanup once (batch only; not used for single Parse).")
cleanup_generated_artifacts(
    PROJECT_ROOT,
    parsed_results_dir=Path(RESULT_DIR),
)

print("Parsing documents...")
for fname in sorted(os.listdir(DOC_DIR)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        continue

    fpath = os.path.join(DOC_DIR, fname)
    did = _doc_id(fpath)
    print(f"Parsing: {fname}")
    try:
        res = parse_document(fpath)
        res["filename"] = fname
        save_all_parse_outputs(did, res, RESULT_DIR, PARSER_VERSION)
        print(" -> Success (Parsing + review JSON)")

    except Exception as e:
        print(f" -> Error: {e}")

print("Single export run from parsed_results/*.json...")
subprocess.run(
    [sys.executable, "export_to_gpt.py", "--from-parsed-cache"],
    cwd=current_dir,
)
print("Done.")
