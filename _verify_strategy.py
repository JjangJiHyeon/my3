import os
import sys
import json
from parsers.pdf_parser import parse_pdf

def verify_parsing():
    test_files = [
        "documents/미래에셋증권 1분기 실적보고서.pdf",
        "documents/미래에셋증권 4분기 실적보고서.pdf"
    ]
    
    results = {}
    for fpath in test_files:
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue
            
        print(f"\nProcessing {fpath}...")
        try:
            res = parse_pdf(fpath)
            meta = res.get("metadata", {})
            print(f"Document Type: {meta.get('document_type')} (Pipeline: {meta.get('pipeline_used')})")
            print(f"Extraction Source Stats: {meta.get('extraction_source_stats')}")
            
            # Check a few pages
            for i, p in enumerate(res.get("pages", [])):
                dbg = p.get("parser_debug", {})
                source = dbg.get("chosen_text_source") or dbg.get("report_text_source")
                skip_ocr = dbg.get("ocr_skipped_due_to_native_recovery")
                print(f"  Page {i+1}: source={source}, skip_ocr={skip_ocr}, text_len={len(p.get('text', ''))}")
                
        except Exception as e:
            print(f"Error parsing {fpath}: {e}")

if __name__ == "__main__":
    verify_parsing()
