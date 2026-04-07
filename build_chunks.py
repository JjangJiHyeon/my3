from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from chunking.builders import build_all_for_document
from chunking.io import load_llm_ready, load_parsed_results, load_review_results, read_json, write_json
from chunking.strategies import BASELINE_STRATEGY, MAIN_STRATEGY

PROJECT_ROOT = Path(__file__).resolve().parent
PARSED_DIR = PROJECT_ROOT / "parsed_results"
REVIEW_DIR = PARSED_DIR / "review"
CHUNK_DIR = PROJECT_ROOT / "chunks"


def _doc_id_for_file(path: Path) -> str:
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()


def _select_docs(args: argparse.Namespace, parsed_docs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if args.doc_id:
        doc = parsed_docs.get(args.doc_id)
        if not doc:
            raise SystemExit(f"doc-id not found in parsed_results: {args.doc_id}")
        return {args.doc_id: doc}
    if args.file:
        path = Path(args.file)
        doc_id = _doc_id_for_file(path)
        doc = parsed_docs.get(doc_id)
        if not doc:
            matches = {
                did: data for did, data in parsed_docs.items()
                if Path(str(data.get("filepath") or "")).name == path.name or str(data.get("filename") or "") == path.name
            }
            if matches:
                return matches
            raise SystemExit(f"parsed cache not found for file: {path}")
        return {doc_id: doc}
    if args.from_cache:
        return parsed_docs
    raise SystemExit("Use --from-cache, --doc-id <id>, or --file <path>.")


def _save_outputs(doc_id: str, outputs: dict[str, dict[str, Any]], out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for strategy in (MAIN_STRATEGY, BASELINE_STRATEGY):
        path = out_dir / f"{doc_id}.{strategy}.json"
        write_json(path, outputs[strategy])
        written.append(path)
    summary_path = out_dir / f"{doc_id}.chunk_summary.json"
    write_json(summary_path, outputs["summary"])
    written.append(summary_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reusable chunk JSON from cached parse results.")
    parser.add_argument("--from-cache", action="store_true", help="Build chunks for every parsed_results/*.json document.")
    parser.add_argument("--doc-id", help="Build chunks for one parsed_results/{doc_id}.json document.")
    parser.add_argument("--file", help="Build chunks for a document already present in parsed cache.")
    parser.add_argument("--out-dir", default=str(CHUNK_DIR), help="Chunk output directory.")
    args = parser.parse_args()

    parsed_docs = load_parsed_results(PARSED_DIR)
    review_docs = load_review_results(REVIEW_DIR)
    llm_ready_by_filename = load_llm_ready(PROJECT_ROOT)
    selected = _select_docs(args, parsed_docs)
    out_dir = Path(args.out_dir)

    print(f"Loaded parsed={len(parsed_docs)} review={len(review_docs)} llm_ready={len(llm_ready_by_filename)}")
    print(f"Building chunks for {len(selected)} document(s) into {out_dir}")

    failures = 0
    for doc_id, parsed_doc in selected.items():
        try:
            filename = str(parsed_doc.get("filename") or "")
            llm_ready = llm_ready_by_filename.get(filename)
            if not llm_ready:
                review_doc = review_docs.get(doc_id)
                if review_doc and review_doc.get("filename"):
                    llm_ready = llm_ready_by_filename.get(str(review_doc.get("filename")))
            outputs = build_all_for_document(parsed_doc=parsed_doc, llm_ready=llm_ready, doc_id=doc_id)
            written = _save_outputs(doc_id, outputs, out_dir)
            counts = {item["strategy_name"]: item["total_chunks"] for item in outputs["summary"]["strategies"]}
            print(f"- {doc_id} {filename}: {counts} -> {', '.join(path.name for path in written)}")
        except Exception as exc:
            failures += 1
            error_payload = {
                "doc_id": doc_id,
                "filename": parsed_doc.get("filename"),
                "error": str(exc),
                "strategy_name": "chunk_build_error",
                "chunks": [],
            }
            write_json(out_dir / f"{doc_id}.chunk_error.json", error_payload)
            print(f"- ERROR {doc_id}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
