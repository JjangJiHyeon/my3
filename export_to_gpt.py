"""
Export parsed document results as structured files for LLM/RAG consumption.

Usage:
  # Ad-hoc mode: parse file(s) and export
  python export_to_gpt.py documents/sample.pdf documents/sample.hwp

  # By doc_id: export already-parsed result from parsed_results/
  python export_to_gpt.py --id 0e989092c3d389b5e7727c1a077e66c5

  # All documents in documents/ folder
  python export_to_gpt.py --all

  # One export run from existing parsed_results/*.json (no re-parse)
  python export_to_gpt.py --from-parsed-cache

Outputs per document (under exports/<run_id>/):
  <filename>_raw.json        - Full parse result
  <filename>_llm_ready.json  - Lightweight: rag_text, block summaries, table markdown
  <filename>_llm_report.md   - Human-readable structured markdown
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from parsers import parse_document, SUPPORTED_EXTENSIONS, PARSER_VERSION


def _make_export_dir(run_id: str) -> Path:
    d = REPO_ROOT / "exports" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_llm_ready(result: dict[str, Any]) -> dict[str, Any]:
    meta = result.get("metadata", {})
    pages = result.get("pages", [])
    llm_pages = []
    for page in pages:
        blocks = page.get("blocks", [])
        block_summary = []
        for b in blocks:
            entry: dict[str, Any] = {
                "type": b.get("type"),
                "text": (b.get("text") or "")[:500],
            }
            table_md = b.get("meta", {}).get("table_markdown")
            if table_md:
                entry["table_markdown"] = table_md
            rows = b.get("meta", {}).get("rows")
            if rows and not table_md:
                entry["table_rows"] = rows[:20]
            block_summary.append(entry)

        llm_pages.append({
            "page_num": page.get("page_num"),
            "rag_text": page.get("rag_text", ""),
            "rag_text_length": len(page.get("rag_text", "")),
            "block_count": len(blocks),
            "block_type_counts": page.get("parser_debug", {}).get("block_type_counts", {}),
            "blocks": block_summary,
        })

    return {
        "filename": result.get("filename"),
        "status": result.get("status"),
        "parser_version": result.get("parser_version", PARSER_VERSION),
        "document_type": meta.get("document_type"),
        "refined_document_type": meta.get("refined_document_type"),
        "pipeline_used": meta.get("pipeline_used"),
        "parser_used": meta.get("parser_used"),
        "page_count": len(pages),
        "total_chars": meta.get("total_chars", 0),
        "quality_score": meta.get("quality_score"),
        "quality_grade": meta.get("quality_grade"),
        "pages": llm_pages,
    }


def _generate_md_report(result: dict[str, Any]) -> str:
    meta = result.get("metadata", {})
    pages = result.get("pages", [])
    lines = [
        f"# {result.get('filename', 'Unknown')}",
        "",
        f"- **Status:** {result.get('status')}",
        f"- **Parser Version:** {result.get('parser_version', 'N/A')}",
        f"- **Document Type:** {meta.get('document_type', 'N/A')}",
        f"- **Refined Type:** {meta.get('refined_document_type', 'N/A')}",
        f"- **Pipeline:** {meta.get('pipeline_used', 'N/A')}",
        f"- **Pages:** {len(pages)}",
        f"- **Total Chars:** {meta.get('total_chars', 0)}",
        f"- **Quality:** {meta.get('quality_grade', 'N/A')} ({meta.get('quality_score', 'N/A')})",
        "",
        "---",
        "",
    ]

    for page in pages:
        pnum = page.get("page_num", "?")
        blocks = page.get("blocks", [])
        rag = page.get("rag_text", "")
        debug = page.get("parser_debug", {})
        type_counts = debug.get("block_type_counts", {})

        lines.append(f"## Page {pnum}")
        lines.append("")
        lines.append(f"**Blocks:** {len(blocks)} | "
                      f"title={type_counts.get('title', 0)}, "
                      f"text={type_counts.get('text', 0)}, "
                      f"table={type_counts.get('table', 0)}, "
                      f"image={type_counts.get('image', 0)}, "
                      f"footer={type_counts.get('footer', 0)}")
        lines.append("")

        if rag:
            lines.append("### RAG Text")
            lines.append("")
            lines.append("```")
            lines.append(rag[:2000])
            if len(rag) > 2000:
                lines.append(f"... ({len(rag)} chars total)")
            lines.append("```")
            lines.append("")
        else:
            lines.append("**RAG Text:** *(empty)*")
            lines.append("")

        table_blocks = [b for b in blocks if b.get("type") == "table"]
        if table_blocks:
            lines.append("### Tables")
            lines.append("")
            for i, tb in enumerate(table_blocks[:5]):
                md = tb.get("meta", {}).get("table_markdown", "")
                text = (tb.get("text") or "")[:300]
                lines.append(f"**Table {i+1}:** {tb.get('meta', {}).get('table_shape', {})}")
                if md:
                    lines.append("")
                    lines.append(md[:500])
                elif text:
                    lines.append("")
                    lines.append(f"```\n{text}\n```")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_one(filepath_or_result: str | dict, export_dir: Path) -> dict[str, str]:
    if isinstance(filepath_or_result, dict):
        result = filepath_or_result
    else:
        print(f"  Parsing: {filepath_or_result}")
        result = parse_document(filepath_or_result)

    filename = result.get("filename", "unknown")
    stem = Path(filename).stem

    raw_path = export_dir / f"{stem}_raw.json"
    llm_path = export_dir / f"{stem}_llm_ready.json"
    md_path = export_dir / f"{stem}_llm_report.md"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    llm_ready = _extract_llm_ready(result)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_ready, f, ensure_ascii=False, indent=2)

    md_report = _generate_md_report(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"  Exported: {stem}")
    print(f"    raw:       {raw_path.name}")
    print(f"    llm_ready: {llm_path.name}")
    print(f"    md_report: {md_path.name}")

    return {"raw": str(raw_path), "llm_ready": str(llm_path), "md_report": str(md_path)}


def write_root_llm_exports(
    result: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Write {stem}_llm_ready.json and {stem}_llm_report.md at repo root (latest-only sidecars)."""
    root = (repo_root or REPO_ROOT).resolve()
    stem = Path(str(result.get("filename") or "export")).stem
    llm_path = root / f"{stem}_llm_ready.json"
    md_path = root / f"{stem}_llm_report.md"

    llm_ready = _extract_llm_ready(result)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_ready, f, ensure_ascii=False, indent=2)

    md_report = _generate_md_report(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    return {"llm_ready": str(llm_path), "md_report": str(md_path)}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    if "--from-parsed-cache" in args:
        pr = REPO_ROOT / "parsed_results"
        run_id = f"run_{int(time.time())}"
        export_dir = _make_export_dir(run_id)
        print(f"Export run: {run_id}")
        print(f"Output dir: {export_dir}")
        print()
        n_ok = 0
        for jf in sorted(pr.glob("*.json")):
            if not jf.is_file():
                continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except Exception as exc:
                print(f"  SKIP {jf.name}: {exc}")
                continue
            try:
                export_one(result, export_dir)
                n_ok += 1
            except Exception as exc:
                print(f"  ERROR exporting {jf.name}: {exc}")
        print(f"\nDone. {n_ok} document(s) exported to {export_dir}")
        return

    run_id = f"run_{int(time.time())}"
    export_dir = _make_export_dir(run_id)
    print(f"Export run: {run_id}")
    print(f"Output dir: {export_dir}")
    print()

    if "--all" in args:
        doc_dir = REPO_ROOT / "documents"
        if not doc_dir.is_dir():
            print(f"ERROR: documents/ directory not found at {doc_dir}")
            raise SystemExit(1)
        files = sorted(
            str(doc_dir / f)
            for f in os.listdir(doc_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print("No supported documents found.")
            raise SystemExit(1)
        for fp in files:
            try:
                export_one(fp, export_dir)
            except Exception as exc:
                print(f"  ERROR parsing {fp}: {exc}")
        print(f"\nDone. {len(files)} documents exported to {export_dir}")

    elif "--id" in args:
        idx = args.index("--id")
        if idx + 1 >= len(args):
            print("ERROR: --id requires a doc_id argument")
            raise SystemExit(1)
        doc_id = args[idx + 1]
        result_path = REPO_ROOT / "parsed_results" / f"{doc_id}.json"
        if not result_path.exists():
            print(f"ERROR: No cached result for doc_id={doc_id}")
            raise SystemExit(1)
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        export_one(result, export_dir)
        print(f"\nDone. Exported to {export_dir}")

    else:
        for fp in args:
            if not os.path.isfile(fp):
                print(f"  SKIP: {fp} (not found)")
                continue
            try:
                export_one(fp, export_dir)
            except Exception as exc:
                print(f"  ERROR parsing {fp}: {exc}")
        print(f"\nDone. Exported to {export_dir}")


if __name__ == "__main__":
    main()
