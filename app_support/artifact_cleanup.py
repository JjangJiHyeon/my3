"""
Delete generated artifacts only under the project root.

Never touches documents/, parsers/, or source code. Paths are resolved and
checked before any removal.
"""

from __future__ import annotations

import fnmatch
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_PROTECTED_NAMES = frozenset({"documents", "parsers", ".git"})

# Repo-root sidecars and reports (same basenames as .gitignore hints).
_ROOT_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "*_llm_ready.json",
    "*_llm_report.md",
    "quality_report.json",
    "quality_report.md",
    "harness_history.json",
)

# Known generated directories at repo root (not documents/parsers).
_ARTIFACT_DIR_NAMES: tuple[str, ...] = (
    "exports",
    "harness_runs",
    "tests_run",
    "temp_verify_results",
    "results",
)


def _strictly_inside(root: Path, target: Path) -> bool:
    """True if target is a path strictly under root (not equal to root)."""
    root_r = root.resolve()
    target_r = target.resolve()
    return root_r in target_r.parents


def _safe_remove_tree(path: Path, root: Path) -> None:
    root_r = root.resolve()
    p = path.resolve()
    if p.name in _PROTECTED_NAMES:
        return
    if not _strictly_inside(root_r, p):
        logger.warning("Refusing to remove tree outside project root: %s", p)
        return
    if not p.is_dir():
        return
    try:
        shutil.rmtree(p)
        logger.info("Removed directory %s", p)
    except OSError as exc:
        logger.warning("Could not remove directory %s: %s", p, exc)


def cleanup_generated_artifacts(
    project_root: Path | str,
    *,
    parsed_results_dir: Path | str,
) -> None:
    """
    Remove cached parse outputs and other generated files under ``project_root`` only.

    - Clears ``parsed_results_dir`` (JSON, review/, previews/) and recreates an empty dir.
    - Removes ``exports/``, ``harness_runs/``, ``tests_run/``, etc. when present.
    - Removes root-level ``*_llm_ready.json``, ``*_llm_report.md``, ``quality_report.json``, etc.
    """
    root = Path(project_root).resolve()
    pr = Path(parsed_results_dir).resolve()

    if pr == root:
        logger.warning("parsed_results_dir equals project root; refusing cleanup")
        return
    if not _strictly_inside(root, pr):
        logger.warning(
            "parsed_results_dir %s is not inside project root %s; skipping cleanup",
            pr,
            root,
        )
        return

    if pr.exists():
        _safe_remove_tree(pr, root)
    try:
        pr.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not recreate parsed_results dir %s: %s", pr, exc)

    for dirname in _ARTIFACT_DIR_NAMES:
        _safe_remove_tree((root / dirname), root)

    root_r = root.resolve()
    for entry in list(root.iterdir()):
        if not entry.is_file():
            continue
        try:
            if entry.resolve().parent != root_r:
                continue
        except OSError:
            continue
        name = entry.name
        if name.endswith(".py"):
            continue
        if not any(fnmatch.fnmatch(name, pat) for pat in _ROOT_ARTIFACT_PATTERNS):
            continue
        try:
            entry.unlink()
            logger.info("Removed file %s", entry)
        except OSError as exc:
            logger.warning("Could not remove file %s: %s", entry, exc)
