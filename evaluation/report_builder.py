"""Build retrieval evaluation reports from computed metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from evaluation.retrieval_eval import GPT_MODEL, STRATEGIES, load_env_file


def _winner(metrics: dict[str, Any]) -> str:
    scores = {
        strategy: metrics["strategies"][strategy]["avg_top1_score"]
        for strategy in STRATEGIES
        if strategy in metrics["strategies"]
    }
    if not scores:
        return "unknown"
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return "tie"
    return ordered[0][0]


def _count_query_wins(metrics: dict[str, Any]) -> dict[str, int]:
    counts = {strategy: 0 for strategy in STRATEGIES}
    counts["tie"] = 0
    for item in metrics.get("query_best_strategy", []):
        counts[item["best_strategy"]] = counts.get(item["best_strategy"], 0) + 1
    return counts


def maybe_build_llm_summary(project_root: Path, metrics: dict[str, Any]) -> str | None:
    load_env_file(project_root)
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    prompt = (
        "다음 retrieval 평가 지표만 근거로 한국어 보고서용 요약을 작성하세요. "
        "retrieval 자체를 수행하지 말고, 전략 비교/실패 사례/다음 단계를 간결히 정리하세요.\n\n"
        f"{json.dumps(metrics, ensure_ascii=False)[:24000]}"
    )
    try:
        client = OpenAI()
        response = client.responses.create(model=GPT_MODEL, input=prompt)
        return getattr(response, "output_text", None) or None
    except Exception:
        return None


def build_report_payload(
    *,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    run_dir: Path,
    llm_summary: str | None = None,
) -> dict[str, Any]:
    text_first = metrics["strategies"].get("text_first_with_visual_support", {})
    baseline = metrics["strategies"].get("llm_ready_native", {})
    return {
        "run_dir": str(run_dir),
        "overall_strategy_comparison": {
            "recommended_by_top1": _winner(metrics),
            "total_queries": metrics["total_queries"],
            "query_win_counts": _count_query_wins(metrics),
            "strategy_metrics": metrics["strategies"],
        },
        "why_text_first_with_visual_support": {
            "avg_top1_score": text_first.get("avg_top1_score", 0.0),
            "avg_top3_score": text_first.get("avg_top3_score", 0.0),
            "doc_coverage": text_first.get("doc_coverage", 0),
            "chunk_type_hit_distribution": text_first.get("chunk_type_hit_distribution", {}),
        },
        "baseline_difference": {
            "llm_ready_native_avg_top1_score": baseline.get("avg_top1_score", 0.0),
            "llm_ready_native_avg_top3_score": baseline.get("avg_top3_score", 0.0),
            "score_differences": metrics.get("strategy_score_differences", []),
        },
        "document_observations": metrics.get("document_average_scores", {}),
        "sparse_like_document_effect": {
            "interpretation": (
                "Use top-k hit scores, document coverage, and chunk type mix as evidence for whether "
                "visual-support text improves retrieval on sparse or slide-like pages."
            ),
            "text_first_chunk_types": text_first.get("chunk_type_hit_distribution", {}),
            "baseline_chunk_types": baseline.get("chunk_type_hit_distribution", {}),
        },
        "failure_or_ambiguous_cases": [
            item for item in metrics.get("query_best_strategy", []) if item.get("best_strategy") in {"llm_ready_native", "tie"}
        ],
        "final_recommendation": _winner(metrics),
        "next_steps": [
            "Review queries where llm_ready_native wins or ties and inspect the matching chunks.",
            "Add answer-level relevance labels if precision or recall needs to be measured beyond score comparison.",
            "Re-run after vector index updates to confirm the recommendation remains stable.",
        ],
        "llm_summary": llm_summary or "",
        "raw_metrics": metrics,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_markdown_report(payload: dict[str, Any]) -> str:
    comparison = payload["overall_strategy_comparison"]
    strategy_metrics = comparison["strategy_metrics"]
    text_first = payload["why_text_first_with_visual_support"]
    baseline = payload["baseline_difference"]
    query_wins = comparison["query_win_counts"]

    lines = [
        "# Retrieval Evaluation Report",
        "",
        "## 전체 전략 비교 요약",
        f"- total_queries: {comparison['total_queries']}",
        f"- recommended_by_top1: {comparison['recommended_by_top1']}",
        f"- query_win_counts: {json.dumps(query_wins, ensure_ascii=False)}",
        "",
    ]
    for strategy in STRATEGIES:
        data = strategy_metrics.get(strategy, {})
        lines.append(
            f"- {strategy}: avg_top1={_fmt(data.get('avg_top1_score', 0.0))}, "
            f"avg_top3={_fmt(data.get('avg_top3_score', 0.0))}, "
            f"doc_coverage={data.get('doc_coverage', 0)}, total_hits={data.get('total_hits', 0)}"
        )

    lines.extend(
        [
            "",
            "## 왜 text_first_with_visual_support 가 더 적합한지",
            f"- avg_top1_score: {_fmt(text_first['avg_top1_score'])}",
            f"- avg_top3_score: {_fmt(text_first['avg_top3_score'])}",
            f"- doc_coverage: {text_first['doc_coverage']}",
            f"- chunk_type_hit_distribution: {json.dumps(text_first['chunk_type_hit_distribution'], ensure_ascii=False)}",
            "",
            "## llm_ready_native baseline 과의 차이",
            f"- llm_ready_native_avg_top1_score: {_fmt(baseline['llm_ready_native_avg_top1_score'])}",
            f"- llm_ready_native_avg_top3_score: {_fmt(baseline['llm_ready_native_avg_top3_score'])}",
            "- query별 top1 score 차이는 report.json의 baseline_difference.score_differences에 저장했습니다.",
            "",
            "## 문서별 관찰",
        ]
    )
    for doc_id, data in payload["document_observations"].items():
        lines.append(
            f"- doc_id={doc_id}, avg_score={_fmt(data['avg_score'])}, "
            f"hit_count={data['hit_count']}, filename={data.get('filename', '')}"
        )

    lines.extend(
        [
            "",
            "## sparse/sparse-like 문서에서의 보완 효과",
            "- sparse-like 효과는 text_first 전략의 hit score, doc coverage, chunk type mix가 baseline과 어떻게 다른지로 확인합니다.",
            f"- text_first_chunk_types: {json.dumps(payload['sparse_like_document_effect']['text_first_chunk_types'], ensure_ascii=False)}",
            f"- baseline_chunk_types: {json.dumps(payload['sparse_like_document_effect']['baseline_chunk_types'], ensure_ascii=False)}",
            "",
            "## 실패/애매 사례",
        ]
    )
    ambiguous = payload["failure_or_ambiguous_cases"]
    if ambiguous:
        for item in ambiguous:
            lines.append(
                f"- {item['query_id']}: best_strategy={item['best_strategy']}, "
                f"top1_scores={json.dumps(item['top1_scores'], ensure_ascii=False)}"
            )
    else:
        lines.append("- top1 기준으로 baseline 우세 또는 동률인 query가 없습니다.")

    lines.extend(
        [
            "",
            "## 최종 추천 전략",
            f"- {payload['final_recommendation']}",
            "",
            "## 다음 단계 제안",
        ]
    )
    lines.extend(f"- {item}" for item in payload["next_steps"])

    if payload.get("llm_summary"):
        lines.extend(["", "## GPT-5.2 보고서 요약", payload["llm_summary"]])

    return "\n".join(lines).rstrip() + "\n"


def write_reports(project_root: Path, run_dir: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    llm_summary = maybe_build_llm_summary(project_root, metrics)
    payload = build_report_payload(metrics=metrics, rows=rows, run_dir=run_dir, llm_summary=llm_summary)
    (run_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(build_markdown_report(payload), encoding="utf-8")
