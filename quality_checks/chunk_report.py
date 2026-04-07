"""Markdown report generation for chunk strategy quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_chunk_strategy_report(metrics: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Render a Korean Markdown report from computed chunk metrics."""
    strategy_metrics = metrics["strategy_metrics"]
    text_first = strategy_metrics.get("text_first_with_visual_support", {})
    native = strategy_metrics.get("llm_ready_native", {})
    doc_rows = metrics.get("doc_level_summary", [])

    lines = [
        "# Chunk Strategy Comparison",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- GPT model: `{manifest['models']['gpt_model']}`",
        f"- Embedding model: `{manifest['models']['embedding_model']}`",
        f"- Input chunks dir: `{manifest['input']['chunks_dir']}`",
        "",
        "## 전체 전략 비교 요약",
        "",
        _strategy_summary_sentence("text_first_with_visual_support", text_first),
        _strategy_summary_sentence("llm_ready_native", native),
        _chunk_type_sentence("text_first_with_visual_support", text_first),
        _chunk_type_sentence("llm_ready_native", native),
        "",
        "| strategy | total_chunks | total_docs | doc_coverage | page_nums | doc_pages | avg_char_len | median_char_len | avg_token_estimate | empty_text | short_chunks | long_chunks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _strategy_table_row("text_first_with_visual_support", text_first),
        _strategy_table_row("llm_ready_native", native),
        "",
        "## text_first_with_visual_support 선택 근거",
        "",
        *_selection_reasons(text_first, native),
        "",
        "## llm_ready_native 가 baseline 으로서 가지는 의미",
        "",
        "- `llm_ready_native`는 native page와 원천 블록을 더 직접적으로 반영하므로, 텍스트 우선 재구성 전략이 원본 구조 대비 얼마나 압축되거나 달라졌는지 비교하는 기준선으로 유용합니다.",
        "- baseline의 chunk 수, chunk_type 분포, 짧은 chunk 비율을 함께 보면 검색 전 단계에서 과분절 여부와 시각 요소 분리 정도를 점검할 수 있습니다.",
        "- 따라서 최종 전략으로 선택하지 않더라도 retrieval 품질 회귀를 탐지하는 비교군으로 유지할 가치가 있습니다.",
        "",
        "## 문서별 관찰 포인트",
        "",
        *_doc_observations(doc_rows),
        "",
        "## chunk 품질상 위험 요소",
        "",
        *_risk_notes(text_first, native),
        "",
        "## 다음 단계(임베딩/검색)로 넘길 때의 주의점",
        "",
        "- 임베딩 전 `empty_retrieval_text_count`와 `source_block_ids_empty_count`가 증가하는 문서는 검색 근거 추적성이 약해질 수 있으므로 별도 검수 대상입니다.",
        f"- `char_len > {metrics['thresholds']['long_chunk_char_len_gt']}` chunk는 한 벡터에 너무 많은 논점이 섞일 수 있고, `char_len < {metrics['thresholds']['short_chunk_char_len_lt']}` chunk는 단독 검색 hit의 설명력이 낮을 수 있습니다.",
        "- `has_table`, `has_chart`, `has_image` 비율은 텍스트 임베딩만으로 회수하기 어려운 시각 정보의 존재량으로 보고, 검색 결과 표시 단계에서 evidence_preview와 원본 page 링크를 함께 노출하는 편이 안전합니다.",
        "- sparse page 관련 chunk는 정보량이 낮거나 OCR/레이아웃 복구 의존도가 높을 수 있으므로, 검색 평가 쿼리에서 해당 페이지가 과도하게 상위 노출되는지 확인해야 합니다.",
        "",
    ]
    return "\n".join(lines)


def write_chunk_strategy_report(metrics: dict[str, Any], manifest: dict[str, Any], path: Path) -> None:
    path.write_text(build_chunk_strategy_report(metrics, manifest), encoding="utf-8")


def _strategy_summary_sentence(strategy: str, row: dict[str, Any]) -> str:
    return (
        f"- `{strategy}`: {row.get('total_docs', 0)}개 문서에서 {row.get('total_chunks', 0)}개 chunk를 생성했고, "
        f"평균 길이는 {row.get('avg_char_len', 0)}자, 중앙값은 {row.get('median_char_len', 0)}자입니다. "
        f"빈 retrieval_text는 {row.get('empty_retrieval_text_count', 0)}개입니다."
    )


def _strategy_table_row(strategy: str, row: dict[str, Any]) -> str:
    return (
        f"| {strategy} | {row.get('total_chunks', 0)} | {row.get('total_docs', 0)} | "
        f"{row.get('doc_coverage', 0)} | {row.get('page_coverage', {}).get('covered_page_count', 0)} | "
        f"{row.get('page_coverage', {}).get('covered_doc_page_count', 0)} | "
        f"{row.get('avg_char_len', 0)} | {row.get('median_char_len', 0)} | "
        f"{row.get('avg_token_estimate', 0)} | {row.get('empty_retrieval_text_count', 0)} | "
        f"{row.get('short_chunk_count', 0)} | {row.get('long_chunk_count', 0)} |"
    )


def _chunk_type_sentence(strategy: str, row: dict[str, Any]) -> str:
    distribution = row.get("chunk_type_distribution", {})
    rendered = ", ".join(f"{key}={value}" for key, value in distribution.items()) or "none"
    flags = row.get("content_flag_ratios", {})
    return (
        f"- `{strategy}` chunk_type 분포: {rendered}. "
        f"has_table={flags.get('has_table', 0)}, has_chart={flags.get('has_chart', 0)}, "
        f"has_image={flags.get('has_image', 0)}, has_text={flags.get('has_text', 0)}."
    )


def _selection_reasons(text_first: dict[str, Any], native: dict[str, Any]) -> list[str]:
    reasons = []
    text_short = int(text_first.get("short_chunk_count", 0))
    native_short = int(native.get("short_chunk_count", 0))
    text_empty = int(text_first.get("empty_retrieval_text_count", 0))
    native_empty = int(native.get("empty_retrieval_text_count", 0))
    text_chunks = int(text_first.get("total_chunks", 0))
    native_chunks = int(native.get("total_chunks", 0))

    if text_empty <= native_empty:
        reasons.append(
            f"- 빈 retrieval_text 수가 baseline 대비 같거나 낮습니다({text_empty} vs {native_empty}). 검색 인덱싱 누락 위험이 커지지 않는 선택입니다."
        )
    else:
        reasons.append(
            f"- 빈 retrieval_text 수는 baseline보다 높습니다({text_empty} vs {native_empty}). 선택 시 이 항목을 우선 보완해야 합니다."
        )

    if text_short <= native_short:
        reasons.append(
            f"- 너무 짧은 chunk가 baseline보다 적거나 같습니다({text_short} vs {native_short}). 단편적 hit가 줄어 검색 결과 설명력이 좋아질 가능성이 있습니다."
        )
    else:
        reasons.append(
            f"- 너무 짧은 chunk가 baseline보다 많습니다({text_short} vs {native_short}). 검색 단계에서 최소 길이 필터나 병합 검토가 필요합니다."
        )

    if text_chunks <= native_chunks:
        reasons.append(
            f"- 전체 chunk 수가 baseline보다 작거나 같습니다({text_chunks} vs {native_chunks}). 같은 문서셋에서 임베딩 비용과 검색 후보 수를 줄이는 방향입니다."
        )
    else:
        reasons.append(
            f"- 전체 chunk 수가 baseline보다 많습니다({text_chunks} vs {native_chunks}). 시각 보조 정보 보존 효과와 비용 증가를 함께 봐야 합니다."
        )

    reasons.append(
        "- chunk_type 분포와 `has_table`/`has_chart`/`has_image` 플래그가 결과 JSON에 함께 남아 있어, 텍스트 중심 chunk가 시각 근거를 잃지 않았는지 후속 검증이 가능합니다."
    )
    return reasons


def _doc_observations(doc_rows: list[dict[str, Any]]) -> list[str]:
    if not doc_rows:
        return ["- 문서별 비교 대상이 없습니다."]

    observations = []
    for row in doc_rows:
        diff = row.get("strategy_difference", {})
        strategies = row.get("strategies", {})
        text_first = strategies.get("text_first_with_visual_support", {})
        native = strategies.get("llm_ready_native", {})
        observations.append(
            f"- `{row.get('doc_id')}`: text_first chunk {text_first.get('total_chunks', 0)}개, "
            f"llm_ready_native chunk {native.get('total_chunks', 0)}개, "
            f"chunk 수 차이 {diff.get('chunk_count_delta_text_first_minus_llm_ready_native', 0)}개, "
            f"page coverage 차이 {diff.get('page_coverage_delta_text_first_minus_llm_ready_native', 0)}쪽입니다."
        )
    return observations


def _risk_notes(text_first: dict[str, Any], native: dict[str, Any]) -> list[str]:
    risks = []
    for name, row in (
        ("text_first_with_visual_support", text_first),
        ("llm_ready_native", native),
    ):
        risks.append(
            f"- `{name}`: source_block_ids가 비어 있는 chunk {row.get('source_block_ids_empty_count', 0)}개, "
            f"sparse page 관련 chunk {row.get('sparse_page_related_chunk_count', 0)}개, "
            f"긴 chunk {row.get('long_chunk_count', 0)}개, 짧은 chunk {row.get('short_chunk_count', 0)}개입니다."
        )
    risks.append(
        "- 문서별 chunk 수 표준편차가 큰 전략은 특정 문서에서 과분절 또는 과병합이 발생했을 가능성이 있으므로, doc_level_chunk_summary.json에서 차이가 큰 문서를 먼저 확인해야 합니다."
    )
    return risks
