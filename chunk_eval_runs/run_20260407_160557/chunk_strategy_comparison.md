# Chunk Strategy Comparison

- Run ID: `run_20260407_160557`
- GPT model: `gpt-5.2`
- Embedding model: `text-embedding-3-large`
- Input chunks dir: `C:\Users\jihyeon\Desktop\my3\chunks`

## 전체 전략 비교 요약

- `text_first_with_visual_support`: 8개 문서에서 190개 chunk를 생성했고, 평균 길이는 233.0632자, 중앙값은 88.0자입니다. 빈 retrieval_text는 0개입니다.
- `llm_ready_native`: 8개 문서에서 865개 chunk를 생성했고, 평균 길이는 150.9676자, 중앙값은 53.0자입니다. 빈 retrieval_text는 0개입니다.
- `text_first_with_visual_support` chunk_type 분포: block_group=101, text=89. has_table=0.1526, has_chart=0.0684, has_image=0.1, has_text=1.0.
- `llm_ready_native` chunk_type 분포: chart=13, image=22, native_page=108, table=148, text=513, title=59, unknown=2. has_table=0.1711, has_chart=0.015, has_image=0.0254, has_text=1.0.

| strategy | total_chunks | total_docs | doc_coverage | page_nums | doc_pages | avg_char_len | median_char_len | avg_token_estimate | empty_text | short_chunks | long_chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_first_with_visual_support | 190 | 8 | 1.0 | 19 | 78 | 233.0632 | 88.0 | 57.8947 | 0 | 113 | 0 |
| llm_ready_native | 865 | 8 | 1.0 | 19 | 78 | 150.9676 | 53.0 | 37.3503 | 0 | 677 | 0 |

## text_first_with_visual_support 선택 근거

- 빈 retrieval_text 수가 baseline 대비 같거나 낮습니다(0 vs 0). 검색 인덱싱 누락 위험이 커지지 않는 선택입니다.
- 너무 짧은 chunk가 baseline보다 적거나 같습니다(113 vs 677). 단편적 hit가 줄어 검색 결과 설명력이 좋아질 가능성이 있습니다.
- 전체 chunk 수가 baseline보다 작거나 같습니다(190 vs 865). 같은 문서셋에서 임베딩 비용과 검색 후보 수를 줄이는 방향입니다.
- chunk_type 분포와 `has_table`/`has_chart`/`has_image` 플래그가 결과 JSON에 함께 남아 있어, 텍스트 중심 chunk가 시각 근거를 잃지 않았는지 후속 검증이 가능합니다.

## llm_ready_native 가 baseline 으로서 가지는 의미

- `llm_ready_native`는 native page와 원천 블록을 더 직접적으로 반영하므로, 텍스트 우선 재구성 전략이 원본 구조 대비 얼마나 압축되거나 달라졌는지 비교하는 기준선으로 유용합니다.
- baseline의 chunk 수, chunk_type 분포, 짧은 chunk 비율을 함께 보면 검색 전 단계에서 과분절 여부와 시각 요소 분리 정도를 점검할 수 있습니다.
- 따라서 최종 전략으로 선택하지 않더라도 retrieval 품질 회귀를 탐지하는 비교군으로 유지할 가치가 있습니다.

## 문서별 관찰 포인트

- `1483963f1ac161b67f7ff3efbb1432a0`: text_first chunk 29개, llm_ready_native chunk 63개, chunk 수 차이 -34개, page coverage 차이 0쪽입니다.
- `2072f5b049ff3c984e7d1c1a888c1f18`: text_first chunk 31개, llm_ready_native chunk 66개, chunk 수 차이 -35개, page coverage 차이 0쪽입니다.
- `78e892d65ab82f707075563d9fcfe497`: text_first chunk 61개, llm_ready_native chunk 261개, chunk 수 차이 -200개, page coverage 차이 0쪽입니다.
- `7de041057f9dcb58d0a189c04787527a`: text_first chunk 15개, llm_ready_native chunk 288개, chunk 수 차이 -273개, page coverage 차이 0쪽입니다.
- `8b2aad7a88b01e817295ed856e51bc6a`: text_first chunk 18개, llm_ready_native chunk 73개, chunk 수 차이 -55개, page coverage 차이 0쪽입니다.
- `9454b2bef0a064ff78f0cb719036c8f8`: text_first chunk 5개, llm_ready_native chunk 27개, chunk 수 차이 -22개, page coverage 차이 0쪽입니다.
- `ae827bcde785199b8d6bbbc1cc8ff287`: text_first chunk 1개, llm_ready_native chunk 22개, chunk 수 차이 -21개, page coverage 차이 0쪽입니다.
- `f76e5bb34c4e5dd7afeeccdf4553a1f8`: text_first chunk 30개, llm_ready_native chunk 65개, chunk 수 차이 -35개, page coverage 차이 0쪽입니다.

## chunk 품질상 위험 요소

- `text_first_with_visual_support`: source_block_ids가 비어 있는 chunk 0개, sparse page 관련 chunk 58개, 긴 chunk 0개, 짧은 chunk 113개입니다.
- `llm_ready_native`: source_block_ids가 비어 있는 chunk 0개, sparse page 관련 chunk 28개, 긴 chunk 0개, 짧은 chunk 677개입니다.
- 문서별 chunk 수 표준편차가 큰 전략은 특정 문서에서 과분절 또는 과병합이 발생했을 가능성이 있으므로, doc_level_chunk_summary.json에서 차이가 큰 문서를 먼저 확인해야 합니다.

## 다음 단계(임베딩/검색)로 넘길 때의 주의점

- 임베딩 전 `empty_retrieval_text_count`와 `source_block_ids_empty_count`가 증가하는 문서는 검색 근거 추적성이 약해질 수 있으므로 별도 검수 대상입니다.
- `char_len > 1500` chunk는 한 벡터에 너무 많은 논점이 섞일 수 있고, `char_len < 120` chunk는 단독 검색 hit의 설명력이 낮을 수 있습니다.
- `has_table`, `has_chart`, `has_image` 비율은 텍스트 임베딩만으로 회수하기 어려운 시각 정보의 존재량으로 보고, 검색 결과 표시 단계에서 evidence_preview와 원본 page 링크를 함께 노출하는 편이 안전합니다.
- sparse page 관련 chunk는 정보량이 낮거나 OCR/레이아웃 복구 의존도가 높을 수 있으므로, 검색 평가 쿼리에서 해당 페이지가 과도하게 상위 노출되는지 확인해야 합니다.
