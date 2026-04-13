"""Minimal Chroma retriever wrapper."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import logging
import re
from threading import Lock
from typing import Any

from .config import RagApiConfig, load_latest_chroma_index, resolve_chroma_persist_dir, resolve_collection_name
from .hwp_viewer_page_mapper import enrich_hwp_documents_with_viewer_pages, preferred_page_num
from .keyword_retriever import KeywordRetriever
from .observability import optional_traceable, stage_timer
from .qa_ranker import is_qa_boilerplate_document, rerank_qa_documents
from .summary_ranker import rerank_summary_documents

logger = logging.getLogger(__name__)


SUMMARY_MIN_COVERAGE_CHUNKS = 24
SUMMARY_MAX_COVERAGE_CHUNKS = 60
SUMMARY_COVERAGE_CACHE_SIZE = 64
SUMMARY_SELECTION_CACHE_SIZE = 128

_summary_cache_lock = Lock()
_summary_coverage_cache: OrderedDict[tuple[str, str, str, str], tuple[dict[str, Any], ...]] = OrderedDict()
_summary_selection_cache: OrderedDict[tuple[str, str, str, str, int], tuple[str, ...]] = OrderedDict()


@dataclass
class ChromaRetriever:
    config: RagApiConfig

    @optional_traceable(name="retriever.retrieve", run_type="retriever")
    def retrieve(
        self,
        query: str,
        strategy_name: str,
        top_k: int,
        filename_filter: str | None = None,
        doc_id_filter: str | None = None,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.retrieve",
            request_id=request_id,
            strategy_name=strategy_name,
            top_k=top_k,
            filename_filter=filename_filter,
            doc_id_filter=doc_id_filter,
            query_len=len(query),
        ) as event:
            candidate_top_k = max(int(top_k), 1)
            vector_candidate_k = max(candidate_top_k, int(self.config.qa_vector_candidate_k))
            keyword_candidate_k = max(candidate_top_k, int(self.config.qa_keyword_candidate_k))
            fused_candidate_k = max(candidate_top_k, int(self.config.qa_fused_candidate_k))
            final_top_k = min(candidate_top_k, int(self.config.qa_final_top_k))

            vector_documents = self._retrieve_qa_vector_candidates(
                query=query,
                top_k=vector_candidate_k,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
                doc_id_filter=doc_id_filter,
                request_id=request_id,
            )
            keyword_documents = KeywordRetriever(self.config).retrieve(
                query=query,
                top_k=keyword_candidate_k,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
                doc_id_filter=doc_id_filter,
                request_id=request_id,
            )
            fused_documents = _fuse_rrf_documents(
                vector_documents=vector_documents,
                keyword_documents=keyword_documents,
                limit=fused_candidate_k,
                rrf_k=int(self.config.qa_rrf_k),
            )
            documents = rerank_qa_documents(
                query=query,
                documents=fused_documents,
                top_k=final_top_k,
            )
            enrich_hwp_documents_with_viewer_pages(
                self.config.project_root,
                vector_documents + keyword_documents + fused_documents + documents,
            )

            expected_page = _expected_page_from_query(query)
            event["backend"] = "qa_hybrid_rrf_rerank"
            event["vector_candidates"] = len(vector_documents)
            event["keyword_candidates"] = len(keyword_documents)
            event["fused_candidates"] = len(fused_documents)
            event["returned_docs"] = len(documents)
            event["vector_top5"] = _serialize_documents(vector_documents, limit=5)
            event["keyword_top5"] = _serialize_documents(keyword_documents, limit=5)
            event["fused_top5"] = _serialize_documents(fused_documents, limit=5)
            event["reranked_top5"] = _serialize_documents(documents, limit=5)
            event["expected_document_hit_candidates"] = _expected_document_hit(fused_documents, filename_filter, doc_id_filter)
            event["expected_document_hit_top5"] = _expected_document_hit(documents, filename_filter, doc_id_filter)
            event["expected_page_hit_candidates"] = _expected_page_hit(fused_documents, expected_page)
            event["expected_page_hit_top5"] = _expected_page_hit(documents, expected_page)
            event["boilerplate_candidates"] = _boilerplate_count(fused_documents)
            event["boilerplate_top5"] = _boilerplate_count(documents, limit=5)
            logger.info(
                "QA hybrid retrieval request_id=%s vector=%s keyword=%s fused=%s final=%s",
                request_id,
                len(vector_documents),
                len(keyword_documents),
                len(fused_documents),
                len(documents),
            )
            return documents

    @optional_traceable(name="retriever.qa_vector_candidates", run_type="retriever")
    def _retrieve_qa_vector_candidates(
        self,
        query: str,
        top_k: int,
        strategy_name: str,
        filename_filter: str | None,
        doc_id_filter: str | None = None,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.qa_vector_candidates",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
            doc_id_filter=doc_id_filter,
            top_k=top_k,
            query_len=len(query),
        ) as event:
            search_filter = _metadata_filter(
                strategy_name=strategy_name,
                filename_filter=filename_filter,
                doc_id_filter=doc_id_filter,
            )
            vectorstore: Any | None = None
            try:
                with stage_timer(
                    self.config.run_dir,
                    "retriever.vectorstore.init",
                    request_id=request_id,
                    strategy_name=strategy_name,
                    filename_filter=filename_filter,
                    doc_id_filter=doc_id_filter,
                ):
                    vectorstore = self._vectorstore()
                rows = vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=top_k,
                    filter=search_filter or None,
                )
                documents = [_with_named_score(document, "_vector_score", score) for document, score in rows]
                event["backend"] = "relevance_scores"
            except Exception:
                try:
                    if vectorstore is None:
                        with stage_timer(
                            self.config.run_dir,
                            "retriever.vectorstore.init",
                            request_id=request_id,
                            strategy_name=strategy_name,
                            filename_filter=filename_filter,
                            doc_id_filter=doc_id_filter,
                        ):
                            vectorstore = self._vectorstore()
                    rows = vectorstore.similarity_search_with_score(
                        query,
                        k=top_k,
                        filter=search_filter or None,
                    )
                    documents = [_with_named_score(document, "_vector_score", _score_from_distance(score)) for document, score in rows]
                    event["backend"] = "distance_scores"
                except Exception:
                    documents = self._native_chroma_retrieve(
                        query=query,
                        top_k=top_k,
                        strategy_name=strategy_name,
                        filename_filter=filename_filter,
                        doc_id_filter=doc_id_filter,
                        include_scores=True,
                        score_key="_vector_score",
                        request_id=request_id,
                    )
                    event["backend"] = "native_chroma_fallback"
            for rank, document in enumerate(documents, start=1):
                metadata = getattr(document, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["_vector_rank"] = rank
            event["returned_docs"] = len(documents)
            return documents

    @optional_traceable(name="retriever.retrieve_representative", run_type="retriever")
    def retrieve_representative(
        self,
        filename: str,
        strategy_name: str,
        top_k: int,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.retrieve_representative",
            request_id=request_id,
            filename=filename,
            strategy_name=strategy_name,
            top_k=top_k,
        ) as event:
            run_id, collection_name = self._summary_cache_namespace()
            max_chunks = _summary_coverage_budget(top_k)
            try:
                coverage_candidates = self._retrieve_summary_coverage_candidates(
                    strategy_name=strategy_name,
                    filename_filter=filename,
                    request_id=request_id,
                )
            except Exception:
                coverage_candidates = []
            event["coverage_candidates"] = len(coverage_candidates)
            if coverage_candidates:
                selection_key = _summary_selection_cache_key(
                    run_id=run_id,
                    collection_name=collection_name,
                    strategy_name=strategy_name,
                    filename_filter=filename,
                    max_chunks=max_chunks,
                )
                cached_doc_keys = _summary_selection_cache_get(selection_key)
                if cached_doc_keys:
                    documents = _restore_selected_documents(coverage_candidates, cached_doc_keys)
                    if documents:
                        enrich_hwp_documents_with_viewer_pages(self.config.project_root, documents)
                        event["backend"] = "coverage_selection_cache"
                        event["selection_cache_hit"] = True
                        event["returned_docs"] = len(documents)
                        return documents

                documents = _select_summary_coverage_documents(
                    coverage_candidates,
                    max_chunks=max_chunks,
                )
                enrich_hwp_documents_with_viewer_pages(self.config.project_root, documents)
                _summary_selection_cache_put(
                    selection_key,
                    tuple(_document_key(document) for document in documents),
                )
                event["backend"] = "coverage_selection"
                event["selection_cache_hit"] = False
                event["returned_docs"] = len(documents)
                return documents

            query = _summary_representative_query(filename)
            candidate_k = min(80, max(top_k * 4, top_k + 12))
            candidates = self._retrieve_summary_candidates(
                query=query,
                strategy_name=strategy_name,
                top_k=candidate_k,
                filename_filter=filename,
                request_id=request_id,
            )
            documents = rerank_summary_documents(candidates, top_k=top_k)
            enrich_hwp_documents_with_viewer_pages(self.config.project_root, documents)
            event["backend"] = "rerank_candidates"
            event["candidate_docs"] = len(candidates)
            event["returned_docs"] = len(documents)
            return documents

    def _summary_cache_namespace(self) -> tuple[str, str]:
        latest = load_latest_chroma_index(self.config)
        run_id = str(latest.get("run_id") or "")
        collection_name = str(resolve_collection_name(latest) or "")
        return run_id, collection_name

    def _as_langchain_retriever(
        self,
        top_k: int,
        strategy_name: str,
        filename_filter: str | None,
        doc_id_filter: str | None = None,
        request_id: str | None = None,
    ) -> Any:
        with stage_timer(
            self.config.run_dir,
            "retriever.vectorstore.init",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
            doc_id_filter=doc_id_filter,
        ):
            vectorstore = self._vectorstore()
        search_filter = _metadata_filter(
            strategy_name=strategy_name,
            filename_filter=filename_filter,
            doc_id_filter=doc_id_filter,
        )
        search_kwargs: dict[str, Any] = {"k": top_k}
        if search_filter:
            search_kwargs["filter"] = search_filter
        return vectorstore.as_retriever(search_kwargs=search_kwargs)

    def _retrieve_summary_candidates(
        self,
        query: str,
        strategy_name: str,
        top_k: int,
        filename_filter: str,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.summary_candidates",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
            top_k=top_k,
            query_len=len(query),
        ) as event:
            search_filter = _metadata_filter(strategy_name=strategy_name, filename_filter=filename_filter)
            try:
                with stage_timer(
                    self.config.run_dir,
                    "retriever.vectorstore.init",
                    request_id=request_id,
                    strategy_name=strategy_name,
                    filename_filter=filename_filter,
                ):
                    vectorstore = self._vectorstore()
                rows = vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=top_k,
                    filter=search_filter or None,
                )
                documents = [_with_summary_score(document, score) for document, score in rows]
                event["backend"] = "relevance_scores"
                event["returned_docs"] = len(documents)
                return documents
            except Exception:
                try:
                    rows = vectorstore.similarity_search_with_score(
                        query,
                        k=top_k,
                        filter=search_filter or None,
                    )
                    documents = [_with_summary_score(document, _score_from_distance(score)) for document, score in rows]
                    event["backend"] = "distance_scores"
                    event["returned_docs"] = len(documents)
                    return documents
                except Exception:
                    documents = self._native_chroma_retrieve(
                        query=query,
                        top_k=top_k,
                        strategy_name=strategy_name,
                        filename_filter=filename_filter,
                        include_scores=True,
                        request_id=request_id,
                    )
                    event["backend"] = "native_chroma_fallback"
                    event["returned_docs"] = len(documents)
                    return documents

    def _native_chroma_retrieve(
        self,
        query: str,
        top_k: int,
        strategy_name: str,
        filename_filter: str | None,
        include_scores: bool,
        doc_id_filter: str | None = None,
        score_key: str = "_summary_raw_score",
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.native_chroma",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
            doc_id_filter=doc_id_filter,
            top_k=top_k,
            include_scores=include_scores,
            query_len=len(query),
        ) as event:
            latest = load_latest_chroma_index(self.config)
            persist_dir = resolve_chroma_persist_dir(self.config, latest)
            collection_name = resolve_collection_name(latest)
            search_filter = _metadata_filter(
                strategy_name=strategy_name,
                filename_filter=filename_filter,
                doc_id_filter=doc_id_filter,
            )
            embeddings = _openai_embeddings(self.config)
            with stage_timer(
                self.config.run_dir,
                "retriever.native_chroma.embed_query",
                request_id=request_id,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
                doc_id_filter=doc_id_filter,
                query_len=len(query),
            ):
                query_embedding = embeddings.embed_query(query)

            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("chromadb is required for native Chroma retrieval fallback.") from exc

            collection = _chroma_collection(str(persist_dir), collection_name)
            with stage_timer(
                self.config.run_dir,
                "retriever.native_chroma.collection_query",
                request_id=request_id,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
                doc_id_filter=doc_id_filter,
                top_k=top_k,
            ):
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=search_filter or None,
                    include=["metadatas", "documents", "distances"],
                )
            documents = _documents_from_chroma_result(result, include_scores=include_scores, score_key=score_key)
            event["returned_docs"] = len(documents)
            return documents

    def _retrieve_summary_coverage_candidates(
        self,
        strategy_name: str,
        filename_filter: str,
        request_id: str | None = None,
    ) -> list[Any]:
        with stage_timer(
            self.config.run_dir,
            "retriever.summary_coverage_candidates",
            request_id=request_id,
            strategy_name=strategy_name,
            filename_filter=filename_filter,
        ) as event:
            latest = load_latest_chroma_index(self.config)
            run_id = str(latest.get("run_id") or "")
            persist_dir = resolve_chroma_persist_dir(self.config, latest)
            collection_name = str(resolve_collection_name(latest) or "")
            search_filter = _metadata_filter(strategy_name=strategy_name, filename_filter=filename_filter)
            coverage_key = _summary_coverage_cache_key(
                run_id=run_id,
                collection_name=collection_name,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
            )
            cached_payloads = _summary_coverage_cache_get(coverage_key)
            if cached_payloads is not None:
                documents = _documents_from_cached_payloads(cached_payloads)
                event["cache_hit"] = True
                event["returned_docs"] = len(documents)
                return documents

            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("chromadb is required for summary coverage retrieval.") from exc

            collection = _chroma_collection(str(persist_dir), collection_name)
            with stage_timer(
                self.config.run_dir,
                "retriever.summary_coverage_get",
                request_id=request_id,
                strategy_name=strategy_name,
                filename_filter=filename_filter,
            ):
                result = collection.get(
                    where=search_filter or None,
                    include=["metadatas", "documents"],
                )
            documents = _documents_from_chroma_get_result(result)
            _summary_coverage_cache_put(coverage_key, _documents_to_cached_payloads(documents))
            event["cache_hit"] = False
            event["returned_docs"] = len(documents)
            return documents

    def _vectorstore(self) -> Any:
        latest = load_latest_chroma_index(self.config)
        persist_dir = resolve_chroma_persist_dir(self.config, latest)
        collection_name = resolve_collection_name(latest)
        return _cached_vectorstore(
            persist_directory=str(persist_dir),
            collection_name=collection_name,
            embedding_model=self.config.embedding_model,
            embedding_dimensions=self.config.embedding_dimensions,
            openai_api_key=self.config.openai_api_key,
        )


def _summary_coverage_cache_key(
    run_id: str,
    collection_name: str,
    strategy_name: str,
    filename_filter: str,
) -> tuple[str, str, str, str]:
    return (run_id, collection_name, strategy_name, filename_filter)


def _summary_selection_cache_key(
    run_id: str,
    collection_name: str,
    strategy_name: str,
    filename_filter: str,
    max_chunks: int,
) -> tuple[str, str, str, str, int]:
    return (run_id, collection_name, strategy_name, filename_filter, int(max_chunks))


def _summary_coverage_cache_get(key: tuple[str, str, str, str]) -> tuple[dict[str, Any], ...] | None:
    with _summary_cache_lock:
        payloads = _summary_coverage_cache.get(key)
        if payloads is None:
            return None
        _summary_coverage_cache.move_to_end(key)
        return deepcopy(payloads)


def _summary_coverage_cache_put(key: tuple[str, str, str, str], payloads: tuple[dict[str, Any], ...]) -> None:
    with _summary_cache_lock:
        _summary_coverage_cache[key] = deepcopy(payloads)
        _summary_coverage_cache.move_to_end(key)
        while len(_summary_coverage_cache) > SUMMARY_COVERAGE_CACHE_SIZE:
            _summary_coverage_cache.popitem(last=False)


def _summary_selection_cache_get(key: tuple[str, str, str, str, int]) -> tuple[str, ...] | None:
    with _summary_cache_lock:
        chunk_ids = _summary_selection_cache.get(key)
        if chunk_ids is None:
            return None
        _summary_selection_cache.move_to_end(key)
        return tuple(chunk_ids)


def _summary_selection_cache_put(key: tuple[str, str, str, str, int], chunk_ids: tuple[str, ...]) -> None:
    with _summary_cache_lock:
        _summary_selection_cache[key] = tuple(chunk_ids)
        _summary_selection_cache.move_to_end(key)
        while len(_summary_selection_cache) > SUMMARY_SELECTION_CACHE_SIZE:
            _summary_selection_cache.popitem(last=False)


def _documents_to_cached_payloads(documents: list[Any]) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for document in documents:
        payloads.append(
            {
                "page_content": _document_text(document),
                "metadata": deepcopy(getattr(document, "metadata", {}) or {}),
            }
        )
    return tuple(payloads)


def _documents_from_cached_payloads(payloads: tuple[dict[str, Any], ...]) -> list[Any]:
    documents: list[Any] = []
    for payload in payloads:
        metadata = deepcopy(payload.get("metadata", {}) or {})
        documents.append(
            _make_document(
                page_content=str(payload.get("page_content", "") or ""),
                metadata=metadata,
            )
        )
    return documents


def _restore_selected_documents(coverage_candidates: list[Any], cached_doc_keys: tuple[str, ...]) -> list[Any]:
    by_key = {_document_key(document): document for document in coverage_candidates}
    restored: list[Any] = []
    for key in cached_doc_keys:
        document = by_key.get(key)
        if document is not None:
            restored.append(document)
    return restored


def _metadata_filter(
    strategy_name: str,
    filename_filter: str | None,
    doc_id_filter: str | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    if strategy_name:
        filters.append({"strategy_name": strategy_name})
    if filename_filter:
        filters.append({"filename": filename_filter})
    if doc_id_filter:
        filters.append({"doc_id": doc_id_filter})
    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _summary_representative_query(filename: str) -> str:
    return (
        "Find representative explanatory body chunks for a document-wide summary. "
        "Prefer broad business context, major claims, key results, risks, and conclusions. "
        "Avoid table of contents, appendix, title-only, chart-only, image-only, caption-only, or number-only chunks. "
        f"Document: {filename}"
    )


def _summary_coverage_budget(requested_top_k: int) -> int:
    return min(SUMMARY_MAX_COVERAGE_CHUNKS, max(SUMMARY_MIN_COVERAGE_CHUNKS, int(requested_top_k) * 3))


def _select_summary_coverage_documents(documents: list[Any], max_chunks: int) -> list[Any]:
    candidates = [_with_coverage_score(document) for document in documents if _document_text(document)]
    if not candidates:
        return []

    by_page: dict[int, list[tuple[Any, float]]] = {}
    for document, score in candidates:
        page = _document_page(document)
        by_page.setdefault(page, []).append((document, score))
    pages = sorted(page for page in by_page if page > 0)
    if not pages:
        pages = sorted(by_page)

    selected: list[Any] = []
    seen_ids: set[str] = set()
    page_budget = min(len(pages), max_chunks)
    coverage_pages = _choose_coverage_pages(by_page, pages, page_budget)
    for page in coverage_pages:
        document = _best_document_for_page(by_page.get(page, []), seen_ids)
        if document is not None:
            _mark_summary_selection(document, "page_coverage")
            selected.append(document)
            seen_ids.add(_document_key(document))

    remaining = max_chunks - len(selected)
    if remaining > 0:
        page_priorities = {page: _page_coverage_priority(by_page.get(page, [])) for page in by_page}
        fill_candidates = sorted(
            candidates,
            key=lambda item: (
                _coverage_fill_priority(
                    item[0],
                    item[1],
                    page_priorities.get(_document_page(item[0]), 0.0),
                ),
                -_document_page(item[0]),
            ),
            reverse=True,
        )
        page_counts = _page_counts(selected)
        for document, _score in fill_candidates:
            if len(selected) >= max_chunks:
                break
            key = _document_key(document)
            if key in seen_ids:
                continue
            page = _document_page(document)
            if page_counts.get(page, 0) >= 2:
                continue
            if not _is_fill_worthy_coverage_document(document, _score, page_counts.get(page, 0)):
                continue
            _mark_summary_selection(document, "importance_fill")
            selected.append(document)
            seen_ids.add(key)
            page_counts[page] = page_counts.get(page, 0) + 1

    return sorted(selected, key=lambda document: (_document_page(document), _document_chunk_index(document)))


def _choose_coverage_pages(
    by_page: dict[int, list[tuple[Any, float]]],
    pages: list[int],
    page_budget: int,
) -> list[int]:
    if page_budget <= 0:
        return []
    if len(pages) <= page_budget:
        return pages
    page_scores = {page: _page_coverage_priority(by_page.get(page, [])) for page in pages}
    if not page_scores:
        return pages[:page_budget]

    best_page_score = max(page_scores.values(), default=0.0)
    anchor_budget = min(page_budget, max(1, round(page_budget * 0.4)))
    anchor_pool = [
        page
        for page in pages
        if page_scores.get(page, 0.0) >= max(0.45, best_page_score * 0.55)
    ] or pages
    selected = set(_spread_pages_by_quality(anchor_pool, page_scores, anchor_budget))

    page_positions = {page: index for index, page in enumerate(pages)}
    quality_floor = max(0.35, best_page_score * 0.35)
    while len(selected) < page_budget:
        best_page: int | None = None
        best_value = float("-inf")
        for page in pages:
            if page in selected:
                continue
            page_score = page_scores.get(page, 0.0)
            if selected and page_score < quality_floor:
                continue
            min_distance = min(
                abs(page_positions[page] - page_positions[selected_page]) for selected_page in selected
            ) if selected else len(pages)
            spread_bonus = min(0.18, float(min_distance) * 0.04)
            value = page_score + spread_bonus
            if value > best_value:
                best_value = value
                best_page = page
        if best_page is None:
            break
        selected.add(best_page)

    return sorted(selected)


def _best_document_for_page(page_items: list[tuple[Any, float]], seen_ids: set[str]) -> Any | None:
    available = [(document, score) for document, score in page_items if _document_key(document) not in seen_ids]
    if not available:
        return None
    available.sort(key=lambda item: item[1], reverse=True)
    return available[0][0]


def _with_coverage_score(document: Any) -> tuple[Any, float]:
    score = _coverage_score(document)
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        metadata["_summary_coverage_score"] = round(score, 6)
    return document, score


def _coverage_score(document: Any) -> float:
    metadata = getattr(document, "metadata", {}) or {}
    text = _document_text(document)
    normalized_text = _normalize_coverage_text(text)
    chunk_type = str(metadata.get("chunk_type") or "").lower()
    visual_type = str(metadata.get("visual_type") or "").lower()
    score = 0.18

    if chunk_type in {"page", "native_page"}:
        score += 0.42
    elif chunk_type == "block_group":
        score += 0.38
    elif chunk_type == "text":
        score += 0.28
    elif chunk_type == "table_summary":
        score += 0.12 if _is_meaningful_visual_summary(text) else -0.08
    elif chunk_type == "chart_summary":
        score += 0.06 if _is_meaningful_visual_summary(text) else -0.12
    elif chunk_type in {"title", "image"}:
        score -= 0.28

    if visual_type == "table":
        score += 0.05 if _is_meaningful_visual_summary(text) else -0.06
    elif visual_type in {"chart", "chart_like_page"}:
        score += 0.03 if _is_meaningful_visual_summary(text) else -0.08

    text_length = len(text)
    if text_length >= 360:
        score += 0.18
    elif text_length >= 220:
        score += 0.13
    elif text_length >= 140:
        score += 0.08
    elif text_length >= 90:
        score += 0.03
    elif text_length < 60:
        score -= 0.28
    else:
        score -= 0.12

    sentence_count = _sentence_like_count(text)
    metric_signal = _has_metric_signal(text)
    explanatory = _looks_explanatory_summary_text(text)
    metric_and_explanation = metric_signal and explanatory

    if explanatory:
        score += 0.18
    if sentence_count >= 3:
        score += 0.08
    elif sentence_count >= 2:
        score += 0.04
    if metric_signal:
        score += 0.08
    if metric_and_explanation:
        score += 0.16

    if _has_dense_body_text(text):
        score += 0.08
    if _looks_title_only(text):
        score -= 0.24
    if _is_numeric_heavy_text(text):
        score -= 0.22
    if _has_boilerplate_noise(normalized_text):
        score -= 0.34
    if _has_contact_or_distribution_noise(normalized_text):
        score -= 0.26
    if _looks_sparse_list_text(text):
        score -= 0.12

    return score


def _has_metric_signal(text: str) -> bool:
    return bool(
        re.search(
            r"(?:\d[\d,.\-]*\s*(?:%|bp|bps|x|배|건|명|원|달러|usd|krw|억원|백만원|천만원|million|billion|mn|bn))"
            r"|(?:roe|bps|eps|ebitda|cagr|margin|growth|revenue|sales|profit|income|cash flow|guidance)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_explanatory_summary_text(text: str) -> bool:
    if len(text) < 140:
        return False
    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    sentence_count = _sentence_like_count(text)
    return sentence_count >= 2 or len(tokens) >= 18 or len(text) >= 280


def _normalize_coverage_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _sentence_like_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"(?:[.!?]|(?<=[다요음임]))\s+|\n+", text) if part.strip()]
    return len(parts)


def _has_dense_body_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    long_lines = sum(1 for line in lines if len(line) >= 50)
    return long_lines >= 2 or (len(text) >= 260 and len(lines) <= max(4, len(text) // 120))


def _looks_title_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    tokens = [token for token in re.split(r"\s+", stripped) if token]
    if len(stripped) > 110 or len(lines) > 3 or len(tokens) > 14:
        return False
    return _sentence_like_count(stripped) <= 1 and not _has_metric_signal(stripped)


def _is_numeric_heavy_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 40:
        return False
    digit_count = sum(char.isdigit() for char in compact)
    letter_count = len(re.findall(r"[A-Za-z가-힣]", compact))
    symbol_count = len(re.findall(r"[%$€£¥₩/,:;()\-+]", compact))
    return digit_count >= max(14, letter_count) or (digit_count + symbol_count) > max(24, int(len(compact) * 0.55))


def _looks_sparse_list_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    short_lines = sum(1 for line in lines if len(line) <= 24)
    bullet_lines = sum(1 for line in lines if re.match(r"^[\-\u2022\d]+[.)]?\s*", line))
    return short_lines >= max(3, len(lines) - 1) or bullet_lines >= max(2, len(lines) // 2)


def _has_boilerplate_noise(normalized_text: str) -> bool:
    patterns = (
        "contents",
        "table of contents",
        "appendix",
        "appendices",
        "annex",
        "annexes",
        "disclaimer",
        "compliance notice",
        "legal notice",
        "safe harbor",
        "forward-looking",
        "all rights reserved",
        "confidential",
        "목차",
        "부록",
        "별첨",
        "면책",
        "유의사항",
        "공표일",
        "참고자료",
    )
    return any(pattern in normalized_text for pattern in patterns)


def _has_contact_or_distribution_noise(normalized_text: str) -> bool:
    patterns = (
        "analyst",
        "contact",
        "contacts",
        "investor relations",
        "ir contact",
        "e-mail",
        "email",
        "tel",
        "telephone",
        "phone",
        "fax",
        "www.",
        "@",
        "주소",
        "전화",
        "팩스",
        "이메일",
        "담당자",
        "문의",
    )
    phone_like = re.search(r"(?:\+\d{1,3}[\s\-]?)?(?:\d{2,4}[\s\-]?){2,}\d{3,4}", normalized_text)
    return any(pattern in normalized_text for pattern in patterns) or bool(phone_like)


def _is_meaningful_visual_summary(text: str) -> bool:
    return (
        len(text) >= 120
        and not _looks_title_only(text)
        and not _is_numeric_heavy_text(text)
        and not _has_boilerplate_noise(_normalize_coverage_text(text))
        and (_looks_explanatory_summary_text(text) or (_has_metric_signal(text) and _sentence_like_count(text) >= 2))
    )


def _page_coverage_priority(page_items: list[tuple[Any, float]]) -> float:
    if not page_items:
        return 0.0
    sorted_items = sorted(page_items, key=lambda item: item[1], reverse=True)
    top_scores = [score for _document, score in sorted_items[:3]]
    explanatory_docs = sum(
        1 for document, score in sorted_items[:3] if score >= 0.45 and _looks_explanatory_summary_text(_document_text(document))
    )
    strong_docs = sum(1 for _document, score in sorted_items[:3] if score >= 0.65)
    priority = 0.0
    if top_scores:
        priority += top_scores[0] * 0.72
    if len(top_scores) > 1:
        priority += top_scores[1] * 0.2
    if len(top_scores) > 2:
        priority += top_scores[2] * 0.08
    priority += min(0.12, explanatory_docs * 0.05)
    priority += min(0.1, strong_docs * 0.04)
    return priority


def _spread_pages_by_quality(candidate_pages: list[int], page_scores: dict[int, float], budget: int) -> list[int]:
    if budget <= 0 or not candidate_pages:
        return []
    if len(candidate_pages) <= budget:
        return list(candidate_pages)

    selected: list[int] = []
    for bucket_index in range(budget):
        start = round(bucket_index * len(candidate_pages) / budget)
        end = round((bucket_index + 1) * len(candidate_pages) / budget)
        bucket = candidate_pages[start:end] or candidate_pages[start : start + 1]
        best_page = max(bucket, key=lambda page: (page_scores.get(page, 0.0), -page))
        if best_page not in selected:
            selected.append(best_page)

    if len(selected) < budget:
        for page in sorted(candidate_pages, key=lambda item: (page_scores.get(item, 0.0), -item), reverse=True):
            if page in selected:
                continue
            selected.append(page)
            if len(selected) >= budget:
                break
    return selected


def _coverage_fill_priority(document: Any, score: float, page_priority: float) -> float:
    priority = score + min(0.18, page_priority * 0.12)
    if _is_visual_like_coverage_document(document) and not _is_meaningful_visual_summary(_document_text(document)):
        priority -= 0.24
    return priority


def _is_fill_worthy_coverage_document(document: Any, score: float, page_count: int) -> bool:
    text = _document_text(document)
    if score < 0.35:
        return False
    if _has_boilerplate_noise(_normalize_coverage_text(text)) or _has_contact_or_distribution_noise(
        _normalize_coverage_text(text)
    ):
        return False
    if _is_visual_like_coverage_document(document) and not _is_meaningful_visual_summary(text):
        return False
    if page_count >= 1 and not (_looks_explanatory_summary_text(text) or (_has_metric_signal(text) and not _is_numeric_heavy_text(text))):
        return False
    return True


def _is_visual_like_coverage_document(document: Any) -> bool:
    metadata = getattr(document, "metadata", {}) or {}
    chunk_type = str(metadata.get("chunk_type") or "").lower()
    visual_type = str(metadata.get("visual_type") or "").lower()
    return chunk_type in {"table_summary", "chart_summary", "image"} or visual_type in {"table", "chart", "chart_like_page"}


def _document_text(document: Any) -> str:
    return str(getattr(document, "page_content", "") or "").strip()


def _document_page(document: Any) -> int:
    metadata = getattr(document, "metadata", {}) or {}
    return preferred_page_num(metadata)


def _document_chunk_index(document: Any) -> int:
    metadata = getattr(document, "metadata", {}) or {}
    try:
        return int(metadata.get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


def _document_key(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    for key in ("vector_record_id", "chunk_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return f"{_document_page(document)}:{_document_chunk_index(document)}:{hash(_document_text(document))}"


def _page_counts(documents: list[Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for document in documents:
        page = _document_page(document)
        counts[page] = counts.get(page, 0) + 1
    return counts


def _mark_summary_selection(document: Any, reason: str) -> None:
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        metadata["_summary_selection_reason"] = reason


def _with_summary_score(document: Any, score: float) -> Any:
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        metadata["_summary_raw_score"] = float(score)
    return document


def _score_from_distance(distance: float) -> float:
    try:
        return 1.0 - (float(distance) / 2.0)
    except (TypeError, ValueError):
        return 0.0


def _documents_from_chroma_result(
    result: dict[str, Any],
    include_scores: bool,
    score_key: str = "_summary_raw_score",
) -> list[Any]:
    ids = _first_result_list(result.get("ids"))
    metadatas = _first_result_list(result.get("metadatas"))
    documents = _first_result_list(result.get("documents"))
    distances = _first_result_list(result.get("distances"))
    output: list[Any] = []
    for index, raw_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        metadata = dict(metadata)
        if raw_id and not metadata.get("vector_record_id"):
            metadata["vector_record_id"] = str(raw_id)
        if include_scores and index < len(distances):
            metadata[score_key] = _score_from_distance(distances[index])
        page_content = str(documents[index] if index < len(documents) else "")
        output.append(_make_document(page_content=page_content, metadata=metadata))
    return output


def _documents_from_chroma_get_result(result: dict[str, Any]) -> list[Any]:
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or []
    output: list[Any] = []
    for index, raw_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        metadata = dict(metadata)
        if raw_id and not metadata.get("vector_record_id"):
            metadata["vector_record_id"] = str(raw_id)
        page_content = str(documents[index] if index < len(documents) else "")
        output.append(_make_document(page_content=page_content, metadata=metadata))
    return output


def _first_result_list(value: Any) -> list[Any]:
    if isinstance(value, list) and value:
        first = value[0]
        return first if isinstance(first, list) else value
    return []


def _make_document(page_content: str, metadata: dict[str, Any]) -> Any:
    try:
        from langchain_core.documents import Document

        return Document(page_content=page_content, metadata=metadata)
    except ImportError:
        return RetrievedDocument(page_content=page_content, metadata=metadata)


@dataclass
class RetrievedDocument:
    page_content: str
    metadata: dict[str, Any]


def _with_named_score(document: Any, score_key: str, score: float) -> Any:
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        metadata[score_key] = float(score)
    return document


def _fuse_rrf_documents(
    vector_documents: list[Any],
    keyword_documents: list[Any],
    limit: int,
    rrf_k: int,
) -> list[Any]:
    fused: dict[str, Any] = {}
    for source_name, documents in (("vector", vector_documents), ("keyword", keyword_documents)):
        for rank, document in enumerate(documents, start=1):
            key = _document_primary_key(document)
            if key not in fused:
                fused[key] = _make_document(
                    page_content=_document_text(document),
                    metadata=deepcopy(getattr(document, "metadata", {}) or {}),
                )
            fused_document = fused[key]
            metadata = getattr(fused_document, "metadata", None)
            source_metadata = getattr(document, "metadata", {}) or {}
            if not isinstance(metadata, dict):
                continue
            metadata["_hybrid_rrf_score"] = float(metadata.get("_hybrid_rrf_score") or 0.0) + (1.0 / (rrf_k + rank))
            metadata["_hybrid_sources"] = int(metadata.get("_hybrid_sources") or 0) + 1
            if source_name == "vector":
                metadata["_vector_rank"] = rank
                if source_metadata.get("_vector_score") is not None:
                    metadata["_vector_score"] = source_metadata.get("_vector_score")
            else:
                metadata["_keyword_rank"] = rank
                if source_metadata.get("_keyword_score") is not None:
                    metadata["_keyword_score"] = source_metadata.get("_keyword_score")
                if source_metadata.get("_keyword_overlap_tokens") is not None:
                    metadata["_keyword_overlap_tokens"] = source_metadata.get("_keyword_overlap_tokens")

    ranked = sorted(
        fused.values(),
        key=lambda document: (
            float((getattr(document, "metadata", {}) or {}).get("_hybrid_rrf_score") or 0.0),
            int((getattr(document, "metadata", {}) or {}).get("_hybrid_sources") or 0),
            -int((getattr(document, "metadata", {}) or {}).get("_vector_rank") or 10_000),
            -int((getattr(document, "metadata", {}) or {}).get("_keyword_rank") or 10_000),
        ),
        reverse=True,
    )[:limit]

    for rank, document in enumerate(ranked, start=1):
        metadata = getattr(document, "metadata", None)
        if isinstance(metadata, dict):
            metadata["_hybrid_rank"] = rank
    return ranked


def _document_primary_key(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    for key in ("vector_record_id", "chunk_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return f"{_document_page(document)}:{_document_chunk_index(document)}:{hash(_document_text(document))}"


def _serialize_documents(documents: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for document in documents[:limit]:
        metadata = getattr(document, "metadata", {}) or {}
        output.append(
            {
                "vector_record_id": str(metadata.get("vector_record_id") or ""),
                "chunk_id": str(metadata.get("chunk_id") or ""),
                "doc_id": str(metadata.get("doc_id") or ""),
                "filename": str(metadata.get("filename") or ""),
                "page_num": _document_page(document),
                "chunk_type": str(metadata.get("chunk_type") or ""),
                "vector_score": _rounded_number(metadata.get("_vector_score")),
                "keyword_score": _rounded_number(metadata.get("_keyword_score")),
                "hybrid_rrf_score": _rounded_number(metadata.get("_hybrid_rrf_score")),
                "qa_rank_score": _rounded_number(metadata.get("_qa_rank_score")),
                "is_boilerplate": bool(metadata.get("_qa_is_boilerplate")) if "_qa_is_boilerplate" in metadata else is_qa_boilerplate_document(document),
                "preview": _document_text(document)[:180],
            }
        )
    return output


def _rounded_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return None


def _expected_document_hit(documents: list[Any], filename_filter: str | None, doc_id_filter: str | None) -> bool | None:
    if not filename_filter and not doc_id_filter:
        return None
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        if filename_filter and str(metadata.get("filename") or "") == filename_filter:
            return True
        if doc_id_filter and str(metadata.get("doc_id") or "") == doc_id_filter:
            return True
    return False


def _expected_page_from_query(query: str) -> int | None:
    patterns = (
        r"\bpage\s*(\d{1,4})\b",
        r"\bp\.\s*(\d{1,4})\b",
        r"(\d{1,4})\s*페이지",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _expected_page_hit(documents: list[Any], expected_page: int | None) -> bool | None:
    if expected_page is None:
        return None
    return any(_document_page(document) == expected_page for document in documents)


def _boilerplate_count(documents: list[Any], limit: int | None = None) -> int:
    scope = documents if limit is None else documents[:limit]
    return sum(1 for document in scope if is_qa_boilerplate_document(document))


def _openai_embeddings(config: RagApiConfig) -> Any:
    try:
        return _cached_openai_embeddings(
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            api_key=config.openai_api_key,
        )
    except ImportError as exc:
        raise RuntimeError("langchain-openai is required for OpenAI embeddings.") from exc


def _chroma_class() -> Any:
    try:
        return _cached_chroma_class()
    except ImportError as exc:
        raise RuntimeError("langchain-chroma or langchain-community is required for Chroma retrieval.") from exc


@lru_cache(maxsize=8)
def _cached_openai_embeddings(model: str, dimensions: int, api_key: str) -> Any:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        dimensions=dimensions,
        api_key=api_key,
    )


@lru_cache(maxsize=4)
def _cached_chroma_class() -> Any:
    try:
        from langchain_chroma import Chroma

        return Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

        return Chroma


@lru_cache(maxsize=16)
def _cached_vectorstore(
    persist_directory: str,
    collection_name: str | None,
    embedding_model: str,
    embedding_dimensions: int,
    openai_api_key: str,
) -> Any:
    chroma_cls = _chroma_class()
    embeddings = _cached_openai_embeddings(
        model=embedding_model,
        dimensions=embedding_dimensions,
        api_key=openai_api_key,
    )
    kwargs: dict[str, Any] = {
        "persist_directory": persist_directory,
        "embedding_function": embeddings,
    }
    if collection_name:
        kwargs["collection_name"] = collection_name
    return chroma_cls(**kwargs)


@lru_cache(maxsize=8)
def _persistent_chroma_client(path: str) -> Any:
    import chromadb

    return chromadb.PersistentClient(path=path)


@lru_cache(maxsize=16)
def _chroma_collection(path: str, collection_name: str | None) -> Any:
    client = _persistent_chroma_client(path)
    return client.get_collection(collection_name) if collection_name else client.list_collections()[0]
